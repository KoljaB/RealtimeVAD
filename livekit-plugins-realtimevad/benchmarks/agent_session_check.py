"""End-of-turn and interruption timing inside a real LiveKit ``AgentSession``, three VADs, same audio.

``compare_silero.py`` compares the plugins at the VAD-stream level. This script puts each VAD
into an ``AgentSession`` (livekit-agents 1.8.11) so that LiveKit's own end-of-turn,
interruption, pause and false-interruption logic runs on the VAD events:

    AgentSession(vad=<vad>, stt=FakeSTT, llm=FakeLLM, tts=FakeTTS,
                 turn_handling={"turn_detection": "vad",
                                "endpointing": {"min_delay": <0.5 | 0.3>},
                                "interruption": {"mode": "vad", "min_duration": 0.5}},
                 aec_warmup_duration=None)

with LiveKit's test fakes from ``.livekit-src/agents/tests`` (FakeAudioInput, FakeAudioOutput
with ``can_pause=True`` behind a TranscriptSynchronizer, as in ``tests/fake_session.py``).
Real audio goes in as 20 ms / 16 kHz frames at real-time pace; frame i (covering
[0.02 i, 0.02 (i+1)) s) is pushed at t0 + 0.02 (i+1), so ``time.time() - t0`` is audio time.
The fake STT is scripted from the clip's true speech span (``lab.reference``) and sends its
interim at ``off + STT_DELAY/2`` and its final at ``off + STT_DELAY``; that is earlier than any
VAD END, so the VAD, not the STT, decides when the turn ends.

Scenarios (6 ``speech`` clips, the shortest one of each of 6 languages; 6 short backchannels):
  A  end of turn, agent idle, min_delay 0.5 and 0.3: true end of speech -> user state leaves
     "speaking", agent "thinking" (turn committed), first agent audio
  B  the agent speaks a 12 s reply when the user starts a sentence: true onset -> agent paused or
     interrupted, and whether the user's turn is then answered
  C  the same with one real backchannel (< 0.45 s of speech, no crosstalk) and an empty STT
     transcript (a final transcript would interrupt the agent on its own, whatever the VAD
     does): paused? resumed (false-interruption path)? interrupted?
  D  no runtime: LiveKit's ``AudioRecognition._check_vad_silence_requirement`` against its default
     streaming turn detector (``inference.TurnDetector()``, constructed without a connection)

All sessions of one clip and scenario run concurrently in one event loop (one per VAD and
min_delay), each with its own audio pump.

    python benchmarks/agent_session_check.py out.json

Needs this plugin and RealtimeVAD importable, and a checkout of github.com/livekit/agents (the tag of
the installed livekit-agents) at ``.livekit-src/agents`` in the repository root for its test fakes.
"""

import asyncio
import contextlib
import ctypes
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("REALTIMEVAD_ROOT") or Path(__file__).resolve().parents[2])  # a git worktree has no audio
sys.path.insert(0, str(ROOT / "tests" / "audio"))
sys.path.insert(0, str(ROOT / ".livekit-src" / "agents"))  # LiveKit's tests/ as package "tests"

import lab  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents import Agent, AgentSession, inference  # noqa: E402
from livekit.agents.vad import VADEventType  # noqa: E402
from livekit.agents.voice.audio_recognition import AudioRecognition  # noqa: E402
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer  # noqa: E402
from livekit.plugins import realtimevad, silero  # noqa: E402
from tests.fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput  # noqa: E402
from tests.fake_llm import FakeLLM, FakeLLMResponse  # noqa: E402
from tests.fake_stt import FakeSTT, FakeUserSpeech  # noqa: E402
from tests.fake_tts import FakeTTS, FakeTTSResponse  # noqa: E402

SR, FRAME = 16000, 320
STT_DELAY = 0.05  # final transcript 50 ms after the true end of speech, interim at 25 ms
LANGS = ("cmn-Hans-CN", "de-DE", "en-US", "fr-FR", "hi-IN", "ru-RU")
VADS = {
    "silero default (0.55 s)": lambda: silero.VAD.load(),
    "silero 0.25 s": lambda: silero.VAD.load(min_silence_duration=0.25),
    "realtimevad default (0.25 s)": lambda: realtimevad.VAD.load(),
}
LONG_REPLY = "Let me tell you a long story about the history of saddles and stirrups in Europe."
SHORT_REPLY = "Okay."
LONG_REPLY_S, SHORT_REPLY_S = 12.0, 1.0


# -- recording ----------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.t0 = None
        self.events = []

    def __call__(self, kind, value=None, **extra):
        if self.t0 is None:  # before the first frame: keep, time it later as negative
            t = -1.0
        else:
            t = time.time() - self.t0
        self.events.append({"t": round(t, 4), "kind": kind, "value": value, **extra})


class QuietText(FakeTextOutput):
    def flush(self):  # FakeTextOutput prints every agent transcript
        self._pushed_text = ""


class RecordingAudioOutput(FakeAudioOutput):
    def __init__(self, rec):
        super().__init__(can_pause=True)
        self._rec = rec

    def on_playback_started(self, *, created_at):
        self._rec("playback_started")
        super().on_playback_started(created_at=created_at)

    def pause(self):  # LiveKit calls pause() again on every INFERENCE_DONE while paused; log transitions
        if self._paused_at is None:
            self._rec("pause")
        super().pause()

    def resume(self):
        if self._paused_at is not None:
            self._rec("resume")
        super().resume()

    def clear_buffer(self):
        if self._pushed_duration:
            self._rec("clear_buffer", round(min(self._played_duration(), self._pushed_duration), 3))
        super().clear_buffer()


class StreamProxy:
    """Wraps the VAD stream the session opens to timestamp every START/END it hands over."""

    def __init__(self, inner, rec):
        self._inner, self._rec = inner, rec

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        ev = await self._inner.__anext__()
        if ev.type != VADEventType.INFERENCE_DONE:
            self._rec("vad_" + ev.type.value, round(ev.timestamp, 3), speech=round(ev.speech_duration, 3),
                      silence=round(ev.silence_duration, 3))
        elif ev.speech_duration >= 0.5 and not getattr(self, "_reached", False):
            self._reached = True  # first INFERENCE_DONE with speech_duration >= min_duration
            self._rec("vad_speech_0.5")
        if ev.type == VADEventType.END_OF_SPEECH:
            self._reached = False
        return ev


def wrap_vad(vad, rec):
    original = vad.stream

    def stream(*a, **k):
        return StreamProxy(original(*a, **k), rec)

    vad.stream = stream
    return vad


class LogCatcher(logging.Handler):
    """WARNING+ from livekit.* and the plugins, attributed to the runs that are active."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.active = []

    def emit(self, record):
        if "metrics_collected is deprecated" in record.getMessage():
            return  # caused by this script's own subscription
        msg = f"{record.name} {record.levelname}: {record.getMessage()}"
        if record.exc_info:
            msg += " | " + repr(record.exc_info[1])
        for rec in self.active:
            rec("log", msg[:400])


CATCHER = LogCatcher()


class CheckAgent(Agent):
    def __init__(self, rec, say=None):
        super().__init__(instructions="test")
        self._rec, self._say = rec, say

    async def on_enter(self):
        if self._say:
            self.session.say(self._say)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        self._rec("turn_committed", new_message.text_content)


# -- one session ----------------------------------------------------------------
async def run_session(make_vad, audio, speech, *, min_delay, say=None):
    """speech: (on, off, transcript) in audio time; returns the recorder's events."""
    rec = Recorder()
    on, off, text = speech
    # FakeSTT clocks from the first frame it receives (= audio time 0.02)
    speeches = [FakeUserSpeech(start_time=on - 0.02, end_time=off - 0.02, transcript=text, stt_delay=STT_DELAY)]
    llm = [FakeLLMResponse(input=text, content=SHORT_REPLY, ttft=0.1, duration=0.2)] if text else []
    tts = [FakeTTSResponse(input=SHORT_REPLY, audio_duration=SHORT_REPLY_S, ttfb=0.1, duration=0.2),
           FakeTTSResponse(input=LONG_REPLY, audio_duration=LONG_REPLY_S, ttfb=0.1, duration=0.3)]
    session = AgentSession(
        vad=wrap_vad(make_vad(), rec), stt=FakeSTT(fake_user_speeches=speeches), llm=FakeLLM(fake_responses=llm),
        tts=FakeTTS(fake_responses=tts),
        turn_handling={"turn_detection": "vad", "endpointing": {"min_delay": min_delay},
                       "interruption": {"mode": "vad", "min_duration": 0.5}},
        aec_warmup_duration=None)
    audio_in, audio_out = FakeAudioInput(), RecordingAudioOutput(rec)
    sync = TranscriptSynchronizer(next_in_chain_audio=audio_out, next_in_chain_text=QuietText())
    session.input.audio = audio_in
    session.output.audio = sync.audio_output
    session.output.transcription = sync.text_output
    session.on("user_state_changed", lambda ev: rec("user_state", ev.new_state))
    session.on("agent_state_changed", lambda ev: rec("agent_state", ev.new_state))
    session.on("agent_false_interruption", lambda ev: rec("false_interruption", ev.resumed))
    session.on("user_input_transcribed", lambda ev: rec("transcript_final" if ev.is_final else "transcript_interim"))
    session.on("metrics_collected", lambda ev: rec("eou_metrics", round(ev.metrics.end_of_utterance_delay, 4))
               if getattr(ev.metrics, "type", "") == "eou_metrics" else None)
    CATCHER.active.append(rec)
    try:
        await session.start(CheckAgent(rec, say))
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        rec.t0 = t0 = time.time()
        for i in range(len(pcm) // FRAME):
            await asyncio.sleep(max(0.0, t0 + 0.02 * (i + 1) - time.time()))
            audio_in.push(rtc.AudioFrame(data=pcm[i * FRAME:(i + 1) * FRAME].tobytes(), sample_rate=SR,
                                         num_channels=1, samples_per_channel=FRAME))
        rec("audio_done")
    except Exception as e:  # report, keep going with the other runs
        rec("exception", repr(e))
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(session.aclose(), 10)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(sync.aclose(), 5)
        CATCHER.active.remove(rec)
    return rec.events


# -- measurements ---------------------------------------------------------------
def first(evs, kind, value=None, after=-1e9):
    return next((e["t"] for e in evs if e["kind"] == kind and (value is None or e["value"] == value)
                 and e["t"] >= after), None)


def vad_sanity(evs):
    """START/END alternation as the session saw it."""
    seq = [e["kind"] for e in evs if e["kind"] in ("vad_start_of_speech", "vad_end_of_speech")]
    bad, state = 0, "end"
    for k in seq:
        now = "start" if k == "vad_start_of_speech" else "end"
        bad += now == state
        state = now
    return {"vad_starts": seq.count("vad_start_of_speech"), "vad_ends": seq.count("vad_end_of_speech"),
            "vad_out_of_order": bad, "vad_open_at_end": state == "start",
            "log_lines": [e["value"] for e in evs if e["kind"] == "log"],
            "exceptions": [e["value"] for e in evs if e["kind"] == "exception"]}


def measure_a(evs, on, off):
    evs = fed(evs)
    stop = first(evs, "user_state", "listening", after=off - 0.05)
    think = first(evs, "agent_state", "thinking", after=on)
    commit = first(evs, "turn_committed", after=on)
    audio = first(evs, "playback_started", after=on)
    return {"user_stop": None if stop is None else stop - off,
            "thinking": None if think is None else think - off,
            "committed": None if commit is None else commit - off,
            "first_audio": None if audio is None else audio - off,
            "eou_metric": next((e["value"] for e in evs if e["kind"] == "eou_metrics"), None),
            "commits": sum(e["kind"] == "turn_committed" for e in evs),
            "thinking_before_end": sum(e["kind"] == "agent_state" and e["value"] == "thinking" and e["t"] < off
                                       for e in evs),
            "user_segments": sum(e["kind"] == "user_state" and e["value"] == "speaking" for e in evs),
            **vad_sanity(evs)}


def agent_stop(evs, on):
    """First time after onset - 0.3 s that the agent stopped (pause, clear, or state off 'speaking')."""
    cands = [(e["t"], e["kind"]) for e in evs if e["t"] >= on - 0.3 and (
        e["kind"] in ("pause", "clear_buffer") or (e["kind"] == "agent_state" and e["value"] != "speaking"))]
    return min(cands) if cands else (None, None)


def speaking_at(evs, t):
    state = None
    for e in evs:
        if e["kind"] == "agent_state" and e["t"] <= t:
            state = e["value"]
    return state == "speaking"


def fed(evs):
    """Events up to the end of the audio; the session's close clears playback afterwards."""
    end = first(evs, "audio_done")
    return [e for e in evs if end is None or e["t"] < end]


def measure_b(evs, on, off):
    evs = fed(evs)
    t, how = agent_stop(evs, on)
    reply = first(evs, "playback_started", after=on)
    before = [e for e in evs if e["kind"] in ("pause", "clear_buffer") and 0 <= e["t"] < on - 0.3]
    return {"agent_speaking_at_onset": speaking_at(evs, on), "stop": None if t is None else t - on, "stop_how": how,
            "stopped_before_stt_final": t is not None and t < off + STT_DELAY,
            "vad_speech_0.5": (lambda x: None if x is None else x - on)(first(evs, "vad_speech_0.5", after=on - 0.3)),
            "paused": any(e["kind"] == "pause" and e["t"] >= on - 0.3 for e in evs),
            "interrupted": any(e["kind"] == "clear_buffer" and e["t"] >= on - 0.3 for e in evs),
            "answered": first(evs, "turn_committed", after=on) is not None and reply is not None,
            "answer_audio": None if reply is None else reply - off,
            "stops_before_onset": len(before),
            "vad_start": (lambda x: None if x is None else x - on)(first(evs, "vad_start_of_speech", after=0.0)),
            **vad_sanity(evs)}


def measure_c(evs, on, off):
    evs = fed(evs)
    pause = first(evs, "pause", after=on - 0.3)
    resume = first(evs, "resume", after=0.0)  # also resumes of a pause the clip's lead-in caused
    false_ev = next((e for e in evs if e["kind"] == "false_interruption"), None)
    return {"agent_speaking_at_onset": speaking_at(evs, on),
            "paused": first(evs, "pause", after=0.0) is not None, "pause": None if pause is None else pause - on,
            "paused_before_onset": first(evs, "pause", after=0.0) is not None and (pause is None or
                                                                                  first(evs, "pause", after=0.0) < on - 0.3),
            "resumed": resume is not None, "resume_after_end": None if resume is None else resume - off,
            "false_interruption_event": None if false_ev is None else false_ev["value"],
            "interrupted": any(e["kind"] == "clear_buffer" and e["t"] >= on - 0.3 for e in evs),
            "agent_left_speaking": any(e["kind"] == "agent_state" and e["value"] != "speaking" and e["t"] >= on - 0.3
                                       for e in evs),
            **vad_sanity(evs)}


# -- material ----------------------------------------------------------------------
def material():
    rng = np.random.default_rng(0)
    speech = []
    for lang in LANGS:
        c = min((c for c in lab.clips("speech") if c["language"] == lang), key=lambda c: c["duration_s"])
        ref = lab.reference(c["file"])
        speech.append({"id": c["id"], "x": lab.load(c["file"]), "ref": ref, "text": c["transcript"]})
    a = [{"id": s["id"], "audio": lab.apply(s["x"], [("level", -30), ("pad", 0.5, 3.0), ("white", -65)],
                                            s["ref"]["level_db"], rng),
          "on": s["ref"]["on"] + 0.5, "off": s["ref"]["off"] + 0.5, "text": s["text"]} for s in speech]
    b = [{"id": s["id"], "audio": lab.apply(s["x"], [("level", -30), ("pad", 2.0, 3.0), ("white", -65)],
                                            s["ref"]["level_db"], rng),
          "on": s["ref"]["on"] + 2.0, "off": s["ref"]["off"] + 2.0, "text": s["text"]} for s in speech]
    bc = [c for c in lab.clips("backchannel") if not c["others_talking"] and c["speech_end_s"] - c["speech_start_s"] < 0.45]
    bc = bc[:: len(bc) // 6][:6]
    c = [{"id": c["id"], "audio": lab.apply(lab.load(c["file"]), [("pad", 1.5, 3.5), ("white", -65)], 0.0, rng),
          "on": c["speech_start_s"] + 1.5, "off": c["speech_end_s"] + 1.5, "text": "", "word": c["text"]} for c in bc]
    return a, b, c


# -- tables ------------------------------------------------------------------------
def cell(vals, kind):
    if kind == "ms":
        v = [x for x in vals if x is not None]
        if not v:
            return "-"
        s = f"{round(np.median(v) * 1000)} [{round(min(v) * 1000)}..{round(max(v) * 1000)}]"
        return s + (f" n={len(v)}" if len(v) < len(vals) else "")
    if kind == "count":
        return f"{sum(bool(x) for x in vals)}/{len(vals)}"
    if kind == "sum":
        return str(sum(x for x in vals))
    return str(vals)


def table(title, rows, results, keys):
    """rows: (label, field, kind); results[key] = list of measurement dicts."""
    out = [f"\n== {title}"]
    width = max(len(r[0]) for r in rows)
    out.append(" " * width + "".join(f"{k:>32}" for k in keys))
    for label, field, kind in rows:
        out.append(f"{label:{width}}" + "".join(f"{cell([m[field] for m in results[k]], kind):>32}" for k in keys))
    return "\n".join(out)


VAD_ROWS = [("VAD START/END out of order (sum)", "vad_out_of_order", "sum"),
            ("runs with VAD segment open at end", "vad_open_at_end", "count")]


# -- scenario D ----------------------------------------------------------------------
def scenario_d():
    detector = inference.TurnDetector()  # constructs only; nothing is connected
    out = {}
    for name, make in VADS.items():
        vad = make()
        ar = AudioRecognition.__new__(AudioRecognition)  # only the two fields the check reads
        ar._turn_detector, ar._vad = detector, vad
        try:
            ar._check_vad_silence_requirement()
            out[name] = {"min_silence_duration": getattr(vad, "min_silence_duration", None), "accepted": True}
        except ValueError as e:
            out[name] = {"min_silence_duration": getattr(vad, "min_silence_duration", None), "accepted": False,
                         "error": str(e)}
    return out


async def main():
    start = time.time()
    if sys.platform == "win32":
        ctypes.windll.winmm.timeBeginPeriod(1)  # 1 ms timer resolution for the frame pacing
    logging.getLogger().addHandler(CATCHER)
    logging.getLogger().setLevel(logging.WARNING)
    a, b, c = material()
    raw = {"stt_delay": STT_DELAY, "A": [], "B": [], "C": []}
    keys = list(VADS)
    res = {"A0.5": {k: [] for k in keys}, "A0.3": {k: [] for k in keys}, "B": {k: [] for k in keys},
           "C": {k: [] for k in keys}}

    for item in a:
        jobs = [(k, md) for md in (0.5, 0.3) for k in keys]
        outs = await asyncio.gather(*(run_session(VADS[k], item["audio"], (item["on"], item["off"], item["text"]),
                                                  min_delay=md) for k, md in jobs))
        for (k, md), evs in zip(jobs, outs):
            m = measure_a(evs, item["on"], item["off"])
            res[f"A{md}"][k].append(m)
            raw["A"].append({"vad": k, "min_delay": md, "clip": item["id"], "on": item["on"], "off": item["off"],
                             "measure": m, "events": evs})
        print(f"A {item['id']} done ({time.time() - start:.0f} s)", flush=True)
    for name, items, measure in (("B", b, measure_b), ("C", c, measure_c)):
        for item in items:
            outs = await asyncio.gather(*(run_session(VADS[k], item["audio"], (item["on"], item["off"], item["text"]),
                                                      min_delay=0.5, say=LONG_REPLY) for k in keys))
            for k, evs in zip(keys, outs):
                m = measure(evs, item["on"], item["off"])
                res[name][k].append(m)
                raw[name].append({"vad": k, "clip": item["id"], "word": item.get("word"), "on": item["on"],
                                  "off": item["off"], "measure": m, "events": evs})
            print(f"{name} {item['id']} done ({time.time() - start:.0f} s)", flush=True)
    raw["D"] = d = scenario_d()

    text = [f"AgentSession check, livekit-agents 1.8.11, turn_detection='vad', interruption mode 'vad' "
            f"min_duration 0.5 s, fake STT final at true end + {int(STT_DELAY * 1000)} ms.",
            "Values: median [min..max] in ms over the clips (n= when some runs had no such event), or counts."]
    for md in (0.5, 0.3):
        text.append(table(f"A. end of turn, agent idle, min_delay {md} s (6 sentences: {', '.join(i['id'] for i in a)}); "
                          "times after the true end of speech", [
                              ("user state leaves 'speaking'", "user_stop", "ms"),
                              ("agent 'thinking' (turn committed)", "thinking", "ms"),
                              ("on_user_turn_completed", "committed", "ms"),
                              ("first agent audio", "first_audio", "ms"),
                              ("EOU metric end_of_utterance_delay", "eou_metric", "ms"),
                              ("runs with >1 committed user turn", "commits", "count"),
                              ("'thinking' before the true end (sum)", "thinking_before_end", "sum"),
                              ("user 'speaking' segments (sum)", "user_segments", "sum"),
                          ] + VAD_ROWS, {k: [dict(m, commits=m["commits"] > 1) for m in v] for k, v in res[f"A{md}"].items()},
                          keys))
    text.append(table("B. interruption: agent speaking a 12 s reply, user starts a sentence (same 6); "
                      "times after the true onset", [
                          ("agent speaking at onset", "agent_speaking_at_onset", "count"),
                          ("first VAD START (vs. reference onset)", "vad_start", "ms"),
                          ("VAD speech_duration reaches 0.5 s", "vad_speech_0.5", "ms"),
                          ("agent paused/interrupted", "stop", "ms"),
                          ("  of which by VAD (before STT final)", "stopped_before_stt_final", "count"),
                          ("  paused (pause path)", "paused", "count"),
                          ("  interrupted (playback cleared)", "interrupted", "count"),
                          ("user turn answered", "answered", "count"),
                          ("answer audio after true end", "answer_audio", "ms"),
                          ("agent stops before the onset (sum)", "stops_before_onset", "sum"),
                      ] + VAD_ROWS, res["B"], keys))
    text.append(table("C. short real backchannel during agent speech (" +
                      ", ".join(f"{i['word']!r} {round((i['off'] - i['on']) * 1000)} ms" for i in c) +
                      "), empty STT transcript", [
                          ("agent speaking at onset", "agent_speaking_at_onset", "count"),
                          ("agent paused (any time)", "paused", "count"),
                          ("  first pause before onset - 0.3 s (clip lead-in)", "paused_before_onset", "count"),
                          ("  pause after onset", "pause", "ms"),
                          ("  resumed", "resumed", "count"),
                          ("  resume after end of the sound", "resume_after_end", "ms"),
                          ("agent interrupted (playback cleared)", "interrupted", "count"),
                          ("agent state left 'speaking'", "agent_left_speaking", "count"),
                      ] + VAD_ROWS, res["C"], keys))
    text.append("\n== D. default streaming turn detector (inference.TurnDetector) vs VAD min_silence_duration "
                "(AudioRecognition._check_vad_silence_requirement, requires >= 0.25 s)")
    for k, v in d.items():
        text.append(f"{k:32} min_silence_duration={v['min_silence_duration']}  "
                    + ("accepted" if v["accepted"] else f"REJECTED: {v['error']}"))
    logs = {}
    for name in ("A", "B", "C"):
        for r in raw[name]:
            for line in r["measure"]["log_lines"] + r["measure"]["exceptions"]:
                logs.setdefault((r["vad"], line[:160]), 0)
                logs[(r["vad"], line[:160])] += 1
    text.append("\n== warnings/errors logged during the runs (vad, count, message)")
    text += [f"{k[0]:32} {n:3}x {k[1]}" for k, n in sorted(logs.items())] or ["(none)"]
    text.append(f"\ntotal runtime {time.time() - start:.0f} s")
    print("\n".join(text))
    raw["tables"] = "\n".join(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    asyncio.run(main())
