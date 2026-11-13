"""``agent_session_check.py`` with a real, locally running STT instead of LiveKit's fake one.

Same ``AgentSession`` (livekit-agents 1.8.11, ``turn_detection="vad"``, interruption mode "vad",
preemptive generation on as by default), same three VADs, same audio pump. The STT is one of
(``local_stt.py``):

  endpoint   sherpa-onnx streaming zipformer (English), interim + final transcripts; finalizes on
             its own endpoint rule: 0.05 s, 0.2 s or 1.2 s (sherpa-onnx's default) after its last
             token. The model itself is about 0.4 s behind the audio.
  flush      the same recognizer with ``STTCapabilities.manual_flush``: LiveKit flushes it at the
             VAD's END_OF_SPEECH and it finalizes there
  pauseflush the same, and the application flushes it earlier: at the first INFERENCE_DONE of a
             pause that reports 0.1 s of ``raw_accumulated_silence`` (LiveKit has no hook for this;
             the benchmark calls ``stream.flush()`` itself)
  adapter    faster-whisper "base" (non-streaming) behind LiveKit's ``StreamAdapter``, which the
             session adds with its own VAD: one ``recognize()`` per VAD segment

What LiveKit's ``AudioRecognition`` waits for (audio_recognition.py, 1.8.11): the end-of-turn task
starts at the VAD's END_OF_SPEECH or at a final transcript that arrives while the VAD is not in
speech, never without a final transcript; it then sleeps until ``reported stop + min_delay``
(reported stop = END time - ``silence_duration``). So

    turn committed = max(VAD END, final transcript, reported stop + min_delay)

and the reply's LLM call starts at the final transcript (preemptive generation), its audio once the
turn is committed.

Scenarios:
  A  end of turn, agent idle: min_delay 0.5 and 0.3 with a 0.1 s LLM (as in agent_session_check),
     and min_delay 0.5 with a 0.5 s LLM (not every combination for every STT, see STTS). Per run:
     VAD END, final transcript, reply LLM call, turn committed, first agent audio, all after the
     true end of speech.
  B  the agent speaks a 12 s reply when the user starts a sentence: agent paused after the onset.

The three VADs of one clip run side by side (one thread each for the streaming recognizer), the
Whisper sessions one at a time on 4 threads. Run it on a machine that does nothing else: a run whose
audio pump was more than 60 ms late is listed but not used. One call is one STT of one repetition;
``report`` puts the repetitions side by side.

    python benchmarks/agent_session_real_stt.py r1_flush.json 1 flush
    python benchmarks/agent_session_real_stt.py r1_adapter_a.json 1 adapter en-US,de-DE,es-419,fr-FR
    python benchmarks/agent_session_real_stt.py report r*.json

STTs: endpoint0.05, endpoint0.2, endpoint1.2, flush, pauseflush, adapter. Needs sherpa-onnx,
faster-whisper and the models (see local_stt.py) besides what agent_session_check.py needs. CPU
only; nothing leaves the machine.
"""

import asyncio
import contextlib
import ctypes
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_session_check as base  # noqa: E402  (also puts lab and LiveKit's tests on the path)
import lab  # noqa: E402
import local_stt  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer  # noqa: E402
from tests.fake_io import FakeAudioInput  # noqa: E402
from tests.fake_llm import FakeLLM, FakeLLMStream  # noqa: E402
from tests.fake_tts import FakeTTS, FakeTTSResponse  # noqa: E402

SR, FRAME = base.SR, base.FRAME
VADS = base.VADS
KEYS = list(VADS)
WHISPER_LANG = {"en-US": "en", "de-DE": "de", "es-419": "es", "fr-FR": "fr", "cmn-Hans-CN": "zh", "ja-JP": "ja",
                "hi-IN": "hi", "ru-RU": "ru"}
ALL = [(0.5, 0.1), (0.3, 0.1), (0.5, 0.5)]  # (min_delay, LLM time to first token)
TAIL_S = 4.5
MAX_LATE = 0.06  # s; a run whose audio pump was later than this is listed but not used
SHERPA = {"langs": ("en-US",), "per_lang": 10, "parallel": 3}  # the three VADs run side by side
STTS = {
    "endpoint0.05": {"title": "streaming STT, own endpoint 0.05 s after its last token (sherpa-onnx zipformer)",
                     "make": lambda lang: local_stt.SherpaSTT(endpoint_silence=0.05), "variants": ALL[:2], **SHERPA},
    "endpoint0.2": {"title": "streaming STT, own endpoint 0.2 s after its last token",
                    "make": lambda lang: local_stt.SherpaSTT(endpoint_silence=0.2), "variants": ALL[:2],
                    "interruption": True, **SHERPA},
    "endpoint1.2": {"title": "streaming STT, own endpoint 1.2 s (sherpa-onnx's default)",
                    "make": lambda lang: local_stt.SherpaSTT(endpoint_silence=1.2), "variants": ALL[:1], **SHERPA},
    "flush": {"title": "streaming STT that finalizes on LiveKit's flush at the VAD's END (manual_flush)",
              "make": lambda lang: local_stt.SherpaSTT(endpoint_silence=1.2, manual_flush=True), "variants": ALL,
              **SHERPA},
    "pauseflush": {"title": "the same, flushed by the application after 0.1 s of raw VAD silence inside speech",
                   "make": lambda lang: local_stt.SherpaSTT(endpoint_silence=1.2, manual_flush=True),
                   "variants": [ALL[0], ALL[2]], "flush_at_silence": 0.1, **SHERPA},
    # recognition is CPU-heavy: one session at a time, 4 threads
    "adapter": {"title": "non-streaming STT behind StreamAdapter (faster-whisper base, CPU)",
                "make": lambda lang: local_stt.WhisperSTT(size="base", language=WHISPER_LANG[lang]),
                "langs": tuple(WHISPER_LANG), "per_lang": 2, "parallel": 1, "variants": ALL[:1]},
}


class ReplyStream(FakeLLMStream):
    async def _run(self) -> None:
        self._llm.rec("llm_start")
        await asyncio.sleep(self._llm.ttft)
        self._send_chunk(delta=base.SHORT_REPLY)
        await asyncio.sleep(0.1)


class ReplyLLM(FakeLLM):
    """Answers anything with SHORT_REPLY after ``ttft`` (the transcript of a real STT is not known in advance)."""

    def __init__(self, ttft, rec):
        super().__init__()
        self.ttft, self.rec = ttft, rec

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **_):
        return ReplyStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class SilenceProxy(base.StreamProxy):
    """Also calls ``on_silence`` once per pause, when its raw silence reaches ``after`` seconds."""

    def __init__(self, inner, rec, after, on_silence):
        super().__init__(inner, rec)
        self._after, self._on_silence, self._told = after, on_silence, False

    async def __anext__(self):
        ev = await super().__anext__()
        if ev.type == base.VADEventType.INFERENCE_DONE and self._after is not None:
            if ev.speaking and ev.raw_accumulated_silence >= self._after:
                if not self._told:
                    self._told = True
                    self._rec("app_flush")
                    self._on_silence()
            else:
                self._told = False
        return ev


def wrap_vad(vad, rec, after=None, on_silence=None):
    """Like base.wrap_vad, but only the first stream is timestamped: with a non-streaming STT the
    session opens a second one for its StreamAdapter, fed with the same audio."""
    original, n = vad.stream, []

    def stream(*a, **k):
        n.append(1)
        rec("vad_stream", len(n))
        return SilenceProxy(original(*a, **k), rec, after, on_silence) if len(n) == 1 else original(*a, **k)

    vad.stream = stream
    return vad


async def run_session(make_vad, stt, audio, *, min_delay, ttft, say=None, flush_at_silence=None, preemptive=None):
    """stt: an instance no other running session uses (loading one blocks the loop, so they are made before)."""
    rec = base.Recorder()
    getattr(stt, "wrapped_stt", stt)._log = rec
    tts = [FakeTTSResponse(input=base.SHORT_REPLY, audio_duration=base.SHORT_REPLY_S, ttfb=0.1, duration=0.2),
           FakeTTSResponse(input=base.LONG_REPLY, audio_duration=base.LONG_REPLY_S, ttfb=0.1, duration=0.3)]
    session = AgentSession(
        vad=wrap_vad(make_vad(), rec, flush_at_silence, getattr(stt, "flush_streams", None)), stt=stt, llm=ReplyLLM(ttft, rec),
        tts=FakeTTS(fake_responses=tts),
        turn_handling={"turn_detection": "vad", "endpointing": {"min_delay": min_delay},
                       "interruption": {"mode": "vad", "min_duration": 0.5},
                       **({"preemptive_generation": preemptive} if preemptive else {})},
        aec_warmup_duration=None)
    audio_in, audio_out = FakeAudioInput(), base.RecordingAudioOutput(rec)
    sync = TranscriptSynchronizer(next_in_chain_audio=audio_out, next_in_chain_text=base.QuietText())
    session.input.audio = audio_in
    session.output.audio = sync.audio_output
    session.output.transcription = sync.text_output
    session.on("user_state_changed", lambda ev: rec("user_state", ev.new_state))
    session.on("agent_state_changed", lambda ev: rec("agent_state", ev.new_state))
    session.on("user_input_transcribed",
               lambda ev: rec("transcript_final" if ev.is_final else "transcript_interim", ev.transcript))
    base.CATCHER.active.append(rec)
    try:
        await session.start(base.CheckAgent(rec, say))
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        await asyncio.sleep(0.5)  # models loaded, streams open
        rec.t0 = t0 = time.time()
        late = 0.0
        for i in range(len(pcm) // FRAME):
            due = t0 + 0.02 * (i + 1)
            await asyncio.sleep(max(0.0, due - time.time()))
            late = max(late, time.time() - due)
            audio_in.push(rtc.AudioFrame(data=pcm[i * FRAME:(i + 1) * FRAME].tobytes(), sample_rate=SR,
                                         num_channels=1, samples_per_channel=FRAME))
        rec("audio_done", round(late, 4))
    except Exception as e:
        rec("exception", repr(e))
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(session.aclose(), 10)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(sync.aclose(), 5)
        base.CATCHER.active.remove(rec)
    return rec.events


def rel(t, ref):
    return None if t is None else t - ref


def measure_a(evs, on, off):
    late = next((e["value"] for e in evs if e["kind"] == "audio_done"), None)
    evs = base.fed(evs)
    first = base.first
    commits = [e["t"] for e in evs if e["kind"] == "turn_committed" and e["t"] >= off]
    finals = [e["t"] for e in evs if e["kind"] == "transcript_final" and e["t"] >= off - 0.05]
    whole = commits[-1] if commits else None  # the commit that holds the sentence's last words
    return {
        "vad_end": rel(first(evs, "vad_end_of_speech", after=off - 0.05), off),
        "user_stop": rel(first(evs, "user_state", "listening", after=off - 0.05), off),
        "final": rel(finals[-1] if finals else None, off),
        "llm_start": rel(first(evs, "llm_start", after=finals[-1]) if finals else None, off),
        "committed": rel(whole, off),
        "first_audio": rel(first(evs, "playback_started", after=whole) if whole else None, off),
        "first_commit": rel(commits[0] if commits else None, off),
        "split": len(commits) > 1 or any(e["kind"] == "turn_committed" and e["t"] < off for e in evs),
        "commits_before_end": sum(e["kind"] == "turn_committed" and e["t"] < off for e in evs),
        "commits_after_end": sum(e["kind"] == "turn_committed" and e["t"] >= off for e in evs),
        "finals": sum(e["kind"] == "transcript_final" for e in evs),
        "requests": sum(e["kind"] == "stt_request" for e in evs),
        "llm_calls": sum(e["kind"] == "llm_start" for e in evs),
        "request_s": [e["took"] for e in evs if e["kind"] == "stt_request"],
        "text": " | ".join(str(e["value"]) for e in evs if e["kind"] == "turn_committed"),
        "pump_late": late,
        **base.vad_sanity(evs)}


def measure_b(evs, on, off):
    evs = base.fed(evs)
    t, how = base.agent_stop(evs, on)
    reply = base.first(evs, "playback_started", after=on)
    return {"agent_speaking_at_onset": base.speaking_at(evs, on), "stop": rel(t, on), "stop_how": how,
            "vad_speech_0.5": rel(base.first(evs, "vad_speech_0.5", after=on - 0.3), on),
            "first_interim": rel(base.first(evs, "transcript_interim", after=on), on),
            "first_final": rel(base.first(evs, "transcript_final", after=on), on),
            "paused": any(e["kind"] == "pause" and e["t"] >= on - 0.3 for e in evs),
            "interrupted": any(e["kind"] == "clear_buffer" and e["t"] >= on - 0.3 for e in evs),
            "answered": base.first(evs, "turn_committed", after=on) is not None and reply is not None,
            "answer_audio": rel(reply, off),
            "stops_before_onset": sum(e["kind"] in ("pause", "clear_buffer") and 0 <= e["t"] < on - 0.3 for e in evs),
            **base.vad_sanity(evs)}


def material(langs, per_lang, lead):
    rng = np.random.default_rng(0)
    out = []
    for lang in langs:
        for c in [c for c in lab.clips("speech") if c["language"] == lang][:per_lang]:
            ref = lab.reference(c["file"])
            out.append({"id": c["id"], "lang": lang, "text": c["transcript"],
                        "audio": lab.apply(lab.load(c["file"]), [("level", -30), ("pad", lead, TAIL_S), ("white", -65)],
                                           ref["level_db"], rng),
                        "on": ref["on"] + lead, "off": ref["off"] + lead})
    return out




# -- collecting: one repetition per call -----------------------------------------------
async def unit(jobs):
    """Run a few sessions at the same time; note how late their audio pumps were at worst."""
    outs = await asyncio.gather(*(j() for j in jobs))
    late = max((e["value"] for evs in outs for e in evs if e["kind"] == "audio_done"), default=None)
    bad = late is None or late > MAX_LATE
    return [{"outs": outs, "pump_late": late, "valid": not bad, "why": "audio pump late" if bad else ""}]


async def collect(names, path, label, langs=None):
    start = time.time()
    if sys.platform == "win32":
        ctypes.windll.winmm.timeBeginPeriod(1)
    logging.getLogger().addHandler(base.CATCHER)
    logging.getLogger().setLevel(logging.WARNING)
    raw = {"runs": [], "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": label}

    def keep(tries, metas, measure, item):
        for t in tries:
            for meta, evs in zip(metas, t["outs"]):
                raw["runs"].append({**meta, "clip": item["id"], "on": item["on"], "off": item["off"],
                                    "valid": t["valid"], "why": t["why"],
                                    "measure": measure(evs, item["on"], item["off"]), "events": evs})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, default=str)

    for name in names:
        cfg = dict(STTS[name])
        cfg["langs"] = tuple(lang for lang in cfg["langs"] if not langs or lang in langs)
        n = cfg["parallel"]
        pool = {lang: [cfg["make"](lang) for _ in range(n)] for lang in cfg["langs"]}
        warm = material(cfg["langs"][:1], 1, 0.5)[0]  # first use loads models and stalls the loop: not measured
        await asyncio.gather(*(run_session(VADS[KEYS[i % 3]], s, warm["audio"][:SR * 4], min_delay=0.5, ttft=0.1)
                               for i, s in enumerate(pool[cfg["langs"][0]])))
        for item in material(cfg["langs"], cfg["per_lang"], 0.5):
            stts = pool[item["lang"]]
            for v in cfg["variants"]:
                for group in (KEYS[i:i + n] for i in range(0, len(KEYS), n)):  # the three VADs together, or one by one
                    tries = await unit([lambda i=i, k=k: run_session(
                        VADS[k], stts[i], item["audio"], min_delay=v[0], ttft=v[1],
                        flush_at_silence=cfg.get("flush_at_silence")) for i, k in enumerate(group)])
                    keep(tries, [{"scenario": "A", "stt": name, "vad": k, "min_delay": v[0], "llm_ttft": v[1]}
                                 for k in group], measure_a, item)
            print(f"A {name} {item['id']} done ({time.time() - start:.0f} s)", flush=True)
        if cfg.get("interruption"):
            for item in material(cfg["langs"][:1], 10, 2.0):
                stts = pool[item["lang"]]
                tries = await unit([lambda i=i, k=k: run_session(VADS[k], stts[i], item["audio"], min_delay=0.5,
                                                                 ttft=0.1, say=base.LONG_REPLY)
                                    for i, k in enumerate(KEYS)])
                keep(tries, [{"scenario": "B", "stt": name, "vad": k} for k in KEYS], measure_b, item)
                print(f"B {name} {item['id']} done ({time.time() - start:.0f} s)", flush=True)


# -- report over the repetitions --------------------------------------------------------
def ms(x):
    return "-" if x is None else f"{round(x * 1000)}"


def stat(reps, field):
    """reps: per repetition a list of measurements. Median over all runs, and each repetition's median."""
    every = [m[field] for rep in reps for m in rep if m[field] is not None]
    if not every:
        return "-"
    each = [np.median([m[field] for m in rep if m[field] is not None]) for rep in reps
            if any(m[field] is not None for m in rep)]
    missing = sum(m[field] is None for rep in reps for m in rep)
    return f"{ms(np.median(every))} ({' '.join(ms(x) for x in each)})" + (f" n-{missing}" if missing else "")


def gain(reps_a, reps_b, field):
    """How much sooner b is than a: per clip and repetition, then as stat()."""
    diffs = [[x[field] - y[field] for x, y in zip(a, b) if x[field] is not None and y[field] is not None]
             for a, b in zip(reps_a, reps_b)]
    every = [d for rep in diffs for d in rep]
    if not every:
        return "-"
    lo, hi = np.percentile(every, [10, 90])
    return (f"{round(np.median(every) * 1000):+d} ({' '.join(f'{round(np.median(d) * 1000):+d}' for d in diffs if d)})"
            f" p10..p90 {round(lo * 1000):+d}..{round(hi * 1000):+d}")


def total(reps, field):
    return " ".join(str(sum(int(m[field]) for m in rep)) for rep in reps)


def report(paths):
    parts = [json.load(open(p, encoding="utf-8")) for p in paths]
    files = [{"rep": label, "started": min(p["started"] for p in parts if p["rep"] == label),
              "runs": [r for p in parts if p["rep"] == label for r in p["runs"]]}
             for label in sorted({p["rep"] for p in parts})]  # one job per STT (the Whisper one in two halves)
    out = [f"AgentSession with a real local STT, livekit-agents 1.8.11, turn_detection='vad', interruption mode "
           f"'vad' min_duration 0.5 s, preemptive generation on (default), fake LLM and TTS (TTS: first audio 0.1 s "
           f"after its text). {len(files)} repetitions ({', '.join(f['started'] for f in files)}).",
           "Times in ms after the true end of speech: median over all runs (each repetition's median). n-N: N runs "
           "had no such event. Counts: one number per repetition.",
           "'RealtimeVAD sooner' rows: the same clip in the same repetition, other VAD minus RealtimeVAD: "
           "median (each repetition's median) and the 10th..90th percentile of all pairs."]
    runs = [r for f in files for r in f["runs"]]
    bad = [r for r in runs if not r["valid"]]
    good = [r for r in runs if r["valid"]]
    out.append(f"Runs: {len(good)} used, {len(bad)} set aside and repeated"
               + (f" ({', '.join(sorted({r['why'] for r in bad}))})" if bad else "")
               + f". Audio pump at most {max(r['measure'].get('pump_late') or 0 for r in good if r['scenario'] == 'A') * 1000:.0f} ms late.")

    def reps_of(**want):
        """Per repetition the valid measurements matching ``want``, clips in order."""
        return [[r["measure"] for r in f["runs"] if r["valid"] and all(r.get(k) == v for k, v in want.items())]
                for f in files]

    width = 46
    for name, cfg in STTS.items():
        if not any(r["stt"] == name for r in good):
            continue
        for v in cfg["variants"]:
            sel = {k: reps_of(scenario="A", stt=name, vad=k, min_delay=v[0], llm_ttft=v[1]) for k in KEYS}
            n = len(sel[KEYS[0]][0])
            out.append(f"\n== A. {cfg['title']}; {n} sentences ({', '.join(cfg['langs'])}); min_delay {v[0]} s, "
                       f"LLM first token {v[1]} s")
            out.append(" " * width + "".join(f"{k:>34}" for k in KEYS))
            for label, field in (("VAD END_OF_SPEECH", "vad_end"), ("final transcript (the last one)", "final"),
                                 ("reply's LLM call starts", "llm_start"),
                                 ("turn committed (with the last words)", "committed"),
                                 ("first agent audio after it", "first_audio"),
                                 ("first turn committed after the end", "first_commit")):
                out.append(f"{label:{width}}" + "".join(f"{stat(sel[k], field):>34}" for k in KEYS))
            for label, field in (("sentences that became 2+ turns", "split"), ("final transcripts", "finals"),
                                 ("recognize() requests", "requests"), ("LLM calls", "llm_calls"),
                                 ("VAD START/END out of order", "vad_out_of_order")):
                out.append(f"{label:{width}}" + "".join(f"{total(sel[k], field):>34}" for k in KEYS))
            for label, field in (("LLM call", "llm_start"), ("turn committed", "committed"),
                                 ("first agent audio", "first_audio")):
                out.append(f"RealtimeVAD sooner, {label}: than Silero default {gain(sel[KEYS[0]], sel[KEYS[2]], field)}"
                           f"  |  than Silero 0.25 s {gain(sel[KEYS[1]], sel[KEYS[2]], field)}")
        took = [t for r in good if r["stt"] == name and r["scenario"] == "A" for t in r["measure"]["request_s"]]
        if took:
            out.append(f"recognize() took median {np.median(took):.2f} s [{min(took):.2f}..{max(took):.2f}]")
        if cfg.get("interruption"):
            sel = {k: reps_of(scenario="B", stt=name, vad=k) for k in KEYS}
            out.append(f"\n== B. interruption, {cfg['title']}; the agent speaks a 12 s reply, the user starts a "
                       f"sentence ({len(sel[KEYS[0]][0])}); ms after the true onset")
            out.append(" " * width + "".join(f"{k:>34}" for k in KEYS))
            for label, field in (("VAD speech_duration reaches 0.5 s", "vad_speech_0.5"),
                                 ("first interim transcript", "first_interim"),
                                 ("agent paused/interrupted", "stop"), ("answer audio after the true end", "answer_audio")):
                out.append(f"{label:{width}}" + "".join(f"{stat(sel[k], field):>34}" for k in KEYS))
            for label, field in (("agent speaking at onset", "agent_speaking_at_onset"), ("paused", "paused"),
                                 ("user turn answered", "answered"), ("agent stops before the onset", "stops_before_onset")):
                out.append(f"{label:{width}}" + "".join(f"{total(sel[k], field):>34}" for k in KEYS))
            out.append(f"RealtimeVAD sooner, agent paused: than Silero default {gain(sel[KEYS[0]], sel[KEYS[2]], 'stop')}"
                       f"  |  than Silero 0.25 s {gain(sel[KEYS[1]], sel[KEYS[2]], 'stop')}")
    logs = {}
    for r in runs:
        for line in r["measure"]["log_lines"] + r["measure"]["exceptions"]:
            key = (r["stt"], r["vad"], "used" if r["valid"] else "set aside", line[:150])
            logs[key] = logs.get(key, 0) + 1
    out.append("\n== warnings/errors logged during the runs (stt, vad, run, count, message)")
    out += [f"{k[0]:12} {k[1]:30} {k[2]:9} {n:3}x {k[3]}" for k, n in sorted(logs.items())] or ["(none)"]
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        print(report(sys.argv[2:]))
    else:
        asyncio.run(collect(sys.argv[3].split(","), sys.argv[1], sys.argv[2], sys.argv[4].split(",") if len(sys.argv) > 4 else None))
