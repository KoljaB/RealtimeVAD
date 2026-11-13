"""Does transcribing at the pause make a LiveKit agent answer sooner? ``SpeculativeSTT`` in an AgentSession.

Same session as ``agent_session_real_stt.py`` (livekit-agents 1.8.11, ``turn_detection="vad"``,
``min_delay`` 0.5 s, fake LLM and TTS, real local STT, audio at real-time pace). Four arms per STT,
all with 0.55 s of silence to END_OF_SPEECH:

  silero                 the Silero plugin and the STT the way LiveKit runs it: Whisper behind its
                         StreamAdapter (one request at END_OF_SPEECH), or the streaming recognizer
                         flushed by LiveKit at END_OF_SPEECH (``manual_flush``)
  silero+speculative     the Silero plugin, the STT inside ``realtimevad.SpeculativeSTT``: the
                         transcript is requested after 0.1 s of Silero's raw silence
  realtimevad            RealtimeVAD with its END at Silero's timing (``quiet_end_s`` out of reach:
                         only Silero's frame count ends a segment, the reported stop is Silero's last
                         voiced frame) and the STT the way LiveKit runs it. The early pause is in
                         ``raw_accumulated_silence`` but nothing uses it.
  realtimevad+speculative  the same VAD, the STT inside ``SpeculativeSTT``: the transcript is
                         requested 0.1 s after the audio went quiet

Preemptive generation is on in all arms with ``max_retries`` 10 and ``max_speech_duration`` 30 s
(LiveKit's defaults, 3 and 10 s, switch it off for a long sentence with several pauses, in every
arm alike).

    python benchmarks/speculative_check.py r1_whisper_a.json 1 whisper en-US,de-DE,es-419,fr-FR
    python benchmarks/speculative_check.py r1_sherpa.json 1 sherpa
    python benchmarks/speculative_check.py report r*.json
"""

import asyncio
import ctypes
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_session_real_stt as real  # noqa: E402
import local_stt  # noqa: E402
from agent_session_real_stt import base  # noqa: E402
from livekit.plugins import realtimevad, silero  # noqa: E402

SILENCE = 0.55
PREEMPTIVE = {"max_retries": 10, "max_speech_duration": 30.0}
ARMS = ["silero", "silero+speculative", "realtimevad", "realtimevad+speculative"]
STTS = {
    # recognition is CPU-heavy and a speculative request may overlap the next: one session at a time,
    # two requests side by side on 3 threads each
    "whisper": {"title": "faster-whisper base (not streaming)", "langs": tuple(real.WHISPER_LANG), "per_lang": 2,
                "parallel": 1, "ttft": [0.5],
                "make": lambda lang: local_stt.WhisperSTT(size="base", language=real.WHISPER_LANG[lang])},
    "sherpa": {"title": "sherpa-onnx streaming zipformer with manual_flush", "langs": ("en-US",), "per_lang": 10,
               "parallel": 4, "ttft": [0.5, 0.1],
               "make": lambda lang: local_stt.SherpaSTT(endpoint_silence=1.2, manual_flush=True)},
}


def session_parts(arm, inner):
    vad = (silero.VAD.load(min_silence_duration=SILENCE) if arm.startswith("silero")
           else realtimevad.VAD.load(min_silence_duration=SILENCE, quiet_end_s=1e6))
    return vad, (realtimevad.SpeculativeSTT(stt=inner, vad=vad) if arm.endswith("+speculative") else inner)


def measure(evs, on, off):
    m = real.measure_a(evs, on, off)
    evs = base.fed(evs)
    audio = None if m["first_audio"] is None else m["first_audio"] + off
    calls = [e["t"] for e in evs if e["kind"] == "llm_start" and (audio is None or e["t"] <= audio)]
    m["reply_llm"] = real.rel(calls[-1] if calls else None, off)  # the call whose reply was spoken
    hyp = [e["t"] for e in evs if e["kind"] in ("stt_request", "stt_flush", "stt_endpoint") and e["value"]]
    m["transcript_ready"] = real.rel(hyp[-1] if hyp else None, off)  # the recognizer's last result
    m["stt_work"] = sum(e["kind"] in ("stt_request", "stt_flush") for e in evs)
    return m


async def collect(name, path, label, langs=None):
    start = time.time()
    if sys.platform == "win32":
        ctypes.windll.winmm.timeBeginPeriod(1)
    logging.getLogger().addHandler(base.CATCHER)
    logging.getLogger().setLevel(logging.WARNING)
    cfg = dict(STTS[name])
    cfg["langs"] = tuple(lang for lang in cfg["langs"] if not langs or lang in langs)
    if name == "whisper":
        local_stt.whisper_model("base", workers=2, threads=3)
    raw = {"runs": [], "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": label}
    pool = {lang: [cfg["make"](lang) for _ in range(cfg["parallel"])] for lang in cfg["langs"]}

    async def one(arm, inner, audio, ttft):
        vad, stt = session_parts(arm, inner)
        return await real.run_session(lambda: vad, stt, audio, min_delay=0.5, ttft=ttft, preemptive=PREEMPTIVE)

    warm = real.material(cfg["langs"][:1], 1, 0.5)[0]  # first use loads models: not measured
    await asyncio.gather(*(one(ARMS[i], s, warm["audio"][:real.SR * 4], 0.1)
                           for i, s in enumerate(pool[cfg["langs"][0]])))
    for item in real.material(cfg["langs"], cfg["per_lang"], 0.5):
        inners = pool[item["lang"]]
        for ttft in cfg["ttft"]:
            n = cfg["parallel"]
            for group in (ARMS[i:i + n] for i in range(0, len(ARMS), n)):
                outs = await asyncio.gather(*(one(arm, inners[i], item["audio"], ttft) for i, arm in enumerate(group)))
                for arm, evs in zip(group, outs):
                    m = measure(evs, item["on"], item["off"])
                    late = m["pump_late"]
                    raw["runs"].append({"stt": name, "arm": arm, "llm_ttft": ttft, "clip": item["id"],
                                        "on": item["on"], "off": item["off"],
                                        "valid": late is not None and late <= real.MAX_LATE, "measure": m,
                                        "events": evs})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, default=str)
        print(f"{name} {item['id']} done ({time.time() - start:.0f} s)", flush=True)


def report(paths):
    parts = [json.load(open(p, encoding="utf-8")) for p in paths]
    labels = sorted({p["rep"] for p in parts})
    runs = [dict(r, rep=p["rep"]) for p in parts for r in p["runs"]]
    used = [r for r in runs if r["valid"]]
    out = [f"SpeculativeSTT in an AgentSession, livekit-agents 1.8.11, turn_detection='vad', min_delay 0.5 s, "
           f"{SILENCE} s of silence to END_OF_SPEECH in every arm, preemptive generation {PREEMPTIVE}, fake LLM and "
           f"TTS (TTS: first audio 0.1 s after its text). {len(labels)} repetitions.",
           "Times in ms after the true end of speech: median over all runs (each repetition's median). n-N: N runs "
           "had no such event. Counts: one number per repetition.",
           "'sooner than silero' rows: the same clip in the same repetition, silero minus that arm: median (each "
           "repetition's median) and the 10th..90th percentile of all pairs.",
           f"Runs: {len(used)} used, {len(runs) - len(used)} not used (audio pump more than "
           f"{real.MAX_LATE * 1000:.0f} ms late). Audio pump at most "
           f"{max(r['measure']['pump_late'] for r in used) * 1000:.0f} ms late in the used runs."]
    width = 44
    for name, cfg in STTS.items():
        for ttft in cfg["ttft"]:
            clips = sorted({r["clip"] for r in used if r["stt"] == name and r["llm_ttft"] == ttft})
            if not clips:
                continue
            # only clips that every arm has in that repetition, so the rows compare the same sentences
            sel = {}
            for arm in ARMS:
                sel[arm] = []
                for label in labels:
                    have = {r["clip"]: r["measure"] for r in used
                            if r["stt"] == name and r["llm_ttft"] == ttft and r["arm"] == arm and r["rep"] == label}
                    whole = [c for c in clips if all(any(
                        r["clip"] == c and r["arm"] == a and r["rep"] == label and r["stt"] == name
                        and r["llm_ttft"] == ttft for r in used) for a in ARMS)]
                    sel[arm].append([have[c] for c in whole])
            out.append(f"\n== {cfg['title']}; {len(clips)} sentences ({', '.join(cfg['langs'])}); "
                       f"LLM first token after {ttft} s")
            out.append(" " * width + "".join(f"{a:>30}" for a in ARMS))
            for label, field in (("VAD END_OF_SPEECH", "vad_end"), ("recognizer's last result", "transcript_ready"),
                                 ("final transcript reaches LiveKit", "final"),
                                 ("LLM call whose reply is spoken starts", "reply_llm"),
                                 ("turn committed", "committed"), ("first agent audio", "first_audio")):
                out.append(f"{label:{width}}" + "".join(f"{real.stat(sel[a], field):>30}" for a in ARMS))
            for label, field in (("STT requests / flushes", "stt_work"), ("LLM calls", "llm_calls"),
                                 ("sentences that became 2+ turns", "split")):
                out.append(f"{label:{width}}" + "".join(f"{real.total(sel[a], field):>30}" for a in ARMS))
            for label, field in (("LLM call", "reply_llm"), ("first agent audio", "first_audio")):
                for arm in ARMS[1:]:
                    out.append(f"{label}, {arm} sooner than silero: {real.gain(sel[ARMS[0]], sel[arm], field)}")
            out.append(f"first agent audio, realtimevad+speculative sooner than silero+speculative: "
                       f"{real.gain(sel[ARMS[1]], sel[ARMS[3]], 'first_audio')}")
        took = [t for r in used if r["stt"] == name for t in r["measure"]["request_s"]]
        if took:
            out.append(f"recognize() took median {np.median(took):.2f} s [{min(took):.2f}..{max(took):.2f}]")
    logs = {}
    for r in runs:
        for line in r["measure"]["log_lines"] + r["measure"]["exceptions"]:
            key = (r["stt"], r["arm"], line[:150])
            logs[key] = logs.get(key, 0) + 1
    out.append("\n== warnings/errors logged during the runs (stt, arm, count, message)")
    out += [f"{k[0]:8} {k[1]:24} {n:3}x {k[2]}" for k, n in sorted(logs.items())] or ["(none)"]
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        print(report(sys.argv[2:]))
    else:
        asyncio.run(collect(sys.argv[3], sys.argv[1], sys.argv[2], sys.argv[4].split(",") if len(sys.argv) > 4 else None))
