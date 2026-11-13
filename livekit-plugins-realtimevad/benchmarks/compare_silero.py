"""RealtimeVAD plugin against LiveKit's Silero plugin on the same audio, at the VAD interface.

Both get identical 20 ms frames through ``vad.stream()`` and their VADEvents are compared for
the four things an integration has to get right:

  opening       when START_OF_SPEECH comes after the speech began; speech that never starts
  ending        when END_OF_SPEECH comes after the speech stopped, how exactly it reports the
                stop (timestamp - silence_duration), when LiveKit would start its turn detector
                (raw_accumulated_silence >= 0.2 s) and commit the turn (END, but not before
                stop + min_delay; STT assumed instant)
  segmentation  how many START/END pairs a turn with pauses becomes
  interruption  when an INFERENCE_DONE reaches speech_duration >= 0.5 s (LiveKit's default
                min_duration: that interrupts the agent), and how many short sounds get there

Needs the test audio of the RealtimeVAD repo (tests/audio/fetch_audio.py, fetch_backchannels.py).

    python benchmarks/compare_silero.py [out.json]
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents.vad import VADEventType as T
from livekit.plugins import realtimevad, silero

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "audio"))
import lab  # noqa: E402

CONFIGS = {
    "silero default (0.55 s)": lambda: silero.VAD.load(),
    "silero 0.25 s": lambda: silero.VAD.load(min_silence_duration=0.25),
    "realtimevad early end (0.25 s)": lambda: realtimevad.VAD.load(min_silence_duration=0.25, early_end=True),
    "realtimevad early end 0.55 s": lambda: realtimevad.VAD.load(min_silence_duration=0.55, early_end=True),
}


async def events(vad, x, rate=16000, frame_ms=20):
    """Push float audio as int16 frames; returns the VADEvents as dicts."""
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    if rate != 16000:  # plain linear resampling is enough to exercise the plugin's own resampler
        t = np.arange(int(len(pcm) * rate / 16000)) * 16000 / rate
        pcm = np.interp(t, np.arange(len(pcm)), pcm).astype(np.int16)
    stream, step, out = vad.stream(), rate * frame_ms // 1000, []
    for i in range(0, len(pcm) - step + 1, step):
        stream.push_frame(rtc.AudioFrame(data=pcm[i : i + step].tobytes(), sample_rate=rate, num_channels=1,
                                         samples_per_channel=step))
    stream.end_input()
    async for ev in stream:
        seconds = sum(f.samples_per_channel for f in ev.frames) / rate
        out.append({"type": ev.type, "t": ev.timestamp, "speech": ev.speech_duration, "silence": ev.silence_duration,
                    "raw_silence": ev.raw_accumulated_silence, "raw_speech": ev.raw_accumulated_speech,
                    "speaking": ev.speaking, "frames_s": seconds})
    await stream.aclose()
    return out


def measure(evs, on, off):
    starts = [e for e in evs if e["type"] == T.START_OF_SPEECH]
    ends = [e for e in evs if e["type"] == T.END_OF_SPEECH]
    inference = [e for e in evs if e["type"] == T.INFERENCE_DONE]
    row = {"starts": len(starts), "ends": len(ends)}
    first = next((e for e in starts if on - 0.3 <= e["t"] <= off + 0.3), None)
    before = [e for e in starts + ends if e["t"] < on - 0.3]  # crosstalk may have opened a segment already
    row["running"] = bool(before) and before[-1]["type"] == T.START_OF_SPEECH if first is None else False
    row["start_delay"] = None if first is None else first["t"] - on
    row["onset_error"] = None if first is None else first["t"] - first["speech"] - on
    row["start_padding"] = None if first is None else first["frames_s"] - first["speech"]
    last = next((e for e in ends if e["t"] >= off), None)  # the END that closes the turn
    if last is not None:
        stop = last["t"] - last["silence"]
        row.update(end_delay=last["t"] - off, stop_error=stop - off, end_frames_s=last["frames_s"],
                   commit_0_5=max(last["t"], stop + 0.5) - off, commit_0_3=max(last["t"], stop + 0.3) - off)
    row["early_ends"] = sum(e["t"] < off - 0.1 for e in ends)  # ENDs before the turn was over
    detector = next((e["t"] for e in inference if e["t"] >= off - 0.05 and e["speaking"] and e["raw_silence"] >= 0.2), None)
    row["turn_detector_delay"] = None if detector is None else detector - off
    interrupt = next((e["t"] for e in inference if e["speech"] >= 0.5), None)
    row["interrupt_delay"] = None if interrupt is None else interrupt - on
    return row


def summary(rows):
    def pick(field):
        return [r[field] for r in rows if r.get(field) is not None]

    def med(field):
        return None if not pick(field) else round(float(np.median(pick(field))) * 1000)

    def p90(field):
        return None if not pick(field) else round(float(np.percentile(pick(field), 90)) * 1000)

    n = len(rows)
    return {
        "clips": n,
        "not heard (no START, none running)": sum(r["start_delay"] is None and not r["running"] for r in rows),
        "START after onset ms (median / p90)": f"{med('start_delay')} / {p90('start_delay')}",
        "reported onset - true onset ms (median / p90)": f"{med('onset_error')} / {p90('onset_error')}",
        "START padding ms": med("start_padding"),
        "turn-closing END found": sum("end_delay" in r for r in rows),
        "END after stop ms (median / p90)": f"{med('end_delay')} / {p90('end_delay')}",
        "reported stop - true stop ms": med("stop_error"),
        "turn detector starts after stop ms": med("turn_detector_delay"),
        "commit after stop ms, min_delay 0.5 / 0.3": f"{med('commit_0_5')} / {med('commit_0_3')}",
        "clips with an END before the turn was over": sum(r["early_ends"] > 0 for r in rows),
        "START/END pairs per clip": round(sum(r["ends"] for r in rows) / n, 2),
        "clips reaching speech_duration 0.5 s (interrupt)": sum(r["interrupt_delay"] is not None for r in rows),
        "interrupt after onset ms (median)": med("interrupt_delay"),
    }


def material():
    """name -> list of (audio, onset, offset)."""
    rng = np.random.default_rng(0)
    speech = lab.clips("speech")
    s24 = [speech[10 * lang + i] for lang in range(8) for i in (0, 4, 8)]
    refs = {c["file"]: lab.reference(c["file"]) for c in s24}
    sets = {"sentences, 8 languages (24)": [], "phone line at -45 dBFS (24)": []}
    for c in s24:
        ref = refs[c["file"]]
        x = lab.load(c["file"])
        sets["sentences, 8 languages (24)"].append(
            (lab.apply(x, [("level", -30), ("pad", 0.5, 1.5), ("white", -65)], ref["level_db"], rng), ref["on"] + 0.5, ref["off"] + 0.5))
        sets["phone line at -45 dBFS (24)"].append(
            (lab.apply(x, [("level", -45), ("pad", 0.5, 1.5), ("phone",)], ref["level_db"], rng), ref["on"] + 0.5, ref["off"] + 0.5))
    for name, category in (("real turn starts, meetings (60)", "turn_start"), ("turns with a 0.5-1.5 s pause (30)", "hesitation"),
                           ("backchannels 0.15-0.8 s (200)", "backchannel")):
        sets[name] = [(lab.apply(lab.load(c["file"]), [("pad", 0.0, 1.5)], 0.0, rng), c["speech_start_s"], c["speech_end_s"])
                      for c in lab.clips(category)]
    sets["coughs, laughter, sighs, sneezes (50)"] = [
        (lab.apply(lab.load(c["file"]), [("pad", 0.5, 1.5)], 0.0, rng), 0.5, 0.5 + c["duration_s"]) for c in lab.clips("nonverbal")]
    return sets


async def main():
    sets = material()
    out = {}
    for config, make in CONFIGS.items():
        vad = make()
        out[config] = {}
        for name, items in sets.items():
            rows = [measure(await events(vad, x), on, off) for x, on, off in items]
            out[config][name] = summary(rows)
    # the same audio at 24 kHz in 50 ms frames (LiveKit's room input) must give the same events
    x, on, off = sets["sentences, 8 languages (24)"][1]
    a = await events(realtimevad.VAD.load(), x)
    b = await events(realtimevad.VAD.load(), x, rate=24000, frame_ms=50)

    def key(evs):
        return [(e["type"], round(e["t"], 1)) for e in evs if e["type"] != T.INFERENCE_DONE]

    same = key(a) == key(b)
    for name in sets:
        print(f"\n== {name}")
        fields = list(next(iter(out.values()))[name].keys())
        width = max(len(f) for f in fields)
        print(" " * width, *(f"{c:>30}" for c in CONFIGS))
        for f in fields:
            print(f"{f:{width}}", *(f"{str(out[c][name][f]):>30}" for c in CONFIGS))
    print("\n24 kHz / 50 ms frames give the same START/END (to 0.1 s):", same)
    out["24 kHz / 50 ms frames give the same START/END"] = same
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
