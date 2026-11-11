"""How much earlier does RealtimeVAD detect speech-off than Silero alone?

Feeds each WAV (16 kHz mono, one user turn with trailing silence) through RealtimeVAD and
compares, at the turn's final speech end:
  - RealtimeVAD speech-off: its last "pause" event before the segment ends
  - Silero speech-off: the first frame of the final run of frames below a threshold,
    for thresholds 0.5 (Silero default), 0.35 (VADIterator's speech-over threshold),
    0.45 (RealtimeVAD/Miep continue threshold), from the very same probabilities
  - Silero VADIterator end: threshold 0.5 to start, 100 ms below 0.35 to end (its defaults)
It also counts pauses inside the segment that were followed by more speech (false
speech-off alarms) and estimates the room noise level of each file.

    python benchmarks/speech_off_lead.py <wav dir> <out.json> [processes]
"""

import glob
import json
import math
import sys
import wave
from multiprocessing import Pool

import numpy as np

from RealtimeVAD import VAD

FRAME_S = 512 / 16000


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def noise_dbfs(x):
    """10th percentile of 10 ms block levels: the room between words."""
    blocks = x[: len(x) // 160 * 160].reshape(-1, 160)
    rms = np.sqrt((blocks * blocks).mean(axis=1)) + 1e-9
    return float(20 * np.log10(np.percentile(rms, 10)))


def first_of_final_silent_run(frames, threshold, until):
    """Time of the first frame of the last run of frames below threshold before `until`."""
    t_off = None
    for t, p in frames:
        if t > until:
            break
        if p < threshold:
            if t_off is None:
                t_off = t
        else:
            t_off = None
    return t_off


def vaditerator_end(frames, start_thr=0.5, end_thr=0.35, min_silence_s=0.1):
    """Silero VADIterator's end time with default parameters (end = when it is reported)."""
    triggered, temp_end, last_end = False, None, None
    for t, p in frames:
        if p >= start_thr:
            temp_end = None
            triggered = True
        elif triggered and p < end_thr:
            if temp_end is None:
                temp_end = t - FRAME_S  # Silero records the frame start
            if t - temp_end >= min_silence_s:
                last_end, triggered, temp_end = t, False, None
    return last_end


def measure(path):
    x = load(path)
    vad = VAD(frame_events=True)
    frames, pauses, ends = [], [], []
    for i in range(0, len(x), 320):
        for e in vad.feed(x[i : i + 320]):
            if e.kind == "frame":
                frames.append((e.time, e.frame.probability))
            elif e.kind == "pause":
                pauses.append((e.time, e.source))
            elif e.kind == "end":
                ends.append((e.time, e.segment.start, e.segment.speech_end))
    if not ends or not frames:
        return {"file": path, "skip": "no complete segment"}
    end_t, seg_start, _ = ends[-1]
    seg_pauses = [p for p in pauses if seg_start <= p[0] <= end_t]
    if not seg_pauses:
        return {"file": path, "skip": "no pause"}
    rvad_off, rvad_src = seg_pauses[-1]
    out = {"file": path, "noise_dbfs": round(noise_dbfs(x), 1), "rvad_off": rvad_off, "rvad_source": rvad_src,
           "false_pauses": len(seg_pauses) - 1, "speech_s": round(end_t - seg_start, 2)}
    horizon = end_t + 1.0  # Silero's own speech-off may come after RealtimeVAD's end
    for thr in (0.5, 0.35, 0.45):
        t = first_of_final_silent_run(frames, thr, horizon)
        out[f"silero_off_{thr}"] = t
        out[f"lead_ms_{thr}"] = None if t is None else round((t - rvad_off) * 1000, 1)
    # Silero's own false speech-offs inside the turn: a drop below the threshold that is
    # followed by speech again before RealtimeVAD's final speech-off.
    for thr in (0.5, 0.35):
        below, count = False, 0
        for t, p in frames:
            if t < seg_start or t > rvad_off:
                continue
            if p < thr and not below:
                below = True
            elif p >= thr and below:
                below, count = False, count + 1
        out[f"silero_false_offs_{thr}"] = count
    vi = vaditerator_end([f for f in frames if f[0] >= seg_start - 0.2])
    out["vaditerator_end"] = vi
    out["lead_ms_vaditerator"] = None if vi is None else round((vi - rvad_off) * 1000, 1)
    return out


def main():
    files = sorted(glob.glob(sys.argv[1].rstrip("/\\") + "/*.wav"))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        rows = pool.map(measure, files, chunksize=8)
    json.dump(rows, open(sys.argv[2], "w"), indent=0)
    ok = [r for r in rows if "skip" not in r]
    print(f"{len(files)} files, {len(ok)} measured, {len(files) - len(ok)} skipped")


if __name__ == "__main__":
    main()
