"""Does one long stream get the same "pause" as a fresh detector? Turn by turn.

Every other benchmark here gives each clip a fresh VAD(). This one feeds many turns to ONE
detector as a continuous stream, and each turn also to a fresh detector, and compares them
turn by turn against stream time: the lead of the last "pause" over Silero's own speech-off,
the quiet pauses per second of speech, the quiet threshold at the last "pause", and how far
the stream's last "pause" is from the fresh detector's. What the detector learns while it
runs (the speaker's level, the room, Silero's state) must not make it worse than it was.

    python benchmarks/long_stream.py <dir of turn WAVs> [--turns 400] [--passes 8] ...
    python benchmarks/long_stream.py <one recording.wav>     # e.g. from demo/server.py --record

A directory: its WAVs (16 kHz mono int16, one turn each) in name order, or oldest first with
--order mtime, each followed by --gap-s of silence (white noise at the clip's own room level,
or digital silence). --passes N plays the same turns N times: a detector that does not drift
gives every pass the same numbers. --voice DB puts a second voice after each turn (another
turn, that many dB under -30 dBFS, --voice-gap-s before the next turn), --room DB adds white
room noise to everything.

One recording: the stream is the file; each segment found in it is replayed to a fresh
detector from --lead-in-s before its start. This is how to tell, for a recorded live session
that went bad, whether the detector's learned state did it (stream worse than fresh) or the
audio (both the same).

Both runs use real Silero, so the fresh detector's probabilities differ a little from the
stream's (other state, other frame phase). The lead is measured against each run's own
probabilities; the direct comparison of the two last pauses needs none.
"""

import argparse
import glob
import json
import math
import os
import wave

import numpy as np

from RealtimeVAD import VAD

SR = 16000
DEMO = dict(onset_events=True, start_s=0.32, confirm_threshold=0.8, rearm_s=0.064)  # demo/server.py


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def learned(vad):
    """What the detector has learned, in dBFS (None = nothing yet)."""
    return {"threshold": round(20 * math.log10(max(vad._quiet, 1e-9)), 1), "speaker": vad._level_db,
            "latest_speech": vad._speech_db, "room": vad._noise_db}


def feed(vad, x, t0=0.0):
    """Events of x with times relative to its start (t0 = stream time there)."""
    frames, pauses, ends = [], [], []
    for i in range(0, len(x), 320):
        for e in vad.feed(x[i:i + 320]):
            if e.kind == "frame":
                frames.append((e.time - t0, e.frame.probability))
            elif e.kind == "pause":
                pauses.append((e.time - t0, e.source, learned(vad)))
            elif e.kind == "end":
                ends.append((e.time - t0, e.segment.start - t0))
    return frames, pauses, ends


def turn(frames, pauses, ends, first, last, until=math.inf, threshold=0.45):
    """The last segment that starts in [first, last): its last pause against Silero's speech-off
    (looked for up to 1 s after the end, and not beyond ``until``, where the next speech starts)."""
    ends = [e for e in ends if first <= e[1] < last]
    if not ends:
        return None
    end, start = ends[-1]
    mine = [p for p in pauses if ends[0][1] <= p[0] <= end]
    off = None  # first frame of the final run below the continue threshold
    for t, p in frames:
        if t > min(end + 1.0, until):
            break
        off = (t if off is None else off) if p < threshold else None
    out = {"start": start, "end": end, "speech_s": sum(e[0] - e[1] for e in ends), "silero_off": off,
           "quiet_pauses": sum(p[1] == "quiet" for p in mine)}
    final = [p for p in mine if p[0] >= start]
    if final:
        t, source, state = final[-1]
        out.update(last_pause=t, source=source, learned=state,
                   lead_ms=None if off is None else round((off - t) * 1000, 1))
    return out


def turns_of_directory(a, kw):
    files = glob.glob(os.path.join(a.audio, "*.wav"))
    files.sort(key=os.path.getmtime if a.order == "mtime" else None)
    files = files[a.skip:a.skip + a.turns]
    voices = [load(f) for f in files[:50]] if a.voice is not None else []
    rng = np.random.default_rng(1)
    stream, t0, rows = VAD(frame_events=True, **kw), 0.0, []
    for k, path in enumerate(files * a.passes):
        x = load(path)
        n = int(a.gap_s * SR)
        blocks = x[: len(x) // 160 * 160].reshape(-1, 160)
        room = float(np.percentile(np.sqrt((blocks * blocks).mean(axis=1)), 10)) if len(blocks) else 0.0
        if a.gap == "digital" or room < 10 ** (-90 / 20):
            gap = np.zeros(n, np.float32)
        else:
            gap = (rng.standard_normal(n) * room).astype(np.float32)
        if voices:  # 1 s after the turn; then --voice-gap-s (default: the whole gap) to the next turn
            v = voices[(k * 7 + 3) % len(voices)]
            loud = math.sqrt(float(np.mean(np.sort(v * v)[-len(v) // 4:]))) + 1e-9
            v = v * np.float32(10 ** ((a.voice - 30) / 20) / loud)
            gap = np.concatenate((gap[:SR], v, gap if a.voice_gap_s is None else gap[:int(a.voice_gap_s * SR)]))
        y = np.concatenate((x, gap))
        if a.room is not None:
            y = (y + rng.standard_normal(len(y)) * 10 ** (a.room / 20)).astype(np.float32)
        before = learned(stream)
        s = turn(*feed(stream, y, t0), 0.0, len(x) / SR)
        f = turn(*feed(VAD(frame_events=True, **kw), y), 0.0, len(x) / SR)
        rows.append({"turn": k, "pass": k // len(files), "file": os.path.basename(path), "t": round(t0, 2),
                     "before": before, "stream": s, "fresh": f})
        t0 += len(y) / SR
    return rows, t0


def turns_of_recording(a, kw):
    x = load(a.audio)
    frames, pauses, ends = feed(VAD(frame_events=True, **kw), x)
    rows = []
    for k, (end, start) in enumerate(ends):
        until = ends[k + 1][1] if k + 1 < len(ends) else math.inf  # the next segment's start
        s = turn(frames, pauses, ends, start, start + 1e-6, until)
        t0 = max(0.0, start - a.lead_in_s)
        part = x[int(t0 * SR):int((end + 1.0) * SR)]
        f = turn(*feed(VAD(frame_events=True, **kw), part), start - t0 - 0.2, min(until - 0.1, end) - t0, until - t0)
        for key in ("start", "end", "last_pause", "silero_off"):  # stream times, like the stream's
            if f and f.get(key) is not None:
                f[key] += t0
        rows.append({"turn": k, "pass": 0, "file": os.path.basename(a.audio), "t": round(start, 2),
                     "before": None, "stream": s, "fresh": f})
    return rows, len(x) / SR


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else float("nan")


def block(rows):
    """One table row: the stream and the fresh detector over these turns."""
    def of(who, key):
        return [r[who].get(key) if r[who] else None for r in rows]

    def state(who, key):
        return [r[who]["learned"][key] if r[who] and "learned" in r[who] else None for r in rows]

    def per_s(who):
        speech = sum(r[who]["speech_s"] for r in rows if r[who])
        return sum(r[who]["quiet_pauses"] for r in rows if r[who]) / speech if speech else float("nan")

    def earlier(who):
        leads = [v for v in of(who, "lead_ms") if v is not None]
        return 100 * sum(v > 0 for v in leads) / len(leads) if leads else float("nan")

    both = [r for r in rows if r["stream"] and r["fresh"] and "last_pause" in r["stream"] and "last_pause" in r["fresh"]]
    shift = [(r["stream"]["last_pause"] - r["fresh"]["last_pause"]) * 1000 for r in both]
    n = max(1, len(shift))
    return {"turns": len(rows), "minute": rows[0]["t"] / 60,
            "lead": (median(of("stream", "lead_ms")), median(of("fresh", "lead_ms"))),
            "earlier": (earlier("stream"), earlier("fresh")),
            "missed": (sum(not r["stream"] for r in rows), sum(not r["fresh"] for r in rows)),
            "pauses_per_s": (per_s("stream"), per_s("fresh")),
            "threshold": (median(state("stream", "threshold")), median(state("fresh", "threshold"))),
            "speaker": (median(state("stream", "speaker")), median(state("fresh", "speaker"))),
            "shift": median(shift), "later": 100 * sum(v > 30 for v in shift) / n,
            "sooner": 100 * sum(v < -30 for v in shift) / n}


def report(rows, seconds, passes):
    if not rows:
        print("no turns found")
        return []
    if passes > 1:
        groups = [[r for r in rows if r["pass"] == p] for p in range(passes)]
        name = "pass"
    else:
        size = max(1, math.ceil(len(rows) / 10))
        groups = [rows[i:i + size] for i in range(0, len(rows), size)]
        name = "turns"
    out = [block(g) for g in groups]
    print(f"{len(rows)} turns in one stream of {seconds / 60:.1f} min. S = the one long-running detector, "
          f"F = a fresh detector per turn.\n")
    print(f"| {name} | from min | lead over Silero, median ms S / F | earlier than Silero % S / F | quiet pauses per s S / F "
          "| quiet threshold dBFS S / F | speaker level dBFS S / F | last pause S minus F, median ms "
          "| S more than 30 ms later % | S more than 30 ms sooner % | no segment S / F |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    first = 0
    for i, b in enumerate(out):
        label = str(i + 1) if passes > 1 else f"{first + 1}-{first + b['turns']}"
        first += b["turns"]
        print(f"| {label} | {b['minute']:.1f} | {b['lead'][0]:.0f} / {b['lead'][1]:.0f} | {b['earlier'][0]:.0f} / {b['earlier'][1]:.0f} "
              f"| {b['pauses_per_s'][0]:.2f} / {b['pauses_per_s'][1]:.2f} | {b['threshold'][0]:.1f} / {b['threshold'][1]:.1f} "
              f"| {b['speaker'][0]:.1f} / {b['speaker'][1]:.1f} | {b['shift']:+.1f} | {b['later']:.1f} | {b['sooner']:.1f} "
              f"| {b['missed'][0]} / {b['missed'][1]} |")
    a, z = out[0], out[-1]
    print(f"\nFirst against last row, the long-running detector: lead {a['lead'][0]:.0f} -> {z['lead'][0]:.0f} ms, "
          f"quiet pauses {a['pauses_per_s'][0]:.2f} -> {z['pauses_per_s'][0]:.2f} per s, "
          f"last pause against the fresh detector {a['shift']:+.1f} -> {z['shift']:+.1f} ms.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="a directory of turn WAVs, or one long recording")
    ap.add_argument("--turns", type=int, default=400)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--order", choices=("name", "mtime"), default="name")
    ap.add_argument("--passes", type=int, default=1, help="play the same turns this often")
    ap.add_argument("--gap", choices=("room", "digital"), default="room")
    ap.add_argument("--gap-s", type=float, default=2.0)
    ap.add_argument("--voice", type=float, help="a second voice after each turn, dB relative to -30 dBFS")
    ap.add_argument("--voice-gap-s", type=float, help="silence between that voice and the next turn")
    ap.add_argument("--room", type=float, help="white room noise over everything, dBFS")
    ap.add_argument("--lead-in-s", type=float, default=2.0, help="recording: audio a fresh detector gets before a segment")
    ap.add_argument("--demo", action="store_true", help="the live demo's detector settings")
    ap.add_argument("--options", default="{}", help="VAD arguments as JSON")
    ap.add_argument("--json", help="write every turn to this file")
    a = ap.parse_args()
    kw = {**(DEMO if a.demo else {}), **json.loads(a.options)}
    if os.path.isdir(a.audio):
        rows, seconds = turns_of_directory(a, kw)
    else:
        rows, seconds = turns_of_recording(a, kw)
        a.passes = 1
    report(rows, seconds, a.passes)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    main()
