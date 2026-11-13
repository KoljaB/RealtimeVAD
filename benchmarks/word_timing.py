"""RealtimeVAD's events against word timings, on many speakers.

Clips come from tests/audio/fetch_word_timing.py: turn-like stretches of one speaker's words
with their real silence before and after, from the AMI Meeting Corpus (own headset, a
distant table microphone, own headset with other people talking) and LibriSpeech (clean and
"other"), each with word start and end times. Every clip is fed to a fresh detector.

Three detectors see the same audio:

  rvad      VAD()               RealtimeVAD with its defaults
  silero    VAD(quiet_rms=0)    same thresholds, quiet clock off: Silero's silence count alone
  iterator  Silero's VADIterator with its defaults (start at 0.5, end after 100 ms below
            0.35), computed from rvad's frame probabilities exactly as silero_vad does it

and "Silero p < 0.5" is the first frame below 0.5 after frames at or above it.

What is measured (per clip; the report groups by condition and by speaker):

  pauses     where every `pause` falls by the labels (inside a word and how deep, in a gap and
             how long the gap is, after the last word) and how long the audio then stayed
             quiet (10 ms blocks the quiet clock counted before a louder one)
  speech-off the last `pause` before the last `end`: against the last word's labelled end,
             and against Silero p < 0.5 on the same frames (the lead)
  ends       every `end` with words still to come (>= 100 ms of them), per detector; speech
             outside all segments' audio
  starts     when `start` is decided, against the first word's labelled start
  delay      what a confirmation delay on `pause` would do, replayed from the quiet clock's
             public state: a pause counts only once N quiet blocks have followed without a
             break. With N = 1 the replay reproduces the library's events (checked).

Labels are forced alignments. The report never reads a label more finely than it deserves:
"inside a word" is given at several depths, and for the clips' own acoustics every clip
carries the level of the speech, of the room, and where the audio around the labelled end
of the last word actually falls to the room level.

    python benchmarks/word_timing.py                 # measure and print the tables
    python benchmarks/word_timing.py --report        # tables from the saved rows
    python benchmarks/word_timing.py --examples 5    # also list example clips per failure pattern
    python benchmarks/word_timing.py --clip ID       # one clip, event by event
    python benchmarks/word_timing.py --limit 3 --options '{"speech_margin_db": 8}'   # other settings, every 3rd clip

Rows go to benchmarks/word_timing_rows.json (git-ignored). About 10 h of audio: 2 minutes
with --processes 16, an estimated 5 with the default 6. The workers run at below-normal
priority so that a shared machine stays usable.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import sys
import time
import wave
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "audio" / "data" / "word_timing"
ROWS = Path(__file__).resolve().parent / "word_timing_rows.json"

SR, FRAME, BLOCK = 16000, 512, 160
FRAME_S, BLOCK_S = FRAME / SR, BLOCK / SR
OPTIONS: dict = {}  # extra VAD arguments for the rvad detector (--options)
DELAYS = (1, 2, 3, 4, 5, 6, 7, 9, 11)  # quiet blocks needed before a pause counts (1 = as shipped)
MORE_S = 0.1  # an end is "mid-utterance" when at least this much labelled speech follows
CONDITIONS = ("libri_clean", "libri_other", "ami_headset", "ami_headset_xtalk", "ami_far")
NAMES = {"libri_clean": "LibriSpeech clean", "libri_other": "LibriSpeech other",
         "ami_headset": "AMI headset, alone", "ami_headset_xtalk": "AMI headset, others talk",
         "ami_far": "AMI distant mic, alone"}


def low_priority() -> None:
    """Below-normal priority for this process: the benchmark must not starve a desktop."""
    try:
        if sys.platform == "win32":
            import ctypes

            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:  # priority is a courtesy, never a reason to fail
        pass


def _options(options: dict) -> None:
    low_priority()
    OPTIONS.update(options)  # in every worker process


def load(path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


# ------------------------------------------------------------------ detectors


def run(x: np.ndarray, **options):
    """Feed x boundary by boundary; return events and a trace of the quiet clock's state."""
    from RealtimeVAD import VAD

    try:
        vad = VAD(frame_events=True, resume_events=True, **options)
    except TypeError:  # a library without the "resume" event
        vad = VAD(frame_events=True, **options)
    frames, starts, pauses, ends, trace, resumes = [], [], [], [], [], []
    n = 0
    while n < len(x):
        step = min(BLOCK - n % BLOCK, FRAME - n % FRAME, len(x) - n)
        events = vad.feed(x[n:n + step])
        n += step
        t = n / SR
        seen, begun = [], None
        for e in events:
            if e.kind == "frame":
                f = e.frame
                frames.append((e.time, f.probability))
                seen.append(("f", f.voiced, f.silent_frames, f.in_speech))
                if begun is not None:  # the frame that decided the start
                    starts.append((begun, e.time))
                    begun = None
            elif e.kind == "start":
                begun = e.time
                seen.append(("s",))
            elif e.kind == "pause":
                pauses.append((e.time, e.source))
            elif e.kind == "resume":
                resumes.append(e.time)
            elif e.kind == "end":
                ends.append((e.time, e.segment.speech_end, e.time - len(e.segment.audio) / SR, bool(getattr(e.segment, "early", False))))
                seen.append(("e",))
        armed, unbroken, blocks = vad.quiet_clock
        trace.append((t, n % BLOCK == 0, seen, vad.in_speech, armed, unbroken, blocks))
    still = None
    if vad.in_speech:
        still = n / SR - len(vad.audio()) / SR
    return {"frames": frames, "starts": starts, "pauses": pauses, "ends": ends, "open": still, "trace": trace,
            "resumes": resumes}


def replay(trace, need: int):
    """The library's pause events if a quiet-clock pause needed `need` unbroken quiet blocks."""
    out, paused, was_armed, silent = [], False, False, 0
    for t, block_done, seen, in_speech, armed, unbroken, blocks in trace:
        ended = any(s[0] == "e" for s in seen)
        if block_done and was_armed and not armed and not ended:
            paused = False  # enough sound re-armed the clock
        for s in seen:
            if s[0] == "f":
                _, voiced, silent_now, speaking = s
                if speaking:
                    if voiced and silent:
                        paused = False
                    if silent_now == 1 and not paused:
                        out.append((t, "silero"))
                        paused = True
                    silent = silent_now
                else:
                    silent = 0
            else:  # start or end: a fresh segment state
                paused, silent = False, 0
        if block_done and in_speech and armed and unbroken and blocks >= need and not paused:
            out.append((t, "quiet"))
            paused = True
        was_armed = armed and in_speech
    return out


def quiet_runs(trace, pauses):
    """Per pause: seconds the audio stayed quiet (or unvoiced) after it, and whether that ran into the end."""
    times = [r[0] for r in trace]
    out = []
    for t, source in pauses:
        i = bisect.bisect_left(times, t - 1e-9)
        length, to_end = 0.0, False
        if source == "quiet":
            for r in trace[i:]:
                if any(s[0] == "e" for s in r[2]):
                    to_end = True
                    break
                if r[1]:
                    if not (r[4] and r[5]):
                        break
                    length = r[6] * BLOCK_S
        else:
            first = True
            for r in trace[i:]:
                stop = False
                for s in r[2]:
                    if s[0] == "e":
                        to_end = stop = True
                    elif s[0] == "f":
                        if first:
                            first = False
                            length = FRAME_S
                        elif s[2] > 0:
                            length += FRAME_S
                        else:
                            stop = True
                    if stop:
                        break
                if stop:
                    break
        out.append((round(length, 3), to_end))
    return out


def iterator_segments(frames, threshold=0.5, min_silence_s=0.1):
    """silero_vad.VADIterator on these probabilities: (start decided, end decided or None, speech start, speech end)."""
    low = max(threshold - 0.15, 0.01)
    out, triggered, temp_end, begun = [], False, None, None
    for t, p in frames:  # t is the end of the frame, VADIterator's current_sample
        if p >= threshold:
            temp_end = None
            if not triggered:
                triggered, begun = True, t
        elif p < low and triggered:
            if temp_end is None:
                temp_end = t
            if t - temp_end >= min_silence_s - 1e-9:
                out.append((begun, t, begun - FRAME_S, temp_end - FRAME_S))
                triggered, temp_end = False, None
    if triggered:
        out.append((begun, None, begun - FRAME_S, None))
    return out


def first_of_final_low_run(frames, threshold, until):
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


# --------------------------------------------------------------------- labels


class Words:
    def __init__(self, words):
        self.s = [w[1] for w in words]
        self.e = [w[2] for w in words]
        self.first, self.last = self.s[0], self.e[-1]

    def where(self, t):
        """("pre",) | ("word", index, seconds to its end, seconds since its start)
        | ("gap", index of the word before, gap length, seconds to the next word) | ("post", seconds after)."""
        if t <= self.first:
            return ("pre",)
        if t >= self.last:
            return ("post", round(t - self.last, 3))
        i = bisect.bisect_right(self.s, t) - 1
        if t < self.e[i]:
            return ("word", i, round(self.e[i] - t, 3), round(t - self.s[i], 3))
        return ("gap", i, round(self.s[i + 1] - self.e[i], 3), round(self.s[i + 1] - t, 3))

    def after(self, t) -> float:
        """Labelled speech after t, in seconds."""
        return sum(max(0.0, e - max(s, t)) for s, e in zip(self.s, self.e))

    def outside(self, spans) -> float:
        """Labelled speech covered by none of the (start, end) spans."""
        lost = 0.0
        for s, e in zip(self.s, self.e):
            cut = [(max(s, a), min(e, b)) for a, b in spans if a < e and b > s]
            covered, at = 0.0, s
            for a, b in sorted(cut):
                a = max(a, at)
                if b > a:
                    covered += b - a
                    at = b
            lost += (e - s) - covered
        return lost


def acoustics(x, words: Words):
    n = len(x) // BLOCK
    blocks = x[: n * BLOCK].reshape(n, BLOCK)
    db = 10 * np.log10(np.maximum((blocks * blocks).mean(axis=1), 1e-10))
    noise = float(np.percentile(db, 10))
    in_word = np.zeros(n, bool)
    for s, e in zip(words.s, words.e):
        in_word[int(s / BLOCK_S): int(np.ceil(e / BLOCK_S))] = True
    speech = float(np.median(db[in_word])) if in_word.any() else noise
    # the edge of the sound nearest to a label: sound is a block clearly above the room and
    # not far below the speaker; gaps of up to 30 ms belong to it
    audible = db >= max(noise + 6.0, speech - 25.0)

    def edge(t, step):
        i = min(n - 1, max(0, int(t / BLOCK_S) - (step > 0)))  # the last block of the word / its first
        reach = int(0.4 / BLOCK_S)
        if audible[max(0, i - 2): i + 3].any():  # the label lies in or at the sound: follow it outwards
            k, quiet = i, 0
            for j in range(i, max(-1, min(n, i + step * reach)), step):
                if audible[j]:
                    k, quiet = j, 0
                else:
                    quiet += 1
                    if quiet > 3:
                        break
        else:  # the label lies in silence: back to the sound
            k = next((j for j in range(i, max(-1, min(n, i - step * reach)), -step) if audible[j]), None)
            if k is None:
                return None
        return round((k + (step > 0)) * BLOCK_S, 3)

    sound_end, sound_start = edge(words.last, 1), edge(words.first, -1)
    tail = db[int((words.last + 0.05) / BLOCK_S): int((words.last + 0.30) / BLOCK_S)]
    return db, {"noise_db": round(noise, 1), "speech_db": round(speech, 1),
                "sound_end": sound_end, "sound_start": sound_start,
                "tail_db": round(float(tail.mean()), 1) if len(tail) else None,
                "dc": round(float(abs(x.mean())), 5),
                # the room without a DC offset (the levels above are what the quiet clock sees)
                "noise_ac_db": round(float(np.percentile(10 * np.log10(np.maximum(
                    ((blocks - x.mean()) ** 2).mean(axis=1), 1e-10)), 10)), 1)}


def audible_s(db, t0, t1, level) -> float:
    """Seconds of 10 ms blocks at or above `level` between t0 and t1."""
    a, b = int(round(t0 / BLOCK_S)), int(round(t1 / BLOCK_S))
    return round(float((db[a:b] >= level).sum()) * BLOCK_S, 3) if b > a else 0.0


def lost_audible(db, words: "Words", spans, level) -> float:
    """Seconds of audible blocks inside labelled words that no (start, end) span covers."""
    keep = np.zeros(len(db), bool)
    for s, e in zip(words.s, words.e):
        keep[int(s / BLOCK_S): int(np.ceil(e / BLOCK_S))] = True
    for a, b in spans:
        keep[max(0, int(np.floor(a / BLOCK_S))): max(0, int(np.ceil(b / BLOCK_S)))] = False
    return round(float((keep & (db >= level)).sum()) * BLOCK_S, 3)


def loud_again(db, t, level, horizon_s=1.0):
    """Seconds from t until a 10 ms block reaches `level` again, or None."""
    i = int(round(t / BLOCK_S))
    hit = np.nonzero(db[i: i + int(horizon_s / BLOCK_S)] >= level)[0]
    return round(float(hit[0]) * BLOCK_S, 3) if len(hit) else None


# -------------------------------------------------------------------- measure


def segments_of(r, clip_end):
    """(start decided, end decided or None, audio start, audio end) per segment of a VAD run."""
    out = []
    for k, (start, decided) in enumerate(r["starts"]):
        if k < len(r["ends"]):
            out.append((decided, r["ends"][k][0], r["ends"][k][2], r["ends"][k][0]))
        else:
            out.append((decided, None, r["open"], clip_end))
    return out


def ends_of(segs, words: Words, frames):
    """Per end with labelled speech still to come: (time, speech still to come, seconds to
    the next word by label, seconds to Silero's next p >= 0.5)."""
    out = []
    times = [f[0] for f in frames]
    for seg in segs:
        t = seg[1]
        if t is None:
            continue
        more = words.after(t)
        if more < MORE_S:
            continue
        w = words.where(t)
        to_next = 0.0 if w[0] == "word" else w[3] if w[0] == "gap" else round(words.first - t, 3)
        resumed = next((round(ft - t, 3) for ft, p in frames[bisect.bisect_right(times, t):] if p >= 0.5), None)
        out.append((round(t, 3), round(more, 3), to_next, resumed))
    return out


def first_start(segs, words: Words):
    """Seconds from the first word's labelled start to the decision that opens the first
    segment reaching into the utterance; None if there is none."""
    for seg in segs:
        end = seg[1] if seg[1] is not None else 1e9
        if end > words.first and seg[0] < words.last + 0.3:
            return round(seg[0] - words.first, 3)
    return None


def measure(clip):
    x = load(DATA / clip["path"])
    words = Words(clip["words"])
    clip_end = len(x) / SR
    db, row = acoustics(x, words)
    row.update(id=clip["id"], condition=clip["condition"], speaker=clip["speaker"], gender=clip.get("gender"),
               n_words=len(clip["words"]), first=words.first, last=words.last, clip_s=round(clip_end, 3))
    a = run(x, **OPTIONS)
    b = run(x, **{**OPTIONS, "quiet_rms": 0})
    frames = a["frames"]
    row["p_differs"] = sum(1 for (_, p), (_, q) in zip(frames, b["frames"]) if abs(p - q) > 1e-6)
    row["p_max"] = round(max((p for t, p in frames if words.first <= t <= words.last + 0.1), default=0.0), 3)

    # pauses, by label and by what the audio did next
    loud = row["speech_db"] - 12.0
    runs = quiet_runs(a["trace"], a["pauses"])
    row["pauses"] = []
    for (t, source), (quiet_s, to_end) in zip(a["pauses"], runs):
        row["pauses"].append([round(t, 3), source[0], quiet_s, int(to_end), loud_again(db, t, loud), words.where(t)])
    # every "resume": its time, the pause it takes back, and for a quiet-clock pause how long
    # after the first block that was not quiet again (the quiet count's first break) it came
    row["resumes"] = []
    for t in a["resumes"]:
        before = [p for p in row["pauses"] if p[0] <= t + 1e-6]
        if before:
            pt, src, quiet_s = before[-1][:3]
            back = round(t - (pt - BLOCK / SR + quiet_s), 3) if src == "q" and not before[-1][3] else None
            row["resumes"].append([round(t, 3), pt, back])
    offs, above, low_run = [], False, 0
    for i, (t, p) in enumerate(frames):  # Silero p < 0.5 after speech
        if p >= 0.5:
            above = True
        elif above:
            above = False
            k = i
            while k < len(frames) and frames[k][1] < 0.5:
                k += 1
            offs.append([round(t, 3), "p", round((k - i) * FRAME_S, 3), int(k == len(frames)),
                         loud_again(db, t, loud), words.where(t)])
    row["silero_offs"] = offs

    # replay with a confirmation delay
    row["replay_ok"] = replay(a["trace"], 1) == a["pauses"]
    row["delayed"] = {n: [round(t, 3) for t, _ in replay(a["trace"], n)] for n in DELAYS}

    # speech-off at the last end, as in benchmarks/speech_off_lead.py
    seg_a, seg_b, seg_i = segments_of(a, clip_end), segments_of(b, clip_end), iterator_segments(frames)
    soft = max(row["noise_db"] + 6.0, row["speech_db"] - 25.0)
    # every RealtimeVAD end: time, where the speech stopped for it, whether the quiet clock
    # made it (Segment.early; only with early_end=True: it then comes 0.24 or 0.28 s after that point, Silero's count 0.256 s after
    # its last voiced frame), and the labelled speech and audible sound inside that wait
    row["rvad_ends"] = []
    for t, speech_end, _, early in a["ends"]:
        row["rvad_ends"].append([round(t, 3), round(speech_end, 3), int(early),
                                 round(words.after(speech_end) - words.after(t), 3),
                                 audible_s(db, speech_end, t, soft), words.where(speech_end),
                                 round(min((p for ft, p in frames if speech_end + 0.1 < ft <= t + FRAME_S), default=1.0), 3)])
    if a["ends"]:
        end_t = a["ends"][-1][0]
        row["final_end"] = round(end_t, 3)
        row["final_speech_end"] = round(a["ends"][-1][1], 3)
        row["final_start"] = round(seg_a[len(a["ends"]) - 1][0], 3)
        row["final_by_clock"] = bool(row["rvad_ends"][-1][2])
        fp = [p[0] for p in a["pauses"] if row["final_start"] - 0.4 <= p[0] <= end_t]
        if fp:
            # times the quiet count was broken between the speech-off and the end, and the
            # audible sound of the last word that came after the speech-off
            breaks, was = 0, False
            for r in a["trace"]:
                if fp[-1] <= r[0] <= end_t and r[1]:
                    breaks += was and r[4] and not r[5]
                    was = r[4] and r[5]
            row["final_breaks"] = breaks
            row["final_residual"] = audible_s(db, fp[-1], words.last, soft)
        for thr in (0.5, 0.35):
            off = first_of_final_low_run(frames, thr, end_t + 1.0)
            row[f"silero_off_{thr}"] = None if off is None else round(off, 3)
        if row.get("silero_off_0.5") is not None:
            row["silero_residual"] = audible_s(db, row["silero_off_0.5"], words.last, soft)
    row["ends"] = {"rvad": [round(e[0], 3) for e in a["ends"]], "silero": [round(e[0], 3) for e in b["ends"]],
                   "iterator": [round(s[1], 3) for s in seg_i if s[1] is not None]}
    row["open"] = {"rvad": a["open"] is not None, "silero": b["open"] is not None,
                   "iterator": bool(seg_i) and seg_i[-1][1] is None}
    row["mid_ends"] = {"rvad": ends_of(seg_a, words, frames), "silero": ends_of(seg_b, words, frames),
                       "iterator": ends_of([(s[0], s[1]) for s in seg_i], words, frames)}
    pad = 0.03
    row["lost"] = {"rvad": round(words.outside([(s[2], s[3]) for s in seg_a]), 3),
                   "silero": round(words.outside([(s[2], s[3]) for s in seg_b]), 3),
                   "iterator": round(words.outside([(s[2] - pad, clip_end if s[3] is None else s[3] + pad)
                                                    for s in seg_i]), 3)}
    row["lost_audible"] = {"rvad": lost_audible(db, words, [(s[2], s[3]) for s in seg_a], soft),
                           "silero": lost_audible(db, words, [(s[2], s[3]) for s in seg_b], soft),
                           "iterator": lost_audible(db, words, [(s[2] - pad, clip_end if s[3] is None else s[3] + pad)
                                                                for s in seg_i], soft)}
    row["start"] = {"rvad": first_start(seg_a, words), "silero": first_start(seg_b, words),
                    "iterator": first_start([(s[0], s[1]) for s in seg_i], words)}
    if clip["corpus"] == "ami":  # quiet recordings: the README's advice for quiet input
        c = run(x, **{**OPTIONS, "normalize_to_dbfs": -30.0})
        seg_c = segments_of(c, clip_end)
        row["start"]["norm"] = first_start(seg_c, words)
        row["lost_audible"]["norm"] = lost_audible(db, words, [(s[2], s[3]) for s in seg_c], soft)
        row["mid_ends"]["norm"] = ends_of(seg_c, words, c["frames"])
    return row


def show(clip) -> None:
    """One clip, event by event: for reproducing a failure."""
    x = load(DATA / clip["path"])
    words = Words(clip["words"])
    db, ac = acoustics(x, words)
    print(clip["id"], clip["condition"], clip["speaker"], f"{len(x) / SR:.2f} s", DATA / clip["path"])
    print("  ", ac)
    a, b = run(x), run(x, quiet_rms=0)
    lines = [(w[1], f"word  {w[0]!r} until {w[2]:.3f}") for w in clip["words"]]
    lines += [(d, f"rvad start (reported {s:.3f})") for s, d in a["starts"]]
    lines += [(t, f"rvad pause ({src})") for t, src in a["pauses"]]
    lines += [(e[0], f"rvad END (speech end {e[1]:.3f})") for e in a["ends"]]
    lines += [(e[0], "silero-alone END") for e in b["ends"]]
    lines += [(s[1], "VADIterator end") for s in iterator_segments(a["frames"]) if s[1] is not None]
    prev = 1.0
    for t, p in a["frames"]:
        if (p < 0.5) != (prev < 0.5):
            lines.append((t, f"silero p {'<' if p < 0.5 else '>='} 0.5"))
        prev = p
    for t, text in sorted(lines):
        i = min(len(db) - 1, int(t / BLOCK_S))
        print(f"  {t:7.3f}  {db[i]:6.1f} dB  {text}")


# --------------------------------------------------------------------- report


def q(values, p):
    return float(np.percentile(values, p)) if len(values) else float("nan")


def ms(v) -> str:
    return "–" if v != v else f"{v * 1000:.0f}"


def spread(per_speaker) -> str:
    """Quartiles across speakers of a per-speaker value (in ms)."""
    v = list(per_speaker)
    if not v:
        return "–"
    return f"{ms(q(v, 25))} / {ms(q(v, 50))} / {ms(q(v, 75))} (min {ms(min(v))}, max {ms(max(v))})"


def table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pct(n, d) -> str:
    return "–" if not d else f"{100 * n / d:.1f} %"


def by_speaker(rows, value, least=8):
    """value(rows of one speaker) for every speaker with at least `least` rows."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["speaker"]].append(r)
    return {s: value(rs) for s, rs in per.items() if len(rs) >= least}


def final_pause(r, pauses=None):
    """The last pause before the clip's last end (the speech-off), or None."""
    if "final_end" not in r:
        return None
    ts = [p[0] for p in r["pauses"]] if pauses is None else pauses
    ts = [t for t in ts if r["final_start"] - 0.4 <= t <= r["final_end"]]
    return ts[-1] if ts else None


def lead(r, thr="0.5", pauses=None):
    off, fp = r.get(f"silero_off_{thr}"), final_pause(r, pauses)
    return None if off is None or fp is None else off - fp


def clean_final(r) -> bool:
    """The clip's last end closes the utterance: it comes after the last word, within 1 s."""
    return "final_end" in r and r["last"] < r["final_end"] <= r["last"] + 1.0


def end_lead(r):
    """Silero alone's last end minus RealtimeVAD's, where both closed the same segments; else None."""
    e = r["ends"]
    if not clean_final(r) or not e["silero"] or len(e["silero"]) != len(e["rvad"]) or r["open"]["rvad"] or r["open"]["silero"]:
        return None
    return e["silero"][-1] - e["rvad"][-1]


def margin(r):
    """dB by which the 50-300 ms after the last word lie under the quiet floor (the lower of
    quiet_rms = -54 dBFS and 20 dB below the speech level, measured here on the labelled words)."""
    return None if r["tail_db"] is None else min(-54.0, r["speech_db"] - 20.0) - r["tail_db"]


def false_pauses(r, pauses=None):
    """Pauses after the first word began that are not the speech-off."""
    fp = final_pause(r, pauses)
    ts = [p[0] for p in r["pauses"]] if pauses is None else pauses
    return [t for t in ts if t > r["first"] and t < (fp or 0)]


def own_ends(r):
    """RealtimeVAD's mid-utterance ends that Silero alone does not make within 0.4 s."""
    return [e for e in r["mid_ends"]["rvad"] if not any(-0.05 <= s - e[0] <= 0.4 for s in r["ends"]["silero"])]


def report(rows) -> str:
    out = []
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    R = {c: [r for r in rows if r["condition"] == c] for c in conds}
    F = {c: [r for r in R[c] if clean_final(r)] for c in conds}
    speech = {c: sum(r["last"] - r["first"] for r in R[c]) for c in conds}

    out.append("### Data\n")
    out.append(table(
        ["Condition", "clips", "speakers", "labelled speech", "speech level: median (p10 – p90 of clips)",
         "room level", "speech above room"],
        [[NAMES[c], len(R[c]), len({r["speaker"] for r in R[c]}), f"{speech[c] / 3600:.1f} h",
          f"{q([r['speech_db'] for r in R[c]], 50):.0f} dBFS ({q([r['speech_db'] for r in R[c]], 10):.0f} – {q([r['speech_db'] for r in R[c]], 90):.0f})",
          f"{q([r['noise_db'] for r in R[c]], 50):.0f} dBFS",
          f"{q([r['speech_db'] - r['noise_db'] for r in R[c]], 50):.0f} dB"] for c in conds]))
    bad = sum(not r["replay_ok"] for r in rows)
    differs = [r["id"] for r in rows if r["p_differs"] > 0]
    out.append(f"Checks: the delay replay reproduces the library's `pause` events in {len(rows) - bad} of {len(rows)} clips. "
               f"Silero's probabilities differ between RealtimeVAD and Silero alone in {len(differs)} clips"
               + (f" ({', '.join(differs)})" if differs else "") + ".\n")

    out.append("### Labels against the audio\n")
    t = []
    for c in conds:
        d = [r["last"] - r["sound_end"] for r in R[c] if r["sound_end"] is not None]
        s = [r["sound_start"] - r["first"] for r in R[c] if r["sound_start"] is not None]
        t.append([NAMES[c], f"{ms(q(d, 50))} ({ms(q(d, 10))} – {ms(q(d, 90))})", pct(sum(x > 0.05 for x in d), len(d)),
                  pct(sum(x < -0.05 for x in d), len(d)), f"{ms(q(s, 50))} ({ms(q(s, 10))} – {ms(q(s, 90))})"])
    out.append(table(["Condition", "last word: labelled end minus end of the sound, median (p10 – p90) ms",
                      "label > 50 ms after the sound", "label > 50 ms before it",
                      "first word: start of the sound minus labelled start, ms"], t))

    out.append("### Where `pause` falls\n")
    for who, key in (("RealtimeVAD `pause`", "pauses"), ("Silero p < 0.5", "silero_offs")):
        t = []
        for c in conds:
            ev = [p for r in R[c] for p in r[key] if p[5][0] != "pre"]
            n = len(ev)
            word = [p for p in ev if p[5][0] == "word"]
            deep = [p for p in word if p[5][2] >= 0.05 and p[5][3] >= 0.05]
            back = [p for p in deep if not p[3] and p[2] < 0.1]
            depth = [p[5][2] for p in deep]
            t.append([NAMES[c], n, f"{n / speech[c]:.2f}", pct(len(word), n), pct(len(deep), n),
                      f"{ms(q(depth, 50))} ({ms(q(depth, 25))} – {ms(q(depth, 75))})", pct(len(back), n),
                      pct(sum(p[5][0] == "gap" for p in ev), n), pct(sum(p[5][0] == "post" for p in ev), n)])
        out.append(f"{who}:\n")
        out.append(table(["Condition", "events", "per s of speech", "inside a labelled word", "… ≥ 50 ms from both its edges",
                          "… ms before the word's end: median (p25 – p75)", "… and quiet for < 100 ms",
                          "in a labelled gap", "after the last word"], t))

    out.append("### Speech-off against the end of the last word\n")
    t = []
    for c in conds:
        for name, f, res in (("RealtimeVAD", final_pause, "final_residual"),
                             ("Silero p < 0.5", lambda r: r.get("silero_off_0.5"), "silero_residual")):
            d = [(f(r) - r["last"], r.get(res, 0.0)) for r in F[c] if f(r) is not None]
            v = [x for x, _ in d]
            t.append([NAMES[c], name, len(d), f"{ms(q(v, 50))} ({ms(q(v, 10))} – {ms(q(v, 90))})",
                      pct(sum(x < -0.05 for x in v), len(d)), pct(sum(x < -0.1 for x in v), len(d)),
                      pct(sum(x < -0.1 and a >= 0.03 for x, a in d), len(d)),
                      pct(sum(x < -0.1 and a >= 0.1 for x, a in d), len(d)), pct(sum(x < -0.2 for x in v), len(d))])
    out.append(table(["Condition", "speech-off of", "clips", "relative to the labelled end: median (p10 – p90) ms",
                      "> 50 ms before it", "> 100 ms before it", "… with ≥ 30 ms of audible sound still to come",
                      "… with ≥ 100 ms", "> 200 ms before it"], t))

    out.append("### Lead over Silero\n")
    t = []
    for c in conds:
        l5 = [lead(r) for r in F[c] if lead(r) is not None]
        l35 = [lead(r, "0.35") for r in F[c] if lead(r, "0.35") is not None]
        sp = by_speaker(F[c], lambda rs: q([lead(r) for r in rs if lead(r) is not None], 50))
        sp = {k: v for k, v in sp.items() if v == v}
        t.append([NAMES[c], len(l5), f"{ms(q(l5, 50))} ({ms(q(l5, 25))} – {ms(q(l5, 75))})", pct(sum(x > 0 for x in l5), len(l5)),
                  f"{ms(q(l35, 50))}", len(sp), spread(sp.values()), sum(v <= 0.0 for v in sp.values())])
    out.append(table(["Condition", "clips", "speech-off lead over p < 0.5: median (p25 – p75) ms", "earlier in",
                      "over p < 0.35, median ms", "speakers with ≥ 8 clips", "their medians: p25 / median / p75 ms",
                      "speakers with no lead"], t))
    t = []
    for c in conds:
        d = [end_lead(r) for r in F[c] if end_lead(r) is not None]
        it = [r["ends"]["iterator"][-1] - r["ends"]["rvad"][-1] for r in F[c] if r["ends"]["iterator"] and not r["open"]["iterator"]]
        sp = by_speaker([r for r in F[c] if end_lead(r) is not None], lambda rs: q([end_lead(r) for r in rs], 50))
        t.append([NAMES[c], len(d), pct(sum(r["final_by_clock"] for r in F[c]), len(F[c])),
                  f"{ms(q(d, 50))} ({ms(q(d, 25))} – {ms(q(d, 75))})", pct(sum(x > 1e-6 for x in d), len(d)),
                  sum(x < -1e-6 for x in d), len(sp), spread(sp.values()), sum(v <= 1e-6 for v in sp.values()),
                  f"{ms(q([r['final_end'] - r['last'] for r in F[c]], 50))}", f"{ms(q(it, 50))}"])
    out.append(table(["Condition", "clips where both closed the same segments", "last `end` made by the quiet clock",
                      "`end` earlier than Silero alone: median (p25 – p75) ms", "earlier in", "later in (clips)",
                      "speakers with ≥ 8 clips", "their medians: p25 / median / p75 ms", "speakers with median 0",
                      "`end` after the labelled end of the last word, median ms",
                      "VADIterator's end after RealtimeVAD's `end`, median ms"], t))
    out.append("The `end` lead by how far the audio after the last word (50 to 300 ms after its labelled end) lies "
               "under the quiet floor (−54 dBFS, or 20 dB under the speaker when that is lower):\n")
    for name, group in (("LibriSpeech (both)", ("libri_clean", "libri_other")),
                        ("AMI headset (both)", ("ami_headset", "ami_headset_xtalk"))):
        G = [r for c in group if c in F for r in F[c] if margin(r) is not None]
        t = []
        for label, lo, hi in (("≥ 12 dB under", 12, 999), ("6 – 12 dB under", 6, 12), ("3 – 6 dB under", 3, 6),
                              ("0 – 3 dB under", 0, 3), ("0 – 3 dB over", -3, 0), ("3 – 6 dB over", -6, -3), ("> 6 dB over", -999, -6)):
            S = [r for r in G if lo <= margin(r) < hi]
            d = [end_lead(r) for r in S if end_lead(r) is not None]
            ld = [lead(r) for r in S if lead(r) is not None]
            t.append([label, len(S), pct(len(S), len(G)), pct(sum(r["final_by_clock"] for r in S), len(S)), ms(q(d, 50)),
                      ms(q(ld, 50)), f"{q([r.get('final_breaks', 0) for r in S], 50):.0f}" if S else "–"])
        out.append(f"{name}:\n")
        out.append(table(["Audio after the last word", "clips", "share", "`end` made by the quiet clock", "`end` lead, median ms",
                          "speech-off lead, median ms", "breaks of the quiet count, median"], t))

    G = [r for c in ("libri_clean", "libri_other") if c in F for r in F[c]]
    if G:
        t = []
        for label, lo, hi in (("under 0.0003 (−70 dBFS)", 0, 0.0003), ("0.0003 – 0.001", 0.0003, 0.001),
                              ("0.001 – 0.002", 0.001, 0.002), ("0.002 – 0.004 (quiet_rms is 0.002)", 0.002, 0.004),
                              ("over 0.004 (−48 dBFS)", 0.004, 9)):
            S = [r for r in G if lo <= r["dc"] < hi]
            d = [end_lead(r) for r in S if end_lead(r) is not None]
            ld = [lead(r) for r in S if lead(r) is not None]
            t.append([label, len(S), len({r["speaker"] for r in S}), f"{q([r['noise_db'] for r in S], 50):.0f}",
                      f"{q([r['noise_ac_db'] for r in S], 50):.0f}", pct(sum(r["final_by_clock"] for r in S), len(S)),
                      ms(q(ld, 50)), ms(q(d, 50))])
        out.append("LibriSpeech by the DC offset of the clip (mean sample value):\n")
        out.append(table(["DC offset", "clips", "speakers", "room level as fed, dBFS", "room level without the offset, dBFS",
                          "`end` made by the quiet clock", "speech-off lead, median ms", "`end` lead, median ms"], t))

    out.append("### `end` while the speaker goes on\n")
    t = []
    for c in conds:
        n = len(R[c])
        row = [NAMES[c], n]
        for det in ("rvad", "silero", "iterator"):
            row += [f"{sum(len(r['mid_ends'][det]) for r in R[c])} ({pct(sum(bool(r['mid_ends'][det]) for r in R[c]), n)})"]
        only = [e for r in R[c] for e in own_ends(r)]
        back = [e[3] for e in only if e[3] is not None]
        row += [len(only), ms(q(back, 50)), sum(x <= 0.1 for x in back)]
        clock = [e for r in R[c] for e in r["rvad_ends"] if e[2]]
        row += [f"{sum(e[6] >= 0.5 for e in clock)} of {len(clock)}"]
        for det in ("rvad", "silero", "iterator"):
            row.append(sum(r["lost_audible"][det] >= 0.1 for r in R[c]))
        t.append(row)
    out.append(table(["Condition", "clips", "mid-utterance ends (clips with one): RealtimeVAD", "Silero alone", "VADIterator",
                      "RealtimeVAD ends that Silero alone does not make", "… Silero hears speech again after, median ms",
                      "… within 100 ms", "quiet-clock ends during which Silero never fell below 0.5",
                      "clips with ≥ 100 ms of audible labelled speech in no segment: RealtimeVAD", "Silero alone", "VADIterator"], t))

    out.append("### `pause` false alarms and a confirmation delay\n")
    t = []
    for c in conds:
        sp_s = sum(r["last"] - r["first"] for r in F[c])
        false = [p for r in F[c] for p in r["pauses"] if p[0] in set(false_pauses(r))]
        soff = [p for r in F[c] for p in r["silero_offs"] if r["first"] < p[0] < (r.get("silero_off_0.5") or 0)]
        none = sum(1 for r in F[c] if not false_pauses(r))
        sp = by_speaker(F[c], lambda rs: sum(len(false_pauses(r)) for r in rs) / max(1e-9, sum(r["last"] - r["first"] for r in rs)))
        v = list(sp.values())
        t.append([NAMES[c], f"{len(false) / sp_s:.2f}", f"{len(soff) / sp_s:.2f}", pct(none, len(F[c])),
                  f"{q(v, 25):.2f} / {q(v, 50):.2f} / {q(v, 75):.2f} (max {max(v):.2f})" if v else "–",
                  pct(sum(p[1] == "q" for p in false), len(false)),
                  pct(sum(p[5][0] == "word" for p in false), len(false))])
    out.append(table(["Condition", "false `pause` per s of speech", "Silero p < 0.5 followed by speech, per s",
                      "clips without a false `pause`", "per speaker: p25 / median / p75", "from the quiet clock",
                      "inside a labelled word"], t))
    if any(r.get("resumes") for c in conds for r in F[c]):
        t = []
        for c in conds:
            sp_s = sum(r["last"] - r["first"] for r in F[c])
            false = [p[0] for r in F[c] for p in r["pauses"] if p[0] in set(false_pauses(r))]
            taken = [x for r in F[c] for x in r["resumes"] if x[1] in set(false_pauses(r))]
            last = sum(1 for r in F[c] if final_pause(r) is not None and any(x[1] == final_pause(r) for x in r["resumes"]))
            after = [x[0] - x[1] for x in taken]
            back = [x[2] for x in taken if x[2] is not None]
            t.append([NAMES[c], f"{len(taken) / sp_s:.2f}", pct(len(taken), len(false)),
                      f"{ms(q(after, 50))} ({ms(q(after, 25))} – {ms(q(after, 75))}, p90 {ms(q(after, 90))})" if after else "–",
                      f"{ms(q(back, 50))} ({ms(q(back, 25))} – {ms(q(back, 75))}, p90 {ms(q(back, 90))})" if back else "–",
                      pct(last, sum(1 for r in F[c] if final_pause(r) is not None))])
        out.append("The `resume` event (`resume_events=True`): the speech went on after a `pause`.\n")
        out.append(table(["Condition", "`resume` per s of speech", "false `pause`s taken back by a `resume`",
                          "`resume` after its `pause`: median (p25 – p75, p90) ms",
                          "after the first block that was not quiet again (quiet-clock pauses), ms",
                          "clips with a `resume` after the last `pause` (the speech-off)"], t))
    edges = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.24)
    t = []
    for c in conds:
        false = [p for r in F[c] for p in r["pauses"] if p[0] in set(false_pauses(r)) and p[1] == "q"]
        n = len(false)
        row, lo = [NAMES[c], n], 0.0
        for hi in edges:
            row.append(pct(sum(lo < p[2] <= hi + 1e-9 for p in false), n))
            lo = hi
        row.append(pct(sum(p[2] > lo + 1e-9 for p in false), n))
        t.append(row)
    out.append("How long the audio stayed quiet after a false quiet-clock `pause` (10 ms blocks until a louder one; "
               "the last column ran on into an `end`):\n")
    out.append(table(["Condition", "false pauses", "10 ms", "20", "30", "40", "50", "60", "70 – 80", "90 – 100", "110 – 150",
                      "160 – 240", "longer"], t))
    out.append("A `pause` that is reported only after the quiet has lasted this much longer (replayed, library unchanged):\n")
    head = ["Delay"]
    for c in conds:
        head += [f"{NAMES[c]}: false per s", "lead ms (earlier in)"]
    t = []
    for n in DELAYS:
        row = [f"{(n - 1) * 10} ms"]
        for c in conds:
            sp_s = sum(r["last"] - r["first"] for r in F[c])
            ps = {r["id"]: r["delayed"].get(str(n), r["delayed"].get(n)) for r in F[c]}
            false = sum(len(false_pauses(r, ps[r["id"]])) for r in F[c])
            ld = [x for x in (lead(r, "0.5", ps[r["id"]]) for r in F[c]) if x is not None]
            row += [f"{false / sp_s:.2f}", f"{ms(q(ld, 50))} ({pct(sum(x > 0 for x in ld), len(ld))})"]
        t.append(row)
    out.append(table(head, t))

    out.append("### Start\n")
    t = []
    for c in conds:
        n = len(R[c])
        row = [NAMES[c], n]
        for det in ("rvad", "iterator"):
            d = [r["start"][det] for r in R[c] if r["start"][det] is not None]
            row += [f"{ms(q(d, 50))} ({ms(q(d, 25))} – {ms(q(d, 75))}, p90 {ms(q(d, 90))})"]
        both = [r["start"]["rvad"] - r["start"]["iterator"] for r in R[c]
                if r["start"]["rvad"] is not None and r["start"]["iterator"] is not None]
        sp = by_speaker(R[c], lambda rs: q([r["start"]["rvad"] for r in rs if r["start"]["rvad"] is not None], 50))
        sp = {k: v for k, v in sp.items() if v == v}
        none = {det: sum(r["start"].get(det, 0) is None for r in R[c]) for det in ("rvad", "silero", "iterator", "norm")}
        row += [f"{ms(q(both, 50))} (p90 {ms(q(both, 90))})", spread(sp.values()),
                f"{none['rvad']} ({pct(none['rvad'], n)})", none["silero"], f"{none['iterator']} ({pct(none['iterator'], n)})",
                f"{none['norm']} ({pct(none['norm'], n)})" if "norm" in R[c][0]["start"] else "–"]
        t.append(row)
    out.append(table(["Condition", "clips", "`start` decided after the first word's labelled start: median (p25 – p75, p90) ms",
                      "Silero p ≥ 0.5 (VADIterator's start)", "RealtimeVAD after Silero, median ms",
                      "per-speaker medians: p25 / median / p75 ms", "no segment: RealtimeVAD", "Silero alone",
                      "Silero p ≥ 0.5", "RealtimeVAD with normalize_to_dbfs=-30"], t))

    out.append("### Speakers with the least lead\n")
    t = []
    for name, group in (("LibriSpeech", ("libri_clean", "libri_other")), ("AMI headset", ("ami_headset", "ami_headset_xtalk"))):
        G = [r for c in group if c in F for r in F[c] if lead(r) is not None]
        per = collections.defaultdict(list)
        for r in G:
            per[r["speaker"]].append(r)
        per = {s: rs for s, rs in per.items() if len(rs) >= 8}
        ranked = sorted(per, key=lambda s: (q([lead(r) for r in per[s]], 50), s))
        for s in ranked[:6] + ["all"]:
            rs = G if s == "all" else per[s]
            el = [end_lead(r) for r in rs if end_lead(r) is not None]
            t.append([name, f"all {len(per)} speakers" if s == "all" else f"{s} ({rs[0].get('gender') or '?'})", len(rs),
                      ms(q([lead(r) for r in rs], 50)), ms(q(el, 50)), pct(sum(r["final_by_clock"] for r in rs), len(rs)),
                      f"{q([r['speech_db'] for r in rs], 50):.0f}", f"{q([r['noise_db'] for r in rs], 50):.0f}",
                      f"{q([r['tail_db'] for r in rs if r['tail_db'] is not None], 50):.0f}",
                      f"{q([margin(r) for r in rs if margin(r) is not None], 50):.0f}"])
    out.append(table(["Corpus", "speaker", "clips", "speech-off lead, median ms", "`end` lead, median ms",
                      "`end` by the quiet clock", "speech dBFS", "room dBFS", "after the last word dBFS",
                      "dB under the quiet floor"], t))
    return "\n".join(out)


def examples(rows, n: int) -> str:
    """Example clips per failure pattern, worst first: python benchmarks/word_timing.py --clip <id>."""
    out = ["### Example clips\n"]

    def add(title, picked):
        out.append(f"{title}:\n")
        out.append("\n".join(f"- `{r['condition']}/{r['id']}` {note}" for r, note in picked[:n]) + "\n")

    libri = [r for r in rows if r["condition"].startswith("libri")]
    close = [r for r in rows if r["condition"] != "ami_far"]
    fin = [r for r in close if clean_final(r) and final_pause(r) is not None]
    s = sorted((r for r in fin if not r["final_by_clock"] and r.get("final_breaks", 0) >= 3 and (lead(r) or 0) > 0.05),
               key=lambda r: (-r["final_breaks"], r["id"]))
    add("The quiet count broken again and again by the room (speech-off early, `end` left to Silero)",
        [(r, f"{r['final_breaks']} breaks, room {r['noise_db']:.0f} dBFS, speech {r['speech_db']:.0f} dBFS") for r in s])
    s = sorted((r for r in fin if not r["final_by_clock"] and (margin(r) or 0) < 0
                and next(p[1] for p in r["pauses"] if p[0] == final_pause(r)) == "s"),
               key=lambda r: (r["tail_db"] - r["speech_db"], r["id"]))
    add("Nothing counts as quiet after the last word (no early `pause`, no early `end`); the quietest rooms first",
        [(r, f"after the word {r['tail_db']:.0f} dBFS, speech {r['speech_db']:.0f} dBFS, room {r['noise_db']:.0f} dBFS") for r in s])
    s = sorted((r for r in libri if clean_final(r) and final_pause(r) is not None and r.get("final_residual", 0) >= 0.05),
               key=lambda r: (final_pause(r) - r["last"], r["id"]))
    add("Speech-off inside a soft word ending (LibriSpeech)",
        [(r, f"`pause` {1000 * (r['last'] - final_pause(r)):.0f} ms before the labelled end, {1000 * r['final_residual']:.0f} ms "
             f"of audible sound after it") for r in s])
    s = sorted(((r, e) for r in libri for e in own_ends(r) if e[3] is not None), key=lambda x: (x[1][3], x[0]["id"]))
    add("`end` in a short pause that Silero alone rides through (LibriSpeech)",
        [(r, f"`end` at {e[0]:.2f} s, Silero hears speech again {1000 * e[3]:.0f} ms later") for r, e in s])
    s = sorted(((r, e) for r in close for e in r["rvad_ends"] if e[2] and e[6] >= 0.5), key=lambda x: (-x[1][6], x[0]["id"]))
    add("Quiet-clock `end` while Silero stayed above 0.5 throughout",
        [(r, f"`end` at {e[0]:.2f} s, Silero's lowest probability in the wait {e[6]:.2f}") for r, e in s])
    s = sorted((r for r in close if r["lost_audible"]["rvad"] - r["lost_audible"]["silero"] >= 0.1),
               key=lambda r: (r["lost_audible"]["silero"] - r["lost_audible"]["rvad"], r["id"]))
    add("Audible speech in no segment that Silero alone keeps",
        [(r, f"{1000 * r['lost_audible']['rvad']:.0f} ms against {1000 * r['lost_audible']['silero']:.0f} ms, "
             f"speech {r['speech_db']:.0f} dBFS, room {r['noise_db']:.0f} dBFS") for r in s])
    s = sorted((r for r in close if r["start"]["rvad"] is None and r["start"]["iterator"] is not None),
               key=lambda r: (-r["p_max"], r["id"]))
    add("No segment although Silero at 0.5 hears the utterance",
        [(r, f"{r['n_words']} words, {1000 * (r['last'] - r['first']):.0f} ms, highest probability {r['p_max']:.2f}, "
             f"speech {r['speech_db']:.0f} dBFS") for r in s])
    s = sorted((r for r in rows if r["p_differs"]), key=lambda r: (-r["p_differs"], r["id"]))
    add("Silero's probabilities differ with and without the quiet clock", [(r, f"{r['p_differs']} frames") for r in s])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only print the tables from the saved rows")
    ap.add_argument("--clip", help="show one clip event by event (id, or condition/id)")
    ap.add_argument("--processes", type=int, default=min(6, os.cpu_count() or 4), help="worker processes (default 6)")
    ap.add_argument("--limit", type=int, default=0, help="every n-th clip only")
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--options", default="{}", help='VAD arguments for the rvad detector as JSON, e.g. {"end_s": 0.3}')
    ap.add_argument("--examples", type=int, default=0, help="also list this many example clips per failure pattern")
    a = ap.parse_args()
    if not a.report or a.clip:
        if not (DATA / "clips.json").exists():
            raise SystemExit("no clips; run python tests/audio/fetch_word_timing.py first")
        clips = json.loads((DATA / "clips.json").read_text(encoding="utf-8"))
    if a.clip:
        for c in clips:
            if a.clip in (c["id"], f"{c['condition']}/{c['id']}"):
                show(c)
        return 0
    if a.report:
        rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    else:
        if a.limit:
            clips = clips[:: a.limit]
        t0 = time.time()
        low_priority()
        with Pool(a.processes, initializer=_options, initargs=(json.loads(a.options),)) as pool:
            rows = pool.map(measure, clips, chunksize=8)
        Path(a.rows).write_text(json.dumps(rows) + "\n", encoding="utf-8")
        print(f"{len(rows)} clips in {time.time() - t0:.0f} s with {a.processes} processes", file=sys.stderr)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report(rows))
    if a.examples:
        print(examples(rows, a.examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
