"""Why is the last `pause` before an `end` sometimes early, and what would tell it apart?

On the clips of benchmarks/word_timing.py (close microphones only), for every turn whose
last `pause` (the speech-off) comes from the quiet clock:

  early       the pause is more than 100 ms before the last word's labelled end and at least
              30 ms of audible sound of that word are still to come
  true stop   the pause is at most 50 ms before the labelled end, or after it

1. Mechanism: what the quiet clock saw between an early pause and the word's end.
2. Features, all causal, of every 10 ms block from the pause on, and what a gate on them
   would do: the pause is reported at the first quiet block at which the gate is open (at
   the latest when Silero's own first unvoiced frame reports it). Thresholds are chosen on
   half of the speakers and reported on the other half.
3. Two rule candidates, replayed with a model of the library's pause and end logic on the
   same blocks and frames (the model is checked against the library's own events):
     a  "quiet" = within X dB of the learned room level, where the room is at least Y dB
        under the speaker (elsewhere the library's own decision)
     b  after N blocks in a row at least Z dB above the room, a second pause is reported
        at the next quiet block that is not
     c  (the smallest change) a second pause is reported whenever the quiet count starts
        again after N blocks in a row over the quiet threshold; today that takes 64 ms
     d  as c, but only blocks that are also Z dB above the room count (room noise that
        merely crosses the threshold does not ask for a second pause)

The library is not changed. Its per-block decisions are read by wrapping the instance's
`_block_start` (level seen, threshold, verdict) and reading `_noise_db` / `_level_db`; this
works for 0.0.5 and for the DC high-pass / noise-subtraction branch alike.

    python benchmarks/early_pause.py --tag 0.0.5              # trace + replay (PYTHONPATH picks the library)
    python benchmarks/early_pause.py --report 0.0.5 [other]   # tables from saved rows

Rows go to benchmarks/early_pause_rows_<tag>.pkl (git-ignored).
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pickle
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import word_timing as wt  # noqa: E402

SR, FRAME, BLOCK = wt.SR, wt.FRAME, wt.BLOCK
FRAME_S, BLOCK_S = wt.FRAME_S, wt.BLOCK_S
HERE = Path(__file__).resolve().parent
W = 64  # blocks kept from the pause on
GROUPS = (("LibriSpeech clean", ("libri_clean",)), ("LibriSpeech other", ("libri_other",)),
          ("AMI headset", ("ami_headset", "ami_headset_xtalk")))
FEATURES = ("above_room", "under_speaker", "under_threshold", "slope_30ms", "slope_50ms", "silero_p",
            "silero_dp", "high_band", "since_confident", "since_loud", "quiet_run")
NICE = {"above_room": "level above the learned room, dB", "under_speaker": "level under the speaker's level, dB",
        "under_threshold": "level under the quiet threshold, dB", "slope_30ms": "level change over the last 30 ms, dB",
        "slope_50ms": "level change over the last 50 ms, dB", "silero_p": "Silero's probability (latest frame)",
        "silero_dp": "its change over the last 3 frames", "high_band": "share of the energy above 3 kHz (20 ms)",
        "since_confident": "ms since Silero's last frame at 0.9 or more",
        "since_loud": "ms since the level was last within 10 dB of the speaker",
        "quiet_run": "quiet blocks in a row so far (a plain confirmation delay)"}
RULES_A = [(x, y) for y in (20, 25, 30) for x in (6, 9, 12)]
RULES_B = [(z, n) for z in (6, 10, 15) for n in (1, 2, 3)]
RULES_C = [1, 2, 3]
RULES_D = [(z, n) for z in (6, 10) for n in (1, 2, 3)]
ROOM_MIN_DB = -75.0  # rule a never asks for less than this (gated digital silence has no room)


# ---------------------------------------------------------------------- trace


def trace(x: np.ndarray, **options):
    """One pass of the library with its per-block decisions recorded."""
    from RealtimeVAD import VAD

    vad = VAD(frame_events=True, **options)
    rec = {k: [] for k in ("db", "thr", "quiet", "raised", "noise", "level", "armed")}
    inner = vad._block_start

    def spy(arg):
        power = float(np.mean(arg * arg)) if isinstance(arg, np.ndarray) else float(arg)
        thr = vad._quiet
        quiet = inner(arg)
        rec["db"].append(10 * np.log10(max(power, 1e-10)))
        rec["thr"].append(20 * np.log10(max(thr, 1e-9)))
        rec["quiet"].append(bool(quiet))
        rec["raised"].append(thr > vad.quiet_rms * (1 + 1e-6) or bool(getattr(vad, "_by_subtraction", False)))
        rec["noise"].append(np.nan if vad._noise_db is None else vad._noise_db)
        rec["level"].append(np.nan if vad._level_db is None else vad._level_db)
        rec["armed"].append(vad._t_quiet is not None)
        return quiet

    vad._block_start = spy
    frames, starts, pauses, ends = [], [], [], []
    n = 0
    while n < len(x):
        step = min(BLOCK - n % BLOCK, FRAME - n % FRAME, len(x) - n)
        begun = None
        for e in vad.feed(x[n:n + step]):
            if e.kind == "frame":
                frames.append((e.time, e.frame.probability))
                if begun is not None:
                    starts.append(e.time)
                    begun = None
            elif e.kind == "start":
                begun = e.time
            elif e.kind == "pause":
                pauses.append((e.time, e.source))
            elif e.kind == "end":
                ends.append((e.time, e.segment.speech_end, starts[-1] if starts else 0.0))
        n += step
    out = {k: np.asarray(v, dtype=bool if k in ("quiet", "raised", "armed") else np.float32) for k, v in rec.items()}
    out.update(frames=frames, starts=starts, pauses=pauses, ends=ends)
    return out


def replay(quiet, raised, p, frame_quiet, forced, above=None, need=0):
    """The library's pause and end logic on per-block verdicts and per-frame probabilities.

    `forced`: frames at which the library itself started a segment (soft starts are not
    modelled). With `above` (blocks clearly above the room) and `need`, rule b: after `need`
    such blocks in a row a paused segment may pause again, at a quiet block that is not."""
    pauses, ends = [], []
    K, J = len(quiet), len(p)
    kb = jf = 0
    speaking = tail = armed = unbroken = resumed = paused = strict = False
    streak = silent = loud = run = 0
    t_quiet = wait = last_voiced = begun = 0.0
    while True:
        nb, nf = (kb + 1) * BLOCK, (jf + 1) * FRAME
        if kb >= K and jf >= J:
            break
        n = min(nb if kb < K else 1 << 60, nf if jf < J else 1 << 60)
        is_block, is_frame = kb < K and nb == n, jf < J and nf == n
        t = n / SR
        q = bool(quiet[kb]) if is_block else False
        if is_block:
            if armed:
                if q:
                    loud = 0
                else:
                    unbroken = False
                    loud += BLOCK
                    if loud >= 1024:
                        armed, loud, paused, resumed, strict = False, 0, False, False, False
            if above is not None and speaking:
                run = run + 1 if above[kb] else 0
                if paused and run >= need:
                    paused, strict = False, True
        if is_frame:
            pj = p[jf]
            if not speaking:
                voiced = pj >= (0.45 if streak else 0.9)
                if tail:
                    tail = pj >= 0.45
                    if frame_quiet[jf]:
                        voiced = False
                streak = streak + 1 if voiced else 0
                if streak >= 2 or jf in forced:
                    speaking, begun, last_voiced = True, t, t
                    streak = silent = loud = run = 0
                    armed = unbroken = resumed = paused = strict = False
            else:
                if pj >= 0.45:
                    if silent:
                        paused = strict = False
                        if armed and t > t_quiet:
                            unbroken, resumed = False, True
                    silent, last_voiced = 0, t
                else:
                    silent += 1
                    if silent == 1 and not paused:
                        pauses.append((t, "silero"))
                        paused = True
                if silent >= 8:
                    ends.append((t, last_voiced, begun))
                    speaking, tail, streak = False, False, 0
            jf += 1
        if is_block:
            if speaking:
                if armed and unbroken and t - t_quiet + 1e-9 >= wait:
                    ends.append((t, t_quiet, begun))
                    speaking, tail, streak, armed = False, True, 0, False
                elif q and (not armed or not (unbroken or resumed)):
                    armed, t_quiet, unbroken, loud = True, t, True, 0
                    wait = 0.24 + (0.04 if raised[kb] else 0.0)
                    if not paused and not (strict and above[kb]):
                        pauses.append((t, "quiet"))
                        paused, strict = True, False
                elif strict and not paused and q and not above[kb]:
                    pauses.append((t, "quiet"))
                    paused, strict = True, False
            kb += 1
    return pauses, ends


# -------------------------------------------------------------------- measure


def turn(pauses, ends, frames, words, db, soft):
    """The speech-off of one run of events: (clean, pause time, seconds relative to the
    labelled end, audible sound to come, lead over Silero p < 0.5, false pauses)."""
    if not ends:
        return None
    end_t, _, begun = ends[-1]
    clean = words.last < end_t <= words.last + 1.0
    ts = [t for t, _ in pauses if begun - 0.4 <= t <= end_t]
    if not clean or not ts:
        return (clean, None, None, None, None, 0)
    fp = ts[-1]
    off = wt.first_of_final_low_run(frames, 0.5, end_t + 1.0)
    return (True, fp, fp - words.last, wt.audible_s(db, fp, words.last, soft) if fp < words.last else 0.0,
            None if off is None else off - fp, sum(1 for t, _ in pauses if words.first < t < fp))


def high_band(x: np.ndarray, K: int) -> np.ndarray:
    """Share of the energy above 3 kHz in the 20 ms ending with each block (100 Hz and up)."""
    pad = np.concatenate((np.zeros(BLOCK, np.float32), x[: K * BLOCK]))
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * BLOCK)[::BLOCK][:K]
    spec = np.abs(np.fft.rfft(win * np.hanning(2 * BLOCK), axis=1)) ** 2  # 50 Hz bins
    return (spec[:, 60:].sum(axis=1) / (spec[:, 2:].sum(axis=1) + 1e-12)).astype(np.float32)


def measure(clip):
    x = wt.load(wt.DATA / clip["path"])
    words = wt.Words(clip["words"])
    raw_db, ac = wt.acoustics(x, words)
    soft = max(ac["noise_db"] + 6.0, ac["speech_db"] - 25.0)
    a = trace(x, **wt.OPTIONS)
    K, frames = len(a["db"]), a["frames"]
    p = np.array([f[1] for f in frames], np.float32)
    J = len(p)
    row = {"id": clip["id"], "condition": clip["condition"], "speaker": clip["speaker"],
           "last_word": clip["words"][-1][0].lower(), "speech_s": words.last - words.first,
           "speech_db": ac["speech_db"], "noise_db": ac["noise_db"]}
    lib = turn(a["pauses"], a["ends"], frames, words, raw_db, soft)
    row["lib"] = lib

    # replay: the model with the library's own verdicts, then the rule candidates
    fill = lambda v: np.where(np.isnan(v), np.nanmin(v) if np.isfinite(v).any() else -100.0, v)  # noqa: E731
    noise = fill(a["noise"])
    have = ~np.isnan(a["noise"]) & ~np.isnan(a["level"])
    level = fill(a["level"])
    frame_end_block = np.minimum(K - 1, ((np.arange(J) + 1) * FRAME) // BLOCK - 1)
    frame_quiet = a["quiet"][frame_end_block] if K else np.zeros(J, bool)
    forced = {int(round(t * SR / FRAME)) - 1 for t in a["starts"]}
    base_p, base_e = replay(a["quiet"], a["raised"], p, frame_quiet, forced)
    row["replay_same"] = ([round(t, 4) for t, _ in base_p] == [round(t, 4) for t, _ in a["pauses"]]
                          and [round(e[0], 4) for e in base_e] == [round(e[0], 4) for e in a["ends"]])
    sims = {"model": turn(base_p, base_e, frames, words, raw_db, soft)}
    for X, Y in RULES_A:
        near = a["db"] <= np.maximum(noise + X, ROOM_MIN_DB)
        quiet = np.where(have & (level - noise >= Y), near, a["quiet"])
        ps, es = replay(quiet, a["raised"], p, quiet[frame_end_block] if K else frame_quiet, forced)
        sims[f"a {X}/{Y}"] = turn(ps, es, frames, words, raw_db, soft)
    for Z, N in RULES_B:
        above = have & (a["db"] >= noise + Z)
        ps, es = replay(a["quiet"], a["raised"], p, frame_quiet, forced, above, N)
        sims[f"b {Z}/{N}"] = turn(ps, es, frames, words, raw_db, soft)
    for N in RULES_C:
        ps, es = replay(a["quiet"], a["raised"], p, frame_quiet, forced, ~a["quiet"], N)
        sims[f"c {N}"] = turn(ps, es, frames, words, raw_db, soft)
    for Z, N in RULES_D:
        ps, es = replay(a["quiet"], a["raised"], p, frame_quiet, forced, ~a["quiet"] & have & (a["db"] >= noise + Z), N)
        sims[f"d {Z}/{N}"] = turn(ps, es, frames, words, raw_db, soft)
    row["sims"] = sims

    if not lib or not lib[0] or lib[1] is None:
        return row
    fp, rel, residual, lead = lib[1], lib[2], lib[3], lib[4]
    source = next(s for t, s in a["pauses"] if t == fp)
    row["source"] = source
    row["class"] = 1 if rel < -0.1 and residual >= 0.03 else 0 if rel >= -0.05 else -1
    kp = int(round(fp / BLOCK_S)) - 1  # the block the pause was reported on
    kt = min(K - 1, int(np.ceil(words.last / BLOCK_S)) - 1)  # the block the labelled end lies in
    end_t = a["ends"][-1][0]

    # 1. mechanism: the blocks after the pause up to the word's labelled end
    if source == "quiet" and kt > kp:
        s = slice(kp + 1, kt + 1)
        quiet = a["quiet"][s]
        runs = [len(list(g)) for v, g in itertools.groupby(quiet) if not v]
        heard = raw_db[s][: len(quiet)] >= soft
        m = {"blocks": int(len(quiet)), "loud": int((~quiet).sum()), "longest_loud": max(runs, default=0),
             "disarmed": bool((a["armed"][kp:kt] & ~a["armed"][kp + 1:kt + 1]).any()), "audible": int(heard.sum())}
        if heard.any():
            d = a["db"][s][heard]
            m.update(above_room=float(np.median(d - noise[s][heard])), under_speaker=float(np.median(level[s][heard] - d)),
                     under_threshold=float(np.median(a["thr"][s][heard] - d)),
                     above_room_clip=float(np.median(raw_db[s][: len(quiet)][heard] - ac["noise_db"])),
                     under_speech_clip=float(np.median(ac["speech_db"] - raw_db[s][: len(quiet)][heard])),
                     high_band=float(np.median(high_band(x, K)[s][heard])))
        row["mechanism"] = m

    # 2. causal features of every block from the pause on
    blk = np.arange(K)
    latest = ((blk + 1) * BLOCK) // FRAME - 1  # latest finished frame
    pb = np.where(latest >= 0, p[np.clip(latest, 0, max(J - 1, 0))], 0.0) if J else np.zeros(K)
    p3 = np.where(latest >= 3, p[np.clip(latest - 3, 0, max(J - 1, 0))], pb) if J else np.zeros(K)
    conf = np.where(p >= 0.9)[0]
    last_conf = np.full(K, -1e4)
    if len(conf):
        i = np.searchsorted(conf, latest, side="right") - 1
        last_conf = np.where(i >= 0, (conf[np.clip(i, 0, None)] + 1) * FRAME / SR, -1e4)
    loud = a["db"] >= level - 10.0
    last_loud = np.maximum.accumulate(np.where(loud, blk, -10000))
    qrun = np.zeros(K, np.float32)
    c = 0
    for i in range(max(0, kp - 30), K):
        c = c + 1 if a["quiet"][i] else 0
        qrun[i] = c
    lag = lambda n: a["db"] - a["db"][np.clip(blk - n, 0, None)]  # noqa: E731
    feats = np.stack([a["db"] - noise, level - a["db"], a["thr"] - a["db"], lag(3), lag(5), pb, pb - p3, high_band(x, K),
                      np.minimum(2000.0, ((blk + 1) * BLOCK_S - last_conf) * 1000), np.minimum(2000.0, (blk - last_loud) * 10.0),
                      qrun], axis=1).astype(np.float32)
    win = np.full((W, len(FEATURES)), np.nan, np.float32)
    quiet = np.zeros(W, bool)
    n = max(0, min(W, K - kp))
    win[:n], quiet[:n] = feats[kp:kp + n], a["quiet"][kp:kp + n]
    # the gate's latest moment: Silero's own first unvoiced frame after the pause, or the end
    unvoiced = next((t for t, q in frames if t > fp and q < 0.45), end_t)
    cap = 0 if source != "quiet" else max(0, min(W - 1, int(np.ceil(min(unvoiced, end_t) / BLOCK_S)) - 1 - kp))
    bound = 0  # a pause at window block k is still "early" for k < bound
    while bound < W and words.last - (kp + bound + 1) * BLOCK_S > 0.1 and \
            wt.audible_s(raw_db, (kp + bound + 1) * BLOCK_S, words.last, soft) >= 0.03:
        bound += 1
    row.update(win=win, quiet=quiet, cap=cap, bound=bound, lead=lead, rel=rel, residual=residual)
    return row


# --------------------------------------------------------------------- report


def q(v, p):
    v = [x for x in v if x is not None and x == x]
    return float(np.percentile(v, p)) if v else float("nan")


def num(v, digits=0):
    return "–" if v != v else f"{v:.{digits}f}"


def pct(n, d):
    return "–" if not d else f"{100 * n / d:.1f} %"


def held_out(speaker: str) -> bool:
    return zlib.crc32(speaker.encode()) % 2 == 1


def turn_stats(turns, speech_s):
    """Row cells for a list of `turn` results: early shares, lead, false pauses."""
    ok = [t for t in turns if t and t[0] and t[1] is not None]
    n = len(ok)
    lead = [t[4] for t in ok if t[4] is not None]
    return [n, pct(sum(t[2] < -0.05 for t in ok), n), pct(sum(t[2] < -0.1 for t in ok), n),
            pct(sum(t[2] < -0.1 and t[3] >= 0.03 for t in ok), n), pct(sum(t[2] < -0.2 for t in ok), n),
            num(q(lead, 50) * 1000), pct(sum(x > 0 for x in lead), len(lead)),
            f"{sum(t[5] for t in ok) / max(1e-9, speech_s):.2f}"]


def ending(word: str) -> str:
    w = word.strip("'").lower()
    for name, tails in (("fricative (s, z, f, th, sh, ch, x, ce, se, ge)", ("s", "z", "f", "th", "sh", "ch", "x", "ce", "se", "ge", "ve", "fe")),
                        ("stop (t, d, k, p, g, b, ed, te, de, ke)", ("t", "d", "k", "p", "g", "b", "te", "de", "ke", "pe")),
                        ("nasal or liquid (m, n, ng, l, r)", ("m", "n", "ng", "l", "r", "ne", "me", "le", "re"))):
        if w.endswith(tails):
            return name
    return "vowel or other"


def gate_apply(F, Q, cap, mask):
    """Block at which a gated pause is reported, per case."""
    m = Q & mask
    first = np.where(m.any(axis=1), m.argmax(axis=1), W)
    return np.minimum(first, cap)


def gate_metrics(k, C, sel):
    early, true = sel & (C["cls"] == 1), sel & (C["cls"] == 0)
    lead = (C["lead"] - k * BLOCK_S)[sel & ~np.isnan(C["lead"])]
    delayed = k[true] > 0
    return {"removed": float((k[early] >= C["bound"][early]).mean()) if early.any() else float("nan"),
            "delayed": float(delayed.mean()) if true.any() else float("nan"),
            "delay": float(np.median(k[true][delayed]) * 10) if delayed.any() else 0.0,
            "delay_all": float(k[true].mean() * 10) if true.any() else float("nan"),
            "lead": float(np.median(lead) * 1000) if len(lead) else float("nan"),
            "earlier": float((lead > 0).mean()) if len(lead) else float("nan")}


def report(tag, rows) -> str:
    out = [f"## Library: {tag}\n"]
    table = wt.table
    G = {name: [r for r in rows if r["condition"] in conds] for name, conds in GROUPS}
    same = sum(r["replay_same"] for r in rows)
    out.append(f"{len(rows)} clips. The replay model with the library's own block verdicts gives the library's "
               f"`pause` and `end` events exactly in {same} clips ({pct(same, len(rows))}).\n")

    # ---- the cases
    out.append("### How many\n")
    t = []
    for name, R in G.items():
        ok = [r for r in R if "class" in r]
        n = len(ok)
        early = [r for r in ok if r["class"] == 1]
        t.append([name, n, f"{len(early)} ({pct(len(early), n)})", pct(sum(r["source"] == "quiet" for r in early), len(early)),
                  f"{sum(r['class'] == 0 for r in ok)} ({pct(sum(r['class'] == 0 for r in ok), n)})",
                  sum(r["class"] == -1 for r in ok), num(q([-r["rel"] * 1000 for r in early], 50)),
                  num(q([r["residual"] * 1000 for r in early], 50))])
    out.append(table(["Condition", "turns with a speech-off", "early", "… from the quiet clock", "true stops", "in between",
                      "early: ms before the labelled end, median", "audible sound to come, median ms"], t))

    # ---- 1. mechanism
    out.append("### 1. What the quiet clock saw between an early `pause` and the end of the word\n")
    t = []
    for name, R in G.items():
        E = [r["mechanism"] for r in R if r.get("class") == 1 and "mechanism" in r]
        n = len(E)
        A = [m for m in E if m["loud"] == 0]
        B = [m for m in E if m["loud"] > 0]
        t.append([name, n, pct(len(A), n), pct(len(B), n), pct(sum(m["longest_loud"] <= 2 for m in B), len(B)),
                  pct(sum(3 <= m["longest_loud"] <= 6 for m in B), len(B)), pct(sum(m["longest_loud"] >= 7 for m in B), len(B)),
                  num(q([m["loud"] * 10 for m in B], 50)), pct(sum(m["disarmed"] for m in B), len(B))])
    out.append(table(["Condition", "early pauses (quiet clock)", "A: every block up to the word's end counted as quiet",
                      "B: some blocks were not quiet, yet no new `pause`", "B: longest loud stretch ≤ 20 ms", "30 – 60 ms",
                      "≥ 70 ms", "B: loud blocks in all, median ms", "B: clock re-armed on the way"], t))
    t = []
    for name, R in G.items():
        for label, pick in (("A", lambda m: m["loud"] == 0), ("B", lambda m: m["loud"] > 0)):
            E = [r["mechanism"] for r in R if r.get("class") == 1 and "mechanism" in r and pick(r["mechanism"])
                 and "above_room" in r["mechanism"]]
            cell = lambda k: f"{num(q([m[k] for m in E], 50))} ({num(q([m[k] for m in E], 25))} – {num(q([m[k] for m in E], 75))})"  # noqa: E731
            t.append([name, label, len(E), num(q([m["blocks"] * 10 for m in E], 50)), num(q([m["audible"] * 10 for m in E], 50)),
                      cell("above_room"), cell("under_speaker"), cell("under_threshold"), cell("above_room_clip"),
                      cell("under_speech_clip"), num(q([m["high_band"] for m in E], 50), 2)])
    out.append("The audible part of the sound that was still to come (median per turn; then median and quartiles over turns):\n")
    out.append(table(["Condition", "", "turns", "pause to labelled end, ms", "audible, ms", "above the learned room, dB",
                      "under the learned speaker level, dB", "under the quiet threshold, dB (negative: over it)",
                      "above the clip's room (10th percentile), dB", "under the clip's speech level, dB",
                      "share above 3 kHz"], t))

    # ---- 2. features
    C = {k: [] for k in ("F", "Q", "cap", "bound", "cls", "lead", "group", "test")}
    for gi, (name, R) in enumerate(G.items()):
        for r in R:
            if "class" not in r:
                continue
            C["F"].append(r["win"]); C["Q"].append(r["quiet"]); C["cap"].append(r["cap"]); C["bound"].append(r["bound"])  # noqa: E702
            C["cls"].append(r["class"] if r["source"] == "quiet" else -2)
            C["lead"].append(np.nan if r["lead"] is None else r["lead"]); C["group"].append(gi)  # noqa: E702
            C["test"].append(held_out(r["speaker"]))
    C = {k: np.asarray(v) for k, v in C.items()}
    F, Q, cap = C["F"], C["Q"], C["cap"]
    train, test = ~C["test"], C["test"]
    out.append("### 2. Features at the moment the last `pause` fires\n")
    t = []
    auc = {}
    for i, f in enumerate(FEATURES[:-1]):
        e, s = F[C["cls"] == 1, 0, i], F[C["cls"] == 0, 0, i]
        e, s = e[~np.isnan(e)], s[~np.isnan(s)]
        ranks = np.argsort(np.argsort(np.concatenate((e, s)))) + 1
        auc[f] = (ranks[: len(e)].sum() - len(e) * (len(e) + 1) / 2) / (len(e) * len(s))
        d = 2 if f in ("silero_p", "silero_dp", "high_band") else 0
        t.append([NICE[f], f"{num(q(e, 50), d)} ({num(q(e, 25), d)} – {num(q(e, 75), d)})",
                  f"{num(q(s, 50), d)} ({num(q(s, 25), d)} – {num(q(s, 75), d)})", f"{max(auc[f], 1 - auc[f]):.2f}"])
    out.append(table(["Feature at the pause's block", "early: median (p25 – p75)", "true stop: median (p25 – p75)",
                      "separation (AUC, 0.5 = none)"], t))

    def sweep(masks_of):
        """Best operating point per removal target: chosen on the training speakers by the lead left."""
        best = {}
        for label, mask in masks_of:
            k = gate_apply(F, Q, cap, mask)
            m = gate_metrics(k, C, train)
            for target in (0.5, 0.75, 0.9):
                if m["removed"] >= target and (target not in best or m["lead"] > best[target][2]["lead"]):
                    best[target] = (label, k, m)
        return best

    grid = {}
    for i, f in enumerate(FEATURES):
        v = F[:, :, i][~np.isnan(F[:, :, i])]
        grid[f] = np.unique(np.percentile(v, np.concatenate(([0, 0.2, 0.5], np.linspace(1, 99, 50), [99.5, 99.8, 100]))))
    singles, direction = {}, {}
    for i, f in enumerate(FEATURES):
        cands = [((f, s, th), (F[:, :, i] <= th) if s == "≤" else (F[:, :, i] >= th)) for s in ("≤", "≥") for th in grid[f]]
        singles[f] = sweep(cands)
        if 0.75 in singles[f]:
            direction[f] = singles[f][0.75][0][1]
    base = gate_metrics(np.zeros(len(cap), int), C, test)
    pairs = {}
    names = [f for f in FEATURES if f in direction]
    cands = []
    for fa, fb in itertools.combinations(names, 2):
        ia, ib = FEATURES.index(fa), FEATURES.index(fb)
        for ta in grid[fa][::4]:
            ma = (F[:, :, ia] <= ta) if direction[fa] == "≤" else (F[:, :, ia] >= ta)
            for tb in grid[fb][::4]:
                mb = (F[:, :, ib] <= tb) if direction[fb] == "≤" else (F[:, :, ib] >= tb)
                cands.append((((fa, direction[fa], ta), (fb, direction[fb], tb)), ma & mb))
    pairs = sweep(cands)
    top = gate_metrics(cap.astype(int), C, test)
    out.append(f"A gate that never opens leaves the speech-off to Silero's first unvoiced frame. That is the ceiling: it "
               f"removes {pct(top['removed'], 1)} of the early pauses (in the rest Silero itself stops before the labelled end) "
               f"and leaves a lead of {num(top['lead'])} ms.\n")
    for target in (0.5, 0.75, 0.9):
        t = []
        if not any(target in singles[f] for f in FEATURES) and target not in pairs:
            out.append(f"No gate reaches {int(target * 100)} % on the speakers it was chosen on.\n")
            continue
        ranked = sorted((f for f in FEATURES if target in singles[f]),
                        key=lambda f: -gate_metrics(singles[f][target][1], C, test)["lead"])
        entries = [(NICE[f] + f" {singles[f][target][0][1]} {singles[f][target][0][2]:.3g}", singles[f][target][1]) for f in ranked]
        if target in pairs:
            (fa, sa, ta), (fb, sb, tb) = pairs[target][0]
            entries.insert(0, (f"best pair: {NICE[fa]} {sa} {ta:.3g} and {NICE[fb]} {sb} {tb:.3g}", pairs[target][1]))
        for label, k in entries:
            m = gate_metrics(k, C, test)
            cells = [label, pct(m["removed"], 1), pct(m["delayed"], 1), num(m["delay"]), num(m["delay_all"]),
                     num(m["lead"]), pct(m["earlier"], 1)]
            for gi in range(len(GROUPS)):
                mg = gate_metrics(k, C, test & (C["group"] == gi))
                cells.append(f"{pct(mg['removed'], 1)} / {num(mg['lead'])}")
            t.append(cells)
        out.append(f"Gates chosen to remove {int(target * 100)} % of the early pauses on half of the speakers; "
                   f"numbers from the other half (ungated there: lead {num(base['lead'])} ms, earlier than Silero in "
                   f"{pct(base['earlier'], 1)}). Features that cannot reach the target are missing:\n")
        out.append(table(["`pause` is reported at the first quiet block where", "early pauses removed", "true stops delayed",
                          "… by, median ms", "mean delay over all true stops, ms", "speech-off lead left, median ms",
                          "earlier than Silero in"] + [f"{g[0]}: removed / lead ms" for g in GROUPS], t))

    # ---- 3. rule candidates
    out.append("### 3. Rule candidates, replayed\n")
    for name, R in G.items():
        speech = sum(r["speech_s"] for r in R if r["lib"] and r["lib"][0] and r["lib"][1] is not None)
        t = [["library (measured)"] + turn_stats([r["lib"] for r in R], speech)]
        for key, label in ([("model", "replay model, no rule")]
                           + [(f"a {x}/{y}", f"a: quiet = within {x} dB of the room where it is ≥ {y} dB under the speaker") for x, y in RULES_A]
                           + [(f"b {z}/{n}", f"b: second pause after {n * 10} ms at ≥ {z} dB above the room") for z, n in RULES_B]
                           + [(f"c {n}", f"c: second pause when the count starts again after ≥ {n * 10} ms over the threshold") for n in RULES_C]
                           + [(f"d {z}/{n}", f"d: as c, counting only blocks that are also ≥ {z} dB above the room, ≥ {n * 10} ms") for z, n in RULES_D]):
            turns = [r["sims"][key] for r in R]
            sp = sum(r["speech_s"] for r, tn in zip(R, turns) if tn and tn[0] and tn[1] is not None)
            t.append([label] + turn_stats(turns, sp))
        out.append(f"{name}:\n")
        out.append(table(["Rule", "turns", "speech-off > 50 ms before the labelled end", "> 100 ms", "> 100 ms with sound to come",
                          "> 200 ms", "speech-off lead, median ms", "earlier than Silero in", "false `pause` per s of speech"], t))

    # ---- 4. speakers and word endings
    out.append("### 4. A few speakers or general\n")
    t = []
    for name, R in G.items():
        per = collections.defaultdict(list)
        for r in R:
            if "class" in r:
                per[r["speaker"]].append(r["class"] == 1)
        big = {s: v for s, v in per.items() if len(v) >= 8}
        rates = sorted((sum(v) / len(v) for v in big.values()), reverse=True)
        total = sum(sum(v) for v in per.values())
        top = sorted((sum(v) for v in per.values()), reverse=True)
        tenth = max(1, len(per) // 10)
        t.append([name, len(per), pct(sum(any(v) for v in per.values()), len(per)), len(big),
                  f"{num(q(rates, 25) * 100, 1)} / {num(q(rates, 50) * 100, 1)} / {num(q(rates, 75) * 100, 1)} (max {num(max(rates, default=float('nan')) * 100, 1)})",
                  pct(sum(top[:tenth]), total)])
    out.append(table(["Condition", "speakers", "with at least one early pause", "speakers with ≥ 8 turns",
                      "their early rate, %: p25 / median / p75", "share of all early pauses from the top tenth of speakers"], t))
    t = []
    for kind in sorted({ending(r["last_word"]) for r in rows}):
        cells = [kind]
        for name, R in G.items():
            ok = [r for r in R if "class" in r and ending(r["last_word"]) == kind]
            cells.append(f"{pct(sum(r['class'] == 1 for r in ok), len(ok))} of {len(ok)}")
        t.append(cells)
    out.append("Early rate by how the last word is spelled at its end:\n")
    out.append(table(["Last word ends in"] + [g[0] for g in GROUPS], t))

    # ---- examples
    out.append("### Example clips\n")
    E = [r for r in rows if r.get("class") == 1 and "mechanism" in r]
    for title, pick in (("A: the rest of the word stayed under the quiet threshold", lambda r: r["mechanism"]["loud"] == 0),
                        ("B: the rest of the word crossed the threshold, no new `pause`", lambda r: r["mechanism"]["loud"] > 0)):
        for name, conds in (("LibriSpeech", ("libri_clean", "libri_other")), ("AMI headset", ("ami_headset", "ami_headset_xtalk"))):
            s = sorted((r for r in E if pick(r) and r["condition"] in conds), key=lambda r: (-r["residual"], r["id"]))[:10]
            out.append(f"{title} ({name}):\n")
            out.append("\n".join(
                f"- `{r['condition']}/{r['id']}` \"{r['last_word']}\": `pause` {-1000 * r['rel']:.0f} ms before the labelled end, "
                f"{1000 * r['residual']:.0f} ms audible after it, {r['mechanism'].get('above_room', float('nan')):.0f} dB above the room, "
                f"{r['mechanism'].get('under_speaker', float('nan')):.0f} dB under the speaker" for r in s) + "\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="trace the library on PYTHONPATH and save rows under this name")
    ap.add_argument("--report", nargs="+", help="print the tables for these saved tags")
    ap.add_argument("--processes", type=int, default=min(6, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="every n-th clip only")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.tag:
        clips = [c for c in json.loads((wt.DATA / "clips.json").read_text(encoding="utf-8")) if c["condition"] != "ami_far"]
        if a.limit:
            clips = clips[:: a.limit]
        import RealtimeVAD

        t0 = time.time()
        wt.low_priority()
        with Pool(a.processes, initializer=wt.low_priority) as pool:
            rows = pool.map(measure, clips, chunksize=4)
        (HERE / f"early_pause_rows_{a.tag}.pkl").write_bytes(pickle.dumps(rows))
        print(f"{len(rows)} clips in {time.time() - t0:.0f} s with {a.processes} processes; library {RealtimeVAD.__file__}",
              file=sys.stderr)
    for tag in a.report or []:
        print(report(tag, pickle.loads((HERE / f"early_pause_rows_{tag}.pkl").read_bytes())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
