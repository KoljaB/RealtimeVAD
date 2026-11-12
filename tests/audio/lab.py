"""Audio lab for the edge-case suite: clips, effects, detector runs and measurements.

Every case is one stream of audio (a clip plus effects, or several clips in a row) fed to
RealtimeVAD in 10 ms chunks. The same frame probabilities are then replayed through a
second VAD with the quiet clock off (``quiet_rms=0``): that is "Silero alone", on identical
input. Cases run in a process pool; tests only read the results.

The reference for "the speaker is talking" comes from the clean clip: frames Silero scores
p >= 0.5 inside a Silero segment that are no more than 20 dB below the speaker's level and
at least 10 dB above the room (Silero's probability lags into the silence after a word; that
tail is not speech).
"""

from __future__ import annotations

import json
import math
import os
import wave
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
SR = 16000
FRAME = 512
FRAME_S = FRAME / SR
CHUNK = 160  # fed in 10 ms chunks


# -- clips ------------------------------------------------------------------
@lru_cache(maxsize=None)
def catalog() -> tuple:
    """All clips of both manifests, each with an absolute ``file``."""
    out = []
    for name in ("manifest.json", "backchannels.json"):
        for clip in json.loads((HERE / name).read_text(encoding="utf-8"))["clips"]:
            path = clip["path"].replace("\\", "/")
            path = path[len("tests/audio/"):] if path.startswith("tests/audio/") else path
            out.append({**clip, "file": str(HERE / path), "id": Path(path).stem})
    return tuple(out)


def clips(category: str, every: int = 1, **where) -> list:
    """Clips of a category in manifest order; ``every`` thins them out evenly."""
    found = [c for c in catalog() if c["category"] == category and all(c.get(k) == v for k, v in where.items())]
    return found[::every]


def available() -> bool:
    return (HERE / "manifest.json").exists() and all(os.path.exists(c["file"]) for c in catalog())


def load(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


# -- effects (numpy only, deterministic) -------------------------------------
def db(x: np.ndarray) -> float:
    return 10 * math.log10(float(np.mean(x * x)) + 1e-12)


def quantize(x):
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768


def band(x, low, high):
    """Zero-phase band limit with 100 Hz raised-cosine edges."""
    spectrum = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    edge = lambda f0: np.clip((f - (f0 - 50)) / 100, 0, 1)  # noqa: E731
    mask = (0.5 - 0.5 * np.cos(np.pi * edge(low))) * (0.5 + 0.5 * np.cos(np.pi * edge(high)))
    return np.fft.irfft(spectrum * mask, len(x)).astype(np.float32)


def fx_phone(x, rng):
    """Telephone path: 300-3400 Hz, 8 kHz, G.711 mu-law, line noise at -70 dBFS."""
    y = band(x, 300, 3400)[::2]
    c = np.sign(y) * np.log1p(255 * np.abs(np.clip(y, -1, 1))) / np.log1p(255)
    c = np.round(c * 127.5) / 127.5
    y = np.sign(c) * np.expm1(np.abs(c) * np.log1p(255)) / 255
    spectrum = np.zeros(len(x) // 2 + 1, dtype=complex)
    half = np.fft.rfft(y)
    spectrum[: len(half)] = half * 2
    y = np.fft.irfft(spectrum, len(x))
    return (y + rng.standard_normal(len(x)) * 10 ** (-70 / 20)).astype(np.float32)


def fx_reverb(x, rng, rt60):
    """Synthetic room: exponentially decaying noise tail after the direct sound."""
    n = int(rt60 * SR)
    tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    tail[: int(0.005 * SR)] = 0
    ir = np.concatenate(([1.0], 0.5 * tail / np.sqrt(np.sum(tail * tail))))
    size = len(x) + len(ir) - 1
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)].astype(np.float32)


def fx_gate(x, rng, threshold_db):
    """Noise suppression that mutes: 10 ms blocks below the threshold become digital zero."""
    y = x.copy()
    for i in range(0, len(y) - CHUNK + 1, CHUNK):
        if db(y[i : i + CHUNK]) < threshold_db:
            y[i : i + CHUNK] = 0
    return y


def fx_fade_tail(x, rng, start_s, seconds, down_db):
    """The last word fades out: a linear-in-dB ramp over ``seconds`` ending at ``start_s + seconds``."""
    y = x.copy()
    a, b = int(start_s * SR), min(len(y), int((start_s + seconds) * SR))
    y[a:b] *= 10 ** (np.linspace(0, -down_db, b - a) / 20)
    y[b:] *= 10 ** (-down_db / 20)
    return y


def fx_clicks(x, rng, every_s):
    """A 2 ms full-scale click every ``every_s``."""
    y = x.copy()
    for i in range(int(0.5 * SR), len(y) - 64, int(every_s * SR)):
        y[i : i + 32] += 0.8 * np.hanning(32) * rng.choice([-1, 1])
    return y


def apply(x, fx, level_db, rng):
    """``fx``: list of (name, *args). ``level_db`` is the clean clip's speech level."""
    for name, *args in fx:
        if name == "level":  # bring the speech to this dBFS
            x = x * 10 ** ((args[0] - level_db) / 20)
            level_db = args[0]
        elif name == "gain":
            x = x * 10 ** (args[0] / 20)
            level_db += args[0]
        elif name == "noise":  # background file at this speech-to-noise ratio
            noise = load(args[0])
            noise = np.resize(noise[int(args[2] * SR) if len(args) > 2 else 0 :], len(x))
            x = x + noise * 10 ** ((level_db - args[1] - db(noise)) / 20)
        elif name == "white":  # white noise at an absolute dBFS
            x = x + rng.standard_normal(len(x)).astype(np.float32) * 10 ** (args[0] / 20)
        elif name == "dc":
            x = x + args[0]
        elif name == "clip":
            x = np.clip(x, -args[0], args[0])
        elif name == "pad":  # silence before and after, seconds
            x = np.concatenate((np.zeros(int(args[0] * SR), np.float32), x, np.zeros(int(args[1] * SR), np.float32)))
        elif name == "cut":  # keep [from, to) seconds
            x = x[int(args[0] * SR) : int(args[1] * SR)]
        else:
            x = globals()["fx_" + name](x, rng, *args)
    return quantize(np.asarray(x, dtype=np.float32))


# -- detector runs -------------------------------------------------------------
_silero = None


def _model():
    global _silero
    if _silero is None:
        from RealtimeVAD.silero import Silero

        _silero = Silero(restore_s=0)
    return _silero


def detect(x, playback=(), chunk=CHUNK, probability=None, **kw):
    """Feed ``x``; returns frames p, events (kind, time, emitted at, source) and segments."""
    from RealtimeVAD import VAD

    vad = VAD(frame_events=True, onset_events=True, probability=probability or _model(), **kw)
    p, events, segments = [], [], []
    for i in range(0, len(x), chunk):
        t = i / SR
        echo = any(a <= t < b for a, b in playback)
        for e in vad.feed(x[i : i + chunk], playback=echo):
            if e.kind == "frame":
                p.append(e.frame.probability)
                continue
            emitted = min(len(x), i + chunk) / SR
            events.append((e.kind, round(e.time, 4), round(emitted, 4), e.source))
            if e.kind == "end":
                s = e.segment
                segments.append({"from": s.end - len(s.audio) / SR, "start": s.start, "speech_end": s.speech_end,
                                 "end": s.end, "quiet": abs(s.end - s.speech_end - 8 * FRAME_S) > 1e-6})
    running = vad.segment()
    if running is not None:
        segments.append({"from": len(x) / SR - len(running.audio) / SR, "start": running.start,
                         "speech_end": None, "end": None, "quiet": False})
    return np.asarray(p, dtype=np.float32), events, segments


def _scripted(p):
    it = iter(p.tolist())
    return lambda frame: next(it, 0.0)


def reference(file: str) -> dict:
    """Speech of a clean clip: frame mask, level, first and last speech time, room noise."""
    x = load(file)
    p, _, segments = detect(x, quiet_rms=0)
    n = len(p)
    level = np.array([db(x[i * FRAME : (i + 1) * FRAME]) for i in range(n)])
    t = (np.arange(n) + 1) * FRAME_S
    inside = np.zeros(n, bool)
    for s in segments:
        inside |= (t >= s["from"]) & (t <= (s["end"] if s["end"] is not None else t[-1] + 1))
    sure = level[p >= 0.9]
    speaker = float(np.median(sure)) if len(sure) else float(np.max(level))
    blocks = x[: len(x) // CHUNK * CHUNK].reshape(-1, CHUNK)
    noise = float(np.percentile(10 * np.log10((blocks * blocks).mean(axis=1) + 1e-12), 10))
    mask = (p >= 0.5) & inside & (level > max(speaker - 20, noise + 10))
    power = float(np.mean([10 ** (v / 10) for v in level[mask]])) if mask.any() else 1e-12
    return {"mask": mask, "level_db": 10 * math.log10(power), "noise_db": noise, "seconds": len(x) / SR,
            "on": float(t[mask][0] - FRAME_S) if mask.any() else None, "off": float(t[mask][-1]) if mask.any() else None}


def run_case(case: dict) -> dict:
    """One stream. ``case``: parts [(file, fx, level_db)], gaps between parts, optional bed, VAD kwargs."""
    rng = np.random.default_rng(case.get("seed", 0))
    pieces, offsets, t = [], [], 0.0
    for file, fx, level_db in case["parts"]:
        y = apply(load(file), fx, level_db, rng)
        offsets.append(t)
        pieces.append(y)
        t += len(y) / SR
    x = np.concatenate(pieces)
    if case.get("post"):
        x = apply(x, case["post"], 0.0, rng)
    kw = case.get("kw", {})
    p, events, segments = detect(x, playback=case.get("playback", ()), chunk=case.get("chunk", CHUNK), **kw)
    _, s_events, s_segments = detect(x, playback=case.get("playback", ()), probability=_scripted(p),
                                     **{**kw, "quiet_rms": 0})
    return {"key": case["key"], "p": p, "events": events, "segments": segments, "silero_events": s_events,
            "silero_segments": s_segments, "offsets": offsets, "seconds": len(x) / SR}


def run_all(cases: list, refs_for: list) -> tuple:
    """References for the clean files, then every case, in a process pool."""
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as pool:
        refs = dict(zip(refs_for, pool.map(reference, refs_for, chunksize=4)))
        for case in cases:  # fill in each part's clean speech level
            case["parts"] = [(f, fx, refs[f]["level_db"] if f in refs else 0.0) for f, fx in case["parts"]]
        cases = sorted(cases, key=lambda c: -len(c["parts"]))
        results = {r["key"]: r for r in pool.map(run_case, cases, chunksize=2)}
    return refs, results


# -- measurements --------------------------------------------------------------
def frame_times(n):
    return (np.arange(n) + 1) * FRAME_S


def covered(result, ref, offset=0.0, who="segments", after=None):
    """Fraction of the reference speech frames that lie inside a segment."""
    t = frame_times(len(ref["mask"]))[ref["mask"]] + offset
    if after is not None:
        t = t[t > after]
    if not len(t):
        return 1.0
    inside = np.zeros(len(t), bool)
    for s in result[who]:
        inside |= (t >= s["from"]) & (t <= (s["end"] if s["end"] is not None else math.inf) + FRAME_S)
    return float(inside.mean())


def premature_ends(result, ref, offset=0.0, who="segments"):
    """Ends decided while the speaker was talking: reference speech in the 0.15 s before the
    end was decided and at least 100 ms of it in the 0.5 s after."""
    t = frame_times(len(ref["mask"]))[ref["mask"]] + offset
    count = 0
    for s in result[who]:
        if s["end"] is None:
            continue
        before = np.sum((t > s["end"] - 0.15) & (t <= s["end"]))
        after = np.sum((t > s["end"]) & (t <= s["end"] + 0.5))
        count += bool(before >= 1 and after >= 3)
    return count


def first_event(result, kind, after=0.0, who="events"):
    """(time, emitted at) of the first event of a kind emitted after ``after``."""
    for k, t, emitted, _ in result[who]:
        if k == kind and emitted > after:
            return t, emitted
    return None


def last_end(result, who="segments"):
    ends = [s["end"] for s in result[who] if s["end"] is not None]
    return ends[-1] if ends else None


def speech_seconds(result, who="segments"):
    """Total length of all segments (open ones count up to the end of the stream)."""
    return sum((s["end"] if s["end"] is not None else result["seconds"]) - s["start"] for s in result[who])
