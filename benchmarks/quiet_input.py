"""Does RealtimeVAD cut soft speech when the input is quiet (e.g. source-level phone audio)?

Each WAV (16 kHz mono int16, one user turn with trailing silence) is run at its recorded
level and as quiet / phone variants. The reference for "the speaker is talking" comes from
the *original* recording: frames Silero scores p >= 0.5 inside a Silero segment that are also
audible (above max(-54 dBFS, room noise + 15 dB); Silero's probability lags ~100 ms into the
silence after a word, that tail is not speech). A variant therefore cannot hide a loss by
making Silero itself less sure, and attenuating the input must not change what counts as
silence. Per variant, two detectors see the same audio:

  rvad    VAD()             RealtimeVAD with its defaults
  silero  VAD(quiet_rms=0)  same thresholds, quiet clock off = Silero's silence count alone

Reported per variant:
  premature    turns with an end that the same detector does not make on the original
               recording (within 0.2 s) and that is followed by >= 100 ms of reference speech
               within 1 s: the turn was ended while the speaker went on
  cut ends     turns with a quiet-clock end while the reference still has speech inside the
               quiet window (>= 2 frames; also the strict count with p >= 0.9)
  lost speech  turns where >= 100 ms of reference speech after the first start lies in no
               segment (rvad next to silero: the difference is the quiet clock's doing)
  resumed      (handover metric) turns where the variant's own Silero is confident again
               (p >= 0.9, 2 frames) within 0.3 s after a quiet-clock end
  no start     turns without any segment; third number: plain Silero at its default 0.5
               (two frames in a row) finds no speech either
  final end    rvad's last end vs silero's last end on the same audio, where both closed the
               same number of segments inside the file: median gain, and how often rvad is
               later (must be 0)
  speech-off   the pause-event lead of benchmarks/speech_off_lead.py

    python benchmarks/quiet_input.py <wav dir> <out.json> [processes] [max files] [VAD kwargs as JSON]
"""

import glob
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speech_off_lead import first_of_final_silent_run, load, vaditerator_end  # noqa: E402

from RealtimeVAD import VAD  # noqa: E402
from RealtimeVAD.silero import Silero  # noqa: E402

FRAME_S = 512 / 16000
VARIANTS = ("orig", "att6", "att10", "att20", "att30", "att15room", "phone", "phone10", "phone20")
if os.environ.get("QUIET_INPUT_VARIANTS"):  # e.g. "orig,att6,att15room"; "orig" must come first
    VARIANTS = tuple(os.environ["QUIET_INPUT_VARIANTS"].split(","))
ROOM_NOISE_DBFS = -62.0  # raw-microphone room noise for the quiet speaker in "att15room"
LINE_NOISE_DBFS = -70.0
_silero = None
_kwargs = {}


def quantize(x):
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768


def phone(x, seed):
    """Narrowband telephone path: 300-3400 Hz, 8 kHz, G.711 mu-law, back to 16 kHz (zero phase)."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    y = sosfiltfilt(butter(4, [300, 3400], btype="band", fs=16000, output="sos"), x)
    y = resample_poly(y, 1, 2)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
    c = np.round(c * 127.5) / 127.5  # 8 bit
    y = np.sign(c) * np.expm1(np.abs(c) * np.log1p(mu)) / mu
    return resample_poly(y, 2, 1)[: len(x)].astype(np.float32)


def variant(x, name, seed):
    if name == "orig":
        return x
    if name.endswith("room"):  # quiet speaker, unprocessed microphone: pink room noise
        spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(len(x)))
        spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
        pink = np.fft.irfft(spectrum, len(x))
        pink *= 10 ** (ROOM_NOISE_DBFS / 20) / np.sqrt(np.mean(pink * pink))
        return quantize(x * 10 ** (-int(name[3:-4]) / 20) + pink.astype(np.float32))
    if name.startswith("att"):
        return quantize(x * 10 ** (-int(name[3:]) / 20))
    y = phone(x, seed) * 10 ** (-int(name[5:] or 0) / 20)
    noise = np.random.default_rng(seed).standard_normal(len(y)).astype(np.float32)
    return quantize(y + noise * 10 ** (LINE_NOISE_DBFS / 20))


def detect(x, **kw):
    global _silero
    if _silero is None:
        _silero = Silero(restore_s=0)
    vad = VAD(frame_events=True, probability=_silero, **kw)
    frames, pauses, ends = [], [], []
    for i in range(0, len(x), 320):
        for e in vad.feed(x[i : i + 320]):
            if e.kind == "frame":
                frames.append((e.time, e.frame.probability))
            elif e.kind == "pause":
                pauses.append((e.time, e.source))
            elif e.kind == "end":
                s = e.segment
                by_quiet = s.early  # ended by the quiet clock before Silero's count
                ends.append({"t": s.end, "from": s.end - len(s.audio) / 16000, "start": s.start,
                             "speech_end": s.speech_end, "quiet": by_quiet})
    spans = [(e["from"], e["t"]) for e in ends]
    if vad.in_speech:
        spans.append((len(x) / 16000 - len(vad.audio()) / 16000, len(x) / 16000 + 1))
    return frames, pauses, ends, spans, vad.in_speech


def level_db(x, frame_times):
    if not frame_times:
        return None
    power = np.mean([np.mean(x[int(round(t * 16000)) - 512 : int(round(t * 16000))] ** 2) for t in frame_times])
    return round(float(10 * np.log10(power + 1e-12)), 1)


def measure(path):
    x = load(path)
    seed = zlib.crc32(os.path.basename(path).encode())
    blocks = x[: len(x) // 160 * 160].reshape(-1, 160)
    noise = float(20 * np.log10(np.percentile(np.sqrt((blocks * blocks).mean(axis=1)) + 1e-9, 10)))
    audible = max(-54.0, noise + 15.0)
    ref_frames, _, _, ref_spans, _ = detect(x, quiet_rms=0)  # always with default settings
    ref = [t for t, p in ref_frames if p >= 0.5 and any(a <= t <= b for a, b in ref_spans)
           and level_db(x, [t]) > audible]
    ref9 = {t for t, p in ref_frames if p >= 0.9} & set(ref)
    if len(ref) < 5:
        return {"file": os.path.basename(path), "skip": "no reference speech"}
    row = {"file": os.path.basename(path), "ref_speech_s": round(len(ref) * FRAME_S, 2), "v": {}}
    row["noise_dbfs"] = round(noise, 1)
    normal = {}
    for name in VARIANTS:
        y = variant(x, name, seed)
        out = {"speech_dbfs": level_db(y, ref)}
        for who, kw in (("rvad", _kwargs), ("silero", {**_kwargs, "quiet_rms": 0})):
            frames, pauses, ends, spans, running = detect(y, **kw)
            first = min((a for a, _ in spans), default=None)
            lost = [t for t in ref if not any(a <= t <= b + FRAME_S for a, b in spans)]
            out[who] = {
                "segments": len(spans),
                "lost_ms": round(len(lost) * FRAME_S * 1000),
                "lost_after_start_ms": round(sum(first is not None and t > first for t in lost) * FRAME_S * 1000),
                "last_end": ends[-1]["t"] if ends and not running else None,
            }
            normal.setdefault(who, [e["t"] for e in ends])  # "orig" comes first
            out[who]["premature"] = sum(
                all(abs(e["t"] - t) > 0.2 for t in normal[who])
                and sum(e["t"] < t <= e["t"] + 1.0 for t in ref) >= 3 for e in ends)
            if who == "silero":
                hot = [p >= 0.5 for _, p in frames]
                out["silero"]["default_hears"] = any(a and b for a, b in zip(hot, hot[1:]))
                out["p_max"] = round(max(p for _, p in frames), 3)
                out["frames_0.9"] = sum(p >= 0.9 for _, p in frames)
                continue
            cut = cut9 = resumed = 0
            for e in (e for e in ends if e["quiet"]):
                inside = [t for t in ref if e["speech_end"] + FRAME_S - 1e-6 <= t <= e["t"] + 1e-6]
                cut += len(inside) >= 2
                cut9 += sum(t in ref9 for t in inside) >= 2
                after = [p >= 0.9 for t, p in frames if e["t"] < t <= e["t"] + 0.3]
                resumed += any(a and b for a, b in zip(after, after[1:]))
            out["rvad"].update(quiet_ends=sum(e["quiet"] for e in ends), cut=cut, cut9=cut9, resumed=resumed)
            # speech-off lead at the final speech end, as in speech_off_lead.py
            if ends:
                end_t, seg_start = ends[-1]["t"], ends[-1]["start"]
                seg_pauses = [p for p in pauses if seg_start <= p[0] <= end_t]
                if seg_pauses:
                    off = seg_pauses[-1][0]
                    out["rvad"].update(off=off, pauses=len(seg_pauses), speech_s=round(end_t - seg_start, 2))
                    t5 = first_of_final_silent_run(frames, 0.5, end_t + 1.0)
                    vi = vaditerator_end([f for f in frames if f[0] >= seg_start - 0.2])
                    out["rvad"]["lead_ms_0.5"] = None if t5 is None else round((t5 - off) * 1000, 1)
                    out["rvad"]["lead_ms_vaditerator"] = None if vi is None else round((vi - off) * 1000, 1)
        row["v"][name] = out
    return row


def _init(kwargs):
    global _kwargs
    _kwargs = kwargs


def summarize(rows):
    ok = [r for r in rows if "skip" not in r]
    med = lambda v: float("nan") if not len(v) else float(np.median(v))  # noqa: E731
    print(f"{len(rows)} files, {len(ok)} measured")
    print(f"{'variant':9} {'speech':>7} {'premature':>11} {'cut':>5} {'cut.9':>5} {'resum':>5} {'lost>=100ms':>12} {'no start':>12} "
          f"{'final gain':>10} {'later':>5} {'off lead':>8} {'vs VADIt':>8} {'earlier':>7} {'false pause':>11}")
    print(f"{'':9} {'dBFS':>7} {'rvad/silero':>11} {'turns':>5} {'turns':>5} {'turns':>5} {'rvad/silero':>12} {'rvad/sil/0.5':>12} "
          f"{'med ms':>10} {'turns':>5} {'med ms':>8} {'med ms':>8} {'%':>7} {'per s':>11}")
    out = {}
    for name in VARIANTS:
        v = [r["v"][name] for r in ok]
        both = [(a["silero"]["last_end"], a["rvad"]["last_end"]) for a in v
                if a["silero"]["last_end"] is not None and a["rvad"]["last_end"] is not None
                and a["rvad"]["segments"] == a["silero"]["segments"]]
        gain = [(s - r) * 1000 for s, r in both]
        lead = [a["rvad"]["lead_ms_0.5"] for a in v if a["rvad"].get("lead_ms_0.5") is not None]
        lead_vi = [a["rvad"]["lead_ms_vaditerator"] for a in v if a["rvad"].get("lead_ms_vaditerator") is not None]
        s = {
            "speech_dbfs": med([a["speech_dbfs"] for a in v]),
            "premature_rvad": sum(a["rvad"]["premature"] > 0 for a in v),
            "premature_silero": sum(a["silero"]["premature"] > 0 for a in v),
            "cut": sum(a["rvad"]["cut"] > 0 for a in v), "cut9": sum(a["rvad"]["cut9"] > 0 for a in v),
            "resumed": sum(a["rvad"]["resumed"] > 0 for a in v),
            "lost_rvad": sum(a["rvad"]["lost_after_start_ms"] >= 100 for a in v),
            "lost_silero": sum(a["silero"]["lost_after_start_ms"] >= 100 for a in v),
            "nostart_rvad": sum(a["rvad"]["segments"] == 0 for a in v),
            "nostart_silero": sum(a["silero"]["segments"] == 0 for a in v),
            "nostart_silero_default": sum(not a["silero"]["default_hears"] for a in v),
            "final_gain_ms": med(gain), "later": sum(g < -1e-3 for g in gain), "compared": len(gain),
            "lead_ms": med(lead), "lead_vi_ms": med(lead_vi),
            "earlier_pct": 100 * float(np.mean([m > 0 for m in lead])) if lead else float("nan"),
            # pauses before the last one of the final segment, per second of that segment
            "false_pauses_per_s": sum(a["rvad"].get("pauses", 1) - 1 for a in v)
            / max(1e-9, sum(a["rvad"].get("speech_s", 0.0) for a in v)),
        }
        out[name] = s
        print(f"{name:9} {s['speech_dbfs']:7.1f} {s['premature_rvad']:5d}/{s['premature_silero']:<5d} {s['cut']:5d} {s['cut9']:5d} {s['resumed']:5d} "
              f"{s['lost_rvad']:6d}/{s['lost_silero']:<5d} {s['nostart_rvad']:4d}/{s['nostart_silero']}/{s['nostart_silero_default']:<4d} "
              f"{s['final_gain_ms']:10.0f} {s['later']:5d} {s['lead_ms']:8.0f} {s['lead_vi_ms']:8.0f} {s['earlier_pct']:7.1f} {s['false_pauses_per_s']:11.2f}")
    return out


def main():
    files = sorted(glob.glob(sys.argv[1].rstrip("/\\") + "/*.wav"))
    if len(sys.argv) > 4 and int(sys.argv[4]):
        files = files[:: max(1, len(files) // int(sys.argv[4]))][: int(sys.argv[4])]
    kwargs = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4, initializer=_init, initargs=(kwargs,)) as pool:
        rows = pool.map(measure, files, chunksize=4)
    json.dump({"kwargs": kwargs, "summary": summarize(rows), "rows": rows}, open(sys.argv[2], "w"), indent=0)


if __name__ == "__main__":
    main()
