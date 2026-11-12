#!/usr/bin/env python3
"""Download and build the freely licensed test audio described in manifest.json.

    python tests/audio/fetch_audio.py            # download what is missing, build data/, verify
    python tests/audio/fetch_audio.py --verify   # only check data/ against the manifest (no network)

Downloads go to tests/audio/_cache/ and the clips to tests/audio/data/<category>/<name>.wav
(both git-ignored). Every clip is 16 kHz, mono, 16-bit PCM WAV. Running the script again skips
everything that is already present and correct.

Integrity:
  * whole-file downloads are checked against the sha256 in the manifest;
  * for big .tar.gz archives (FLEURS) only the start of the archive is streamed, up to the
    last member that is needed; each extracted member is checked against its sha256 instead
    of the archive's (the archive's own sha256 is recorded for reference);
  * every produced clip is checked against its sha256.

Processing never changes levels: no normalisation, no gain. A clip is cut from the source
(offset_s/length_s), multi-channel audio is averaged to mono, and anything not at 16 kHz is
resampled with scipy.signal.resample_poly, and float samples are quantised to 16 bit
(round(x * 32768), limited to the int16 range). Sources that already are 16 kHz mono 16-bit
are copied sample for sample.

Dependencies: Python 3.9+, numpy, scipy, soundfile (libsndfile >= 1.0.29 decodes the Ogg
Vorbis files; no ffmpeg needed). Lossy sources (Ogg Vorbis) are decoded by libsndfile; if a
different libsndfile build decodes them differently, those clips fail the hash check with a
message saying so.

Maintainers: --record writes the sha256/size of downloads and clips whose manifest value is
empty (use it after adding entries; it never overwrites a recorded value).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
CACHE = HERE / "_cache"
DATA = HERE / "data"
SR = 16000
USER_AGENT = "RealtimeVAD-test-audio-fetch/1.0 (+https://github.com/KoljaB/RealtimeVAD)"
CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def open_url(url: str, attempts: int = 4):
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            return urllib.request.urlopen(req, timeout=60)
        except Exception as e:  # network hiccup: retry with backoff
            last = e
            time.sleep(2 * (i + 1))
    raise FetchError(f"cannot open {url}: {last}")


class Stats:
    downloaded = 0


def mismatch(what: str, expected: str, actual: str, hint: str = "") -> FetchError:
    msg = f"sha256 mismatch for {what}\n  expected {expected}\n  actual   {actual}"
    if hint:
        msg += f"\n  {hint}"
    return FetchError(msg)


# --------------------------------------------------------------------------- downloads

def download_file(key: str, dl: dict, record: bool) -> Path:
    """Whole-file download into _cache/<key>/<filename>, verified by sha256."""
    dest = CACHE / key / dl["filename"]
    want = dl.get("sha256", "")
    if dest.exists() and want and sha256_file(dest) == want:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  downloading {key}: {dl['url']}")
    h = hashlib.sha256()
    n = 0
    with open_url(dl["url"]) as r, open(part, "wb") as f:
        for block in iter(lambda: r.read(CHUNK), b""):
            f.write(block)
            h.update(block)
            n += len(block)
    Stats.downloaded += n
    got = h.hexdigest()
    if not want:
        if not record:
            part.unlink()
            raise FetchError(f"download {key} has no sha256 in the manifest (run with --record)")
        dl["sha256"], dl["bytes"] = got, n
    elif got != want:
        part.unlink()
        raise mismatch(f"download {key} ({dl['url']})", want, got,
                       "the file on the server changed or the download was corrupted")
    part.replace(dest)
    return dest


def fetch_tar_members(key: str, dl: dict, needed: set[str], record: bool) -> None:
    """Stream a (gzipped) tar from the start and extract only `needed` members.

    Stops reading as soon as the last needed member has been extracted, so for members near
    the start of a big archive only a fraction of it is transferred.
    """
    members = dl["members"]
    todo = set()
    for name in needed:
        p = CACHE / key / name
        want = members[name].get("sha256", "")
        if not (p.exists() and want and sha256_file(p) == want):
            todo.add(name)
    if not todo:
        return
    print(f"  streaming {key} for {len(todo)} member(s): {dl['url']}")

    class Counting:
        def __init__(self, f):
            self.f, self.n = f, 0

        def read(self, size=-1):
            b = self.f.read(size)
            self.n += len(b)
            return b

    with open_url(dl["url"]) as r:
        src = Counting(r)
        try:
            with tarfile.open(fileobj=src, mode="r|*") as tar:
                for m in tar:
                    if m.name not in todo:
                        continue
                    b = tar.extractfile(m).read()
                    got = sha256_bytes(b)
                    want = members[m.name].get("sha256", "")
                    if not want:
                        if not record:
                            raise FetchError(f"member {m.name} of {key} has no sha256 (run with --record)")
                        members[m.name]["sha256"], members[m.name]["bytes"] = got, len(b)
                    elif got != want:
                        raise mismatch(f"member {m.name} of {key}", want, got)
                    out = CACHE / key / m.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(b)
                    todo.discard(m.name)
                    if not todo:
                        break
        finally:
            Stats.downloaded += src.n
    if todo:
        raise FetchError(f"{key}: archive ended without members {sorted(todo)}")
    print(f"    read {src.n / 1e6:.1f} MB of the archive")


# --------------------------------------------------------------------------- audio

def source_bytes(clip: dict, downloads: dict) -> bytes:
    key = clip["download"]
    dl = downloads[key]
    if dl["kind"] == "tar_stream":
        return (CACHE / key / clip["source_file"]).read_bytes()
    path = CACHE / key / dl["filename"]
    if dl.get("container") == "tar":
        with tarfile.open(path, "r:") as tar:
            f = tar.extractfile(clip["source_file"])
            if f is None:
                raise FetchError(f"{clip['source_file']} not found in {path}")
            return f.read()
    return path.read_bytes()


def render(clip: dict, raw: bytes):
    """Return int16 mono samples at 16 kHz."""
    import numpy as np
    import soundfile as sf

    info = sf.info(io.BytesIO(raw))
    sr = info.samplerate
    start = int(round(clip.get("offset_s", 0.0) * sr))
    stop = info.frames if clip.get("length_s") is None else start + int(round(clip["length_s"] * sr))
    if stop > info.frames:
        raise FetchError(f"{clip['path']}: cut {start}:{stop} beyond source length {info.frames}")
    if sr == SR and info.channels == 1 and info.subtype == "PCM_16":
        x, _ = sf.read(io.BytesIO(raw), dtype="int16", start=start, stop=stop)
        return np.ascontiguousarray(x)
    from scipy.signal import resample_poly

    x, _ = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True, start=start, stop=stop)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    y = np.round(x * 32768.0)
    return np.clip(y, -32768, 32767).astype("<i2")


def write_wav(path: Path, samples) -> None:
    # stdlib writer: plain 44-byte RIFF header, byte-identical on every platform
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.astype("<i2").tobytes())
    tmp.replace(path)


def check_clip(clip: dict) -> str | None:
    """None if data/<path> is present and correct, otherwise the reason."""
    p = HERE / clip["path"]
    if not p.exists():
        return "missing"
    try:
        with wave.open(str(p), "rb") as w:
            fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
            frames = w.getnframes()
    except (wave.Error, EOFError) as e:
        return f"not a readable WAV ({e})"
    if fmt != (SR, 1, 2):
        return f"format {fmt} is not 16 kHz mono 16-bit"
    if abs(frames / SR - clip["duration_s"]) > 1.0 / SR + 1e-6:
        return f"duration {frames / SR:.4f} s != manifest {clip['duration_s']} s"
    got = sha256_file(p)
    if got != clip["sha256"]:
        return f"sha256 {got} != manifest {clip['sha256']}"
    return None


# --------------------------------------------------------------------------- main

def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(m: dict) -> None:
    tmp = MANIFEST.with_name(MANIFEST.name + ".part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
        f.write("\n")
    tmp.replace(MANIFEST)


def verify(manifest: dict) -> int:
    bad = 0
    for clip in manifest["clips"]:
        why = check_clip(clip)
        if why:
            bad += 1
            print(f"FAIL {clip['path']}: {why}")
    n = len(manifest["clips"])
    print(f"{n - bad}/{n} clips OK" + ("" if not bad else f", {bad} failed"))
    return 1 if bad else 0


def build(manifest: dict, record: bool) -> int:
    downloads = manifest["downloads"]
    clips = manifest["clips"]
    if not record:
        empty = [c["path"] for c in clips if not c.get("sha256")]
        if empty:
            raise FetchError(f"{len(empty)} clip(s) have no sha256 in the manifest, e.g. {empty[0]}")
    todo = [c for c in clips if not c.get("sha256") or check_clip(c)]
    print(f"{len(clips) - len(todo)}/{len(clips)} clips already present and correct")
    if not todo:
        return 0

    # 1. downloads needed by the missing clips
    by_dl: dict[str, list[dict]] = {}
    for c in todo:
        by_dl.setdefault(c["download"], []).append(c)
    for key, cs in by_dl.items():
        dl = downloads[key]
        if dl["kind"] == "tar_stream":
            fetch_tar_members(key, dl, {c["source_file"] for c in cs}, record)
        else:
            download_file(key, dl, record)

    # 2. render clips
    failed = []
    for c in todo:
        samples = render(c, source_bytes(c, downloads))
        dur = round(len(samples) / SR, 6)
        if record and not c.get("sha256"):
            c["duration_s"] = dur
        out = HERE / c["path"]
        write_wav(out, samples)
        if record and not c.get("sha256"):
            c["sha256"] = sha256_file(out)
        why = check_clip(c)
        if why:
            hint = " (lossy source: decoder output differs from the one used to record the manifest)" \
                if downloads[c["download"]].get("lossy") else ""
            failed.append(f"{c['path']}: {why}{hint}")
        else:
            print(f"  built {c['path']}")
    if record:
        save_manifest(manifest)
    if failed:
        raise FetchError("clip check failed:\n  " + "\n  ".join(failed))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="only check data/ against the manifest")
    ap.add_argument("--record", action="store_true",
                    help="maintainers: fill in empty sha256/size values in the manifest")
    args = ap.parse_args(argv)
    t0 = time.time()
    manifest = load_manifest()
    try:
        if args.verify:
            return verify(manifest)
        build(manifest, args.record)
        rc = verify(manifest)
    except FetchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"downloaded {Stats.downloaded / 1e6:.1f} MB in {time.time() - t0:.1f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
