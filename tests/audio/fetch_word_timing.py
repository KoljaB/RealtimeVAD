"""Turn-like clips with word timings from many speakers, for benchmarks/word_timing.py.

Two corpora, both CC BY 4.0 (see tests/audio/LICENSES-word-timing.md):

  AMI Meeting Corpus      conversational English, 18 meetings of 18 different groups in three
                          rooms, 72 speakers. Word times from the AMI manual annotations
                          (manual transcript, forced-aligned).
  LibriSpeech             read English, dev-clean, test-clean, dev-other, test-other, 146
                          speakers. Word times from the Montreal Forced Aligner
                          ("LibriSpeech Alignments", Lugosch et al., Zenodo 2619474).

A clip is one stretch of a speaker's words with its real silence before and after, cut
sample for sample (LibriSpeech: decoded from FLAC), no level change:

  ami_headset        the speaker's own headset; nobody else speaks anywhere in the clip
  ami_far            the same clips from one distant table microphone (Array1-01)
  ami_headset_xtalk  the speaker's own headset while others speak somewhere in the clip
  libri_clean        dev-clean and test-clean
  libri_other        dev-other and test-other (harder recordings: noisier, more accents)

AMI: a clip is a run of the speaker's words with no gap of 1 s or more, 1 s of context on
both sides, in which the speaker makes no other annotated sound. Without crosstalk the
context ends where another participant's words begin: at least 0.5 s before and 0.6 s after
the speaker's words are silent for everyone (people answer fast, so 1 s is rare). LibriSpeech: the words
between two pauses of 0.5 s or more (or the file's start/end), with up to 1 s of the pause
before and all of the pause after, so the trailing silence is 0.47 s or more, often not
much more: LibriSpeech files end about 0.5 s after the last word.

    python tests/audio/fetch_word_timing.py            # download, cut, verify
    python tests/audio/fetch_word_timing.py --select   # maintainer: rewrite the manifest

Downloads (5.7 GB: AMI 3.7 GB, LibriSpeech 1.3 GB, alignments 0.6 GB) go to
tests/audio/_cache/, clips (1.1 GB) and their list to tests/audio/data/word_timing/;
both are git-ignored. Every download is checked against the SHA-256 in
tests/audio/word_timing.json, and so is the clip list with all clip audio.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_backchannels import (ANNOT_SHA256, ANNOT_URL, Annotations, download, is_word,  # noqa: E402
                                sha256_file, write_wav)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / "_cache"
OUT = HERE / "data" / "word_timing"
CLIPS = OUT / "clips.json"
MANIFEST = HERE / "word_timing.json"
SR = 16000

AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.{ch}.wav"
# One meeting per participant group (72 different speakers): Edinburgh (ES), Idiap (IS), TNO (TS).
AMI_MEETINGS = ["ES2002a", "ES2004c", "ES2006d", "ES2008a", "ES2010a", "ES2013a", "ES2015a",
                "IS1001a", "IS1004a", "IS1006a", "IS1008a", "IS1009b",
                "TS3003a", "TS3005a", "TS3006b", "TS3008a", "TS3010a", "TS3012a"]
AMI_FAR = "Array1-01"
AMI_CHANNELS = ("Headset-0", "Headset-1", "Headset-2", "Headset-3", AMI_FAR)
AMI_RUN_GAP_S = 1.0
AMI_CONTEXT_S = 1.0
AMI_MAX_S = 30.0
AMI_MIN_S = 0.2
AMI_PER_SPEAKER = {False: 150, True: 60}  # clips per speaker: alone, with crosstalk
AMI_SOLO_LEAD_S = 0.5  # a clip without crosstalk has at least this much silence of everyone before...
AMI_SOLO_TRAIL_S = 0.6  # ...and after the speaker's words
AMI_KEEP_OUT_S = 0.05  # distance kept to the others' words

LIBRI_URL = "https://openslr.trmal.net/resources/12/{name}.tar.gz"  # openslr.org/12, EU mirror
LIBRI_SETS = {"dev-clean": "libri_clean", "test-clean": "libri_clean",
              "dev-other": "libri_other", "test-other": "libri_other"}
LIBRI_ALIGN_URL = "https://zenodo.org/records/2619474/files/librispeech_alignments.zip"
LIBRI_PAUSE_S = 0.5
LIBRI_LEAD_MIN_S = 0.25
LIBRI_KEEP_OUT_S = 0.03  # distance kept to the neighbouring words

_INTERVAL = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*)"')


def stable(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


# -------------------------------------------------------------------------- AMI


def ami_pieces(ann: Annotations, meeting: str):
    meta = ann.meetings[meeting]
    # the shortest of the meeting's recordings (meetings.xml durations can exceed the audio)
    length = min(sf.info(str(cache_path(AMI_AUDIO.format(m=meeting, ch=ch)))).frames for ch in AMI_CHANNELS) / SR
    spoken, other_sound = {}, {}
    for agent in meta["speakers"]:
        items = ann.items(meeting, agent)
        spoken[agent] = [it for it in items if is_word(it) and it["start"] is not None and it["end"] is not None]
        # everything else the speaker's channel is annotated with (laughs, coughs, gaps in
        # the transcription, untimed words); disfluency markers carry no sound
        other_sound[agent] = [it for it in items if not (is_word(it) and it["start"] is not None)
                              and it["kind"] != "disfmarker" and not it["punc"]]
    out = []
    for agent, spk in sorted(meta["speakers"].items()):
        items = ann.items(meeting, agent)
        index = {it["id"]: i for i, it in enumerate(items)}
        runs, cur = [], []
        for w in spoken[agent]:
            if cur and w["start"] - cur[-1]["end"] >= AMI_RUN_GAP_S:
                runs.append(cur)
                cur = []
            cur.append(w)
        if cur:
            runs.append(cur)
        for run in runs:
            t0, t1 = run[0]["start"], run[-1]["end"]
            c0, c1 = t0 - AMI_CONTEXT_S, t1 + AMI_CONTEXT_S
            if not (AMI_MIN_S <= t1 - t0 <= AMI_MAX_S) or c0 < 0.5 or c1 > length - 0.5:
                continue
            if any(w["end"] <= w["start"] for w in run):
                continue
            bad = False
            for it in other_sound[agent]:
                if it["start"] is None or it["end"] is None:  # untimed: judged by its position
                    i = index[it["id"]]
                    bad |= index[run[0]["id"]] - 2 <= i <= index[run[-1]["id"]] + 2
                else:
                    bad |= it["start"] < c1 and it["end"] > c0
            if bad:
                continue
            # the others' words and vocal sounds near the run: the context is shortened to
            # where they are silent; if that leaves too little, or they speak during the
            # run, the clip counts as crosstalk and keeps the full context
            theirs = [(it["start"], it["end"]) for a in meta["speakers"] if a != agent
                      for it in spoken[a] + [x for x in other_sound[a] if x["kind"] == "vocalsound"]
                      if it["start"] is not None and it["end"] is not None and it["start"] < c1 and it["end"] > c0]
            before = max([e for s, e in theirs if e <= t0] + [c0 - AMI_KEEP_OUT_S]) + AMI_KEEP_OUT_S
            after = min([s for s, e in theirs if s >= t1] + [c1 + AMI_KEEP_OUT_S]) - AMI_KEEP_OUT_S
            others = (any(s < t1 and e > t0 for s, e in theirs)
                      or t0 - before < AMI_SOLO_LEAD_S or after - t1 < AMI_SOLO_TRAIL_S)
            if not others:
                c0, c1 = before, after
            s0, s1 = round(c0 * SR), round(c1 * SR)
            out.append({
                "id": f"{meeting}_{agent}_{round(c0 * 100):06d}",
                "corpus": "ami", "speaker": spk["speaker"], "gender": spk["gender"],
                "meeting": meeting, "agent": agent, "headset": spk["channel"], "others": others,
                "s0": s0, "s1": s1,
                "words": [[w["text"], round(w["start"] - s0 / SR, 3), round(w["end"] - s0 / SR, 3)] for w in run],
            })
    return out


def ami_select(ann: Annotations):
    clips = []
    per = collections.defaultdict(list)
    for m in AMI_MEETINGS:
        for p in ami_pieces(ann, m):
            per[(p["speaker"], p["others"])].append(p)
    for (speaker, others), ps in sorted(per.items()):
        ps = sorted(sorted(ps, key=lambda p: stable(p["id"]))[:AMI_PER_SPEAKER[others]], key=lambda p: p["id"])
        for p in ps:
            base = {k: p[k] for k in ("id", "corpus", "speaker", "gender", "meeting", "agent")}
            for cond, ch in ([("ami_headset_xtalk", f"Headset-{p['headset']}")] if others else
                             [("ami_headset", f"Headset-{p['headset']}"), ("ami_far", AMI_FAR)]):
                clips.append({**base, "condition": cond, "path": f"{cond}/{p['id']}.wav",
                              "source": AMI_AUDIO.format(m=p["meeting"], ch=ch),
                              "source_start_sample": p["s0"], "source_end_sample": p["s1"],
                              "duration_s": round((p["s1"] - p["s0"]) / SR, 4), "words": p["words"]})
    return clips


def ami_sources():
    return [AMI_AUDIO.format(m=m, ch=ch) for m in AMI_MEETINGS
            for ch in AMI_CHANNELS]


# ------------------------------------------------------------------ LibriSpeech


def libri_words(textgrid: str):
    """[(start, end, text)] of the word tier; text "" is silence."""
    tier = textgrid.split('name = "phones"')[0]
    return [(float(a), float(b), t) for a, b, t in _INTERVAL.findall(tier)]


def libri_select(align_zip: Path, genders: dict):
    z = zipfile.ZipFile(align_zip)
    clips = []
    for name in sorted(z.namelist()):
        subset = name.split("/")[0]
        if subset not in LIBRI_SETS or not name.endswith(".TextGrid"):
            continue
        utt = name.rsplit("/", 1)[1][:-9]
        speaker = utt.split("-")[0]
        iv = libri_words(z.read(name).decode("utf-8"))
        if not iv:
            continue
        total = iv[-1][1]
        # boundaries: the file's start and end and every pause of LIBRI_PAUSE_S or more
        groups, cur, lead = [], [], iv[0][1] - iv[0][0] if iv[0][2] == "" else 0.0
        prev_end = 0.0
        for i, (a, b, t) in enumerate(iv):
            if t != "":
                cur.append((a, b, t))
                continue
            last = i == len(iv) - 1
            if i and (b - a >= LIBRI_PAUSE_S or last) and cur:
                groups.append((cur, lead, prev_end, b if last else b - LIBRI_KEEP_OUT_S, b - a))
                cur = []
            if not cur:
                lead, prev_end = b - a, a
        for k, (ws, lead, prev_end, c1, trail) in enumerate(groups):
            t0 = ws[0][0]
            c0 = max(t0 - 1.0, prev_end + (LIBRI_KEEP_OUT_S if prev_end > 0 else 0.0))
            if t0 - c0 < LIBRI_LEAD_MIN_S or trail < LIBRI_PAUSE_S or any(t == "<unk>" for _, _, t in ws):
                continue
            s0, s1 = round(c0 * SR), round(min(c1, total) * SR)
            clips.append({
                "id": f"{utt}_{k}", "corpus": "librispeech", "speaker": f"ls{speaker}",
                "gender": genders.get(speaker), "subset": subset, "utterance": utt,
                "condition": LIBRI_SETS[subset], "path": f"{LIBRI_SETS[subset]}/{utt}_{k}.wav",
                "source": LIBRI_URL.format(name=subset),
                "source_start_sample": s0, "source_end_sample": s1,
                "duration_s": round((s1 - s0) / SR, 4),
                "words": [[t, round(a - s0 / SR, 3), round(b - s0 / SR, 3)] for a, b, t in ws],
            })
    return clips


def libri_genders(tars) -> dict:
    out = {}
    for tar in tars:
        with tarfile.open(tar) as t:
            for line in t.extractfile("LibriSpeech/SPEAKERS.TXT").read().decode("utf-8", "replace").splitlines():
                if not line.startswith(";") and "|" in line:
                    f = [x.strip() for x in line.split("|")]
                    out[f[0]] = f[1]
        break  # the same file is in every archive
    return out


# --------------------------------------------------------------------- pipeline


def cache_path(url: str) -> Path:
    name = url.rsplit("/", 1)[1]
    return CACHE / ("ami" if "amicorpus" in url or "AMICorpusAnnotations" in url else "librispeech") / name


def cut(clips, select_mode: bool) -> None:
    by_src = collections.defaultdict(list)
    for c in clips:
        by_src[c["source"]].append(c)

    def done(c) -> bool:
        p = OUT / c["path"]
        return p.exists() and bool(c.get("sha256")) and sha256_file(p) == c["sha256"]

    def finish(c, pcm) -> None:
        write_wav(OUT / c["path"], pcm)
        if select_mode:
            c["sha256"] = sha256_file(OUT / c["path"])

    def ami(url) -> None:
        cs = [c for c in by_src[url] if not done(c)]
        if not cs:
            return
        with sf.SoundFile(str(cache_path(url))) as f:
            if (f.samplerate, f.channels, f.subtype) != (SR, 1, "PCM_16"):
                raise SystemExit(f"unexpected format in {url}")
            for c in cs:
                if c["source_end_sample"] > f.frames:
                    raise SystemExit(f"{c['id']}: beyond the end of {url}")
                f.seek(c["source_start_sample"])
                finish(c, f.read(c["source_end_sample"] - c["source_start_sample"], dtype="int16"))

    def libri(url) -> None:
        todo = collections.defaultdict(list)
        for c in by_src[url]:
            if not done(c):
                todo[c["utterance"]].append(c)
        if not todo:
            return
        with tarfile.open(cache_path(url)) as t:
            for member in t:
                utt = member.name.rsplit("/", 1)[-1][:-5]
                if not member.name.endswith(".flac") or utt not in todo:
                    continue
                pcm, sr = sf.read(io.BytesIO(t.extractfile(member).read()), dtype="int16")
                if sr != SR or pcm.ndim != 1:
                    raise SystemExit(f"unexpected format in {member.name}")
                for c in todo.pop(utt):
                    if c["source_end_sample"] > len(pcm) + 160:
                        raise SystemExit(f"{c['id']}: beyond the end of {member.name}")
                    finish(c, pcm[c["source_start_sample"]:c["source_end_sample"]])
        if todo:
            raise SystemExit(f"{len(todo)} utterances missing in {url}")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda u: ami(u) if "amicorpus" in u else libri(u), sorted(by_src)))


def selection_hash(clips) -> str:
    return hashlib.sha256(json.dumps(clips, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build(select_mode: bool) -> int:
    t_start = time.time()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
    if manifest is None and not select_mode:
        raise SystemExit("no manifest; run with --select first")
    known = {s["url"]: s["sha256"] for s in (manifest or {}).get("sources", [])}
    urls = [ANNOT_URL, LIBRI_ALIGN_URL] + [LIBRI_URL.format(name=n) for n in LIBRI_SETS] + ami_sources()
    known.setdefault(ANNOT_URL, ANNOT_SHA256)
    if not select_mode and any(u not in known for u in urls):
        raise SystemExit("word_timing.json lacks a source checksum")

    def get(url):
        return download(url, cache_path(url), known.get(url))

    with ThreadPoolExecutor(5) as pool:
        downloaded = sum(pool.map(get, urls))

    ann = Annotations(cache_path(ANNOT_URL))
    genders = libri_genders([cache_path(LIBRI_URL.format(name=n)) for n in LIBRI_SETS])
    clips = ami_select(ann) + libri_select(cache_path(LIBRI_ALIGN_URL), genders)
    counts = collections.Counter(c["condition"] for c in clips)
    speakers = {k: len({c["speaker"] for c in clips if c["condition"] == k}) for k in counts}

    if select_mode:
        cut(clips, True)
        manifest = {
            "description": "Sources of the word-timing benchmark clips. Rebuild the clips with "
                           "python tests/audio/fetch_word_timing.py",
            "license": "CC BY 4.0 (AMI Meeting Corpus; LibriSpeech; LibriSpeech Alignments)",
            "details": "tests/audio/LICENSES-word-timing.md",
            "clips": dict(sorted(counts.items())),
            "speakers": dict(sorted(speakers.items())),
            "clips_sha256": selection_hash(clips),
            "sources": [{"url": u, "sha256": sha256_file(cache_path(u)), "bytes": cache_path(u).stat().st_size}
                        for u in urls],
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.name}")
    else:
        # the committed hash covers selection, word times and every clip's audio checksum
        old = {c["id"] + "/" + c["condition"]: c for c in json.loads(CLIPS.read_text(encoding="utf-8"))} \
            if CLIPS.exists() else {}
        for c in clips:
            c["sha256"] = old.get(c["id"] + "/" + c["condition"], {}).get("sha256")
        cut([c for c in clips], False)
        for c in clips:
            c["sha256"] = sha256_file(OUT / c["path"])
        if selection_hash(clips) != manifest["clips_sha256"]:
            raise SystemExit("the clips differ from the ones word_timing.json was made with "
                             "(selection, word times or audio)")
    OUT.mkdir(parents=True, exist_ok=True)
    CLIPS.write_text(json.dumps(clips, ensure_ascii=False) + "\n", encoding="utf-8")
    hours = sum(c["duration_s"] for c in clips) / 3600
    print(f"{len(clips)} clips, {hours:.1f} h: " + ", ".join(f"{k} {v} ({speakers[k]} speakers)"
                                                               for k, v in sorted(counts.items())))
    print(f"downloaded {downloaded / 1e6:.0f} MB; {time.time() - t_start:.0f} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true", help="maintainer: re-select clips and rewrite the manifest")
    return build(ap.parse_args().select)


if __name__ == "__main__":
    sys.exit(main())
