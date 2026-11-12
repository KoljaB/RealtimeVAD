"""Labelled backchannel, turn-start and hesitation clips from the AMI Meeting Corpus.

Source: AMI Meeting Corpus (https://groups.inf.ed.ac.uk/ami/corpus/), CC BY 4.0.
See tests/audio/LICENSES-backchannels.md for attribution.

Every clip is cut, sample for sample and without any level change, from the
speaker's own individual headset recording (16 kHz mono int16) at times taken
from the AMI manual annotations (forced-aligned word timings and dialogue acts).

    python tests/audio/fetch_backchannels.py            # download, select, cut, verify
    python tests/audio/fetch_backchannels.py --verify   # only check existing clips
    python tests/audio/fetch_backchannels.py --select   # maintainer: rewrite the manifest

Downloads (annotations and the needed headset WAVs) go to tests/audio/_cache/ami/,
clips to tests/audio/data/<category>/; both are git-ignored. Every download and
every clip is checked against the SHA-256 values in tests/audio/backchannels.json.
The selection is deterministic: a normal run re-derives it from the annotations
and stops if it differs from the manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache" / "ami"
DATA = HERE / "data"
MANIFEST = HERE / "backchannels.json"

SR = 16000
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CORPUS_URL = "https://groups.inf.ed.ac.uk/ami/corpus/"
ANNOT_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOT_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Headset-{ch}.wav"
ATTRIBUTION = (
    "Contains audio and annotations from the AMI Meeting Corpus "
    "(https://groups.inf.ed.ac.uk/ami/corpus/), licensed under CC BY 4.0; "
    "clips cut from the individual headset recordings, not otherwise modified."
)

# Meetings the clips come from (scenario meetings of different participant groups).
MEETINGS = ["ES2004c", "ES2006d", "IS1009b", "TS3006b"]

BCK = "bck"  # dialogue-act type "Backchannel" in ontologies/da-types.xml
CONTEXT_S = 1.0

BC_TARGET, BC_MIN_S, BC_MAX_S, BC_PER_SPEAKER = 200, 0.15, 0.8, 25
TS_TARGET, TS_MIN_S, TS_MAX_CLIP_S = 60, 2.0, 8.0
TS_EXCLUDE = {None, BCK, "stl", "fra"}  # untyped, backchannel, stall, fragment: not a real turn
HS_TARGET, HS_GAP = 30, (0.5, 1.5)
HS_CLIP = (6.0, 12.0)
HS_LISTED_GAP_S = 0.3

NITE = "{http://nite.sourceforge.net/}"
_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


# --------------------------------------------------------------------------- io


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None) -> int:
    """Fetch url to dest unless an intact copy exists; return bytes downloaded."""
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "RealtimeVAD-test-fetch/1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    got = sha256_file(tmp)
    if sha256 is not None and got != sha256:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {url}: {got} != {sha256}")
    tmp.replace(dest)
    return dest.stat().st_size


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())
    path.write_bytes(buf.getvalue())


# ------------------------------------------------------------------ annotations


class Annotations:
    """The parts of the AMI NXT annotations this script needs, read from the zip."""

    def __init__(self, zip_path: Path):
        self.z = zipfile.ZipFile(zip_path)
        names = set(self.z.namelist())
        self.names = names
        ont = ET.fromstring(self.z.read("ontologies/da-types.xml"))
        self.da_types = {e.get(NITE + "id"): e.get("name") for e in ont.iter("da-type")}
        sex = {}
        for p in ET.fromstring(self.z.read("corpusResources/participants.xml")).iter("participant"):
            sex[p.get(NITE + "id")] = p.get("sex")
        self.meetings = {}
        for m in ET.fromstring(self.z.read("corpusResources/meetings.xml")).iter("meeting"):
            obs = m.get("observation")
            spk = {}
            for s in m.iter("speaker"):
                g = s.get("global_name")
                # TS-site participants are missing from participants.xml; AMI participant
                # IDs start with their sex (F/M), e.g. FEE005, MTD021PM.
                gender = sex.get(g) or (g[0] if g[:1] in ("F", "M") else None)
                spk[s.get("nxt_agent")] = {"channel": int(s.get("channel")), "speaker": g, "gender": gender,
                                           "gender_source": "participants.xml" if g in sex else "participant id"}
            self.meetings[obs] = {"duration": float(m.get("duration") or 0), "speakers": spk}
        self._cache = {}

    def items(self, meeting: str, agent: str):
        """Tokens of one speaker in file order: dicts with id, kind, text, start, end."""
        key = ("w", meeting, agent)
        if key not in self._cache:
            root = ET.fromstring(self.z.read(f"words/{meeting}.{agent}.words.xml"))
            out = []
            for e in root:
                kind = e.tag.replace(NITE, "")
                st, en = e.get("starttime"), e.get("endtime")
                out.append({
                    "id": e.get(NITE + "id"),
                    "kind": kind,
                    "punc": e.get("punc") == "true",
                    "text": (e.text or "").strip(),
                    "start": float(st) if st not in (None, "") else None,
                    "end": float(en) if en not in (None, "") else None,
                    "vtype": e.get("type"),
                })
            self._cache[key] = out
        return self._cache[key]

    def acts(self, meeting: str, agent: str):
        """Dialogue acts of one speaker: dicts with id, type, first, last (item indices)."""
        key = ("da", meeting, agent)
        if key not in self._cache:
            name = f"dialogueActs/{meeting}.{agent}.dialog-act.xml"
            out = []
            if name in self.names:
                index = {it["id"]: i for i, it in enumerate(self.items(meeting, agent))}
                for d in ET.fromstring(self.z.read(name)).iter("dact"):
                    ptr = d.find(NITE + "pointer")
                    typ = None
                    if ptr is not None:
                        typ = self.da_types.get(_HREF.search(ptr.get("href")).group(1))
                    child = d.find(NITE + "child")
                    if child is None:
                        continue
                    a, b = _HREF.search(child.get("href")).groups()
                    out.append({"id": d.get(NITE + "id"), "type": typ, "first": index[a], "last": index[b or a]})
            self._cache[key] = out
        return self._cache[key]


def is_word(it) -> bool:
    return it["kind"] == "w" and not it["punc"]


def sounds(it) -> bool:
    """Anything audible the speaker produces: a word or a vocal sound (laugh, cough...)."""
    if it["start"] is None or it["end"] is None:
        return False
    if is_word(it):
        return True
    return it["kind"] == "vocalsound" and it["end"] > it["start"]


def token(text: str) -> str:
    t = re.sub(r"[^\w' -]", "", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def stable_key(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


# -------------------------------------------------------------------- candidates


def candidates(ann: Annotations, meeting: str):
    meta = ann.meetings[meeting]
    # all timed words of every participant, for others_talking
    words = {a: [it for it in ann.items(meeting, a) if is_word(it) and it["start"] is not None and it["end"] is not None]
             for a in meta["speakers"]}

    def others_talking(agent, t0, t1):
        return any(w["start"] < t1 and w["end"] > t0 for a, ws in words.items() if a != agent for w in ws)

    out = {"backchannel": [], "turn_start": [], "hesitation": []}
    for agent, spk in sorted(meta["speakers"].items()):
        items = ann.items(meeting, agent)
        acts = ann.acts(meeting, agent)
        audible = [(i, it) for i, it in enumerate(items) if sounds(it)]
        act_of = {}
        for d in acts:
            for i in range(d["first"], d["last"] + 1):
                act_of[i] = d

        def own_sound(t0, t1, skip=()):
            return any(it["start"] < t1 and it["end"] > t0 for i, it in audible if i not in skip)

        base = {"meeting": meeting, "speaker": spk["speaker"], "agent": agent,
                "headset": spk["channel"], "gender": spk["gender"],
                "gender_source": spk["gender_source"]}

        for d in acts:
            span = range(d["first"], d["last"] + 1)
            ws = [i for i in span if is_word(items[i])]
            if not ws or any(items[i]["start"] is None or items[i]["end"] is None for i in ws):
                continue
            t0, t1 = items[ws[0]]["start"], items[ws[-1]]["end"]
            text = " ".join(items[i]["text"] for i in ws)
            if d["type"] == BCK:
                if not (BC_MIN_S <= t1 - t0 <= BC_MAX_S) or t0 < CONTEXT_S + 0.1:
                    continue
                if any(items[i]["kind"] == "vocalsound" and sounds(items[i]) for i in span):
                    continue
                if own_sound(t0 - CONTEXT_S, t1 + CONTEXT_S, skip=set(ws)):
                    continue
                out["backchannel"].append({**base, "id": d["id"], "text": text, "token": token(text),
                                           "t0": t0, "t1": t1, "c0": t0 - CONTEXT_S, "c1": t1 + CONTEXT_S,
                                           "others_talking": others_talking(agent, t0, t1)})
            elif d["type"] not in TS_EXCLUDE and t1 - t0 >= TS_MIN_S and t0 >= CONTEXT_S + 0.1:
                if own_sound(t0 - CONTEXT_S, t0):
                    continue
                c0 = t0 - CONTEXT_S
                c1 = min(t1 + CONTEXT_S, c0 + TS_MAX_CLIP_S)
                out["turn_start"].append({**base, "id": d["id"], "text": text, "dialogue_act": d["type"],
                                          "t0": t0, "t1": min(t1, c1), "c0": c0, "c1": c1,
                                          "others_talking": others_talking(agent, t0, min(t1, c1))})

        # hesitations: maximal runs of the speaker's words with no gap over 1.5 s
        # and no laugh/cough in between
        seq = [(i, it) for i, it in audible]
        runs, cur = [], []
        for i, it in seq:
            if not is_word(it):
                if cur:
                    runs.append(cur)
                cur = []
                continue
            if cur and it["start"] - cur[-1][1]["end"] > HS_GAP[1]:
                runs.append(cur)
                cur = []
            cur.append((i, it))
        if cur:
            runs.append(cur)
        for run in runs:
            t0, t1 = run[0][1]["start"], run[-1][1]["end"]
            c0, c1 = t0 - CONTEXT_S, t1 + CONTEXT_S
            if not (HS_CLIP[0] <= c1 - c0 <= HS_CLIP[1]) or c0 < 0.1:
                continue
            if own_sound(c0, t0) or own_sound(t1, c1):
                continue
            if any(i not in act_of for i, _ in run):
                continue
            # every word index between the first and last must be timed
            if any(items[k]["kind"] == "w" and not items[k]["punc"] and items[k]["start"] is None
                   for k in range(run[0][0], run[-1][0] + 1)):
                continue
            qualifying, within_act, quiet_others = 0, True, True
            for (ia, a), (ib, b) in zip(run, run[1:]):
                g = b["start"] - a["end"]
                if not (HS_GAP[0] <= g <= HS_GAP[1]):
                    continue
                da, db = act_of[ia], act_of[ib]
                if BCK in (da["type"], db["type"]):
                    continue  # a backchannel next to the gap is not a turn going on
                qualifying += 1
                within_act &= da is db
                quiet_others &= not others_talking(agent, a["end"], b["start"])
            if not qualifying:
                continue
            ws = [it for _, it in run]
            gaps = [[a["end"] - c0, b["start"] - c0] for a, b in zip(ws, ws[1:]) if b["start"] - a["end"] >= HS_LISTED_GAP_S]
            out["hesitation"].append({**base, "id": run[0][1]["id"], "text": " ".join(w["text"] for w in ws),
                                      "t0": t0, "t1": t1, "c0": c0, "c1": c1,
                                      "others_talking": others_talking(agent, t0, t1),
                                      "dialogue_acts": sorted({act_of[i]["type"] or "none" for i, _ in run}),
                                      "within_act": within_act, "quiet_others": quiet_others,
                                      "gaps": gaps, "words": [[w["text"], w["start"] - c0, w["end"] - c0] for w in ws]})
    return out


def greedy(pool, n, key):
    """Pick n items one at a time, each time the one with the smallest key(item, counts)."""
    chosen, counts = [], collections.defaultdict(collections.Counter)
    pool = list(pool)
    for _ in range(n):
        live = [c for c in pool if key(c, counts) is not None]
        if not live:
            break
        best = min(live, key=lambda c: key(c, counts))
        pool.remove(best)
        chosen.append(best)
        for field in ("speaker", "token", "dialogue_act", "others_talking", "meeting"):
            counts[field][best.get(field)] += 1
    return chosen


def select(ann: Annotations):
    cands = {"backchannel": [], "turn_start": [], "hesitation": []}
    for m in MEETINGS:
        for k, v in candidates(ann, m).items():
            cands[k] += v

    # Backchannels in a pause (nobody else talking) are rare (about 1 in 12), so all of
    # them are taken first. Token types are then filled in proportion to the square
    # root of their frequency: the common ones dominate, the rare ones still occur.
    avail = collections.Counter(c["token"] for c in cands["backchannel"])

    def bc_key(c, n):
        if n["speaker"][c["speaker"]] >= BC_PER_SPEAKER:
            return None
        share = (n["token"][c["token"]] + 1) / avail[c["token"]] ** 0.5
        return (c["others_talking"], round(share, 9), n["speaker"][c["speaker"]], stable_key(c["id"]))

    bc = greedy(cands["backchannel"], BC_TARGET, bc_key)
    speakers = {c["speaker"] for c in bc}

    def ts_key(c, n):
        if c["speaker"] not in speakers:
            return None
        return (n["speaker"][c["speaker"]], n["dialogue_act"][c["dialogue_act"]], stable_key(c["id"]))

    ts = greedy(cands["turn_start"], TS_TARGET, ts_key)

    def hs_key(c, n):
        return (n["speaker"][c["speaker"]], not c["within_act"], not c["quiet_others"], stable_key(c["id"]))

    hs = greedy(cands["hesitation"], HS_TARGET, hs_key)

    clips = []
    for cat, picked in (("backchannel", bc), ("turn_start", ts), ("hesitation", hs)):
        for c in sorted(picked, key=lambda c: (c["meeting"], c["agent"], c["t0"])):
            s0, s1 = round(c["c0"] * SR), round(c["c1"] * SR)
            c0 = s0 / SR
            name = f"ami_{c['meeting']}_{c['agent']}_{s0 / SR:09.3f}".replace(".", "_") + ".wav"
            e = {
                "path": f"tests/audio/data/{cat}/{name}",
                "category": cat,
                "meeting": c["meeting"],
                "speaker": c["speaker"],
                "agent": c["agent"],
                "headset": c["headset"],
                "gender": c["gender"],
                "gender_source": c["gender_source"],
                "text": c["text"],
            }
            if cat == "backchannel":
                e["token"] = c["token"]
            if cat == "turn_start":
                e["dialogue_act"] = c["dialogue_act"]
            if cat == "hesitation":
                e["dialogue_acts"] = c["dialogue_acts"]
            e.update({
                "speech_start_s": round(c["t0"] - c0, 4),
                "speech_end_s": round(c["t1"] - c0, 4),
                "others_talking": c["others_talking"],
                "duration_s": round((s1 - s0) / SR, 4),
                "source_start_sample": s0,
                "source_end_sample": s1,
                "annotation_id": c["id"],
            })
            if cat == "hesitation":
                e["gaps"] = [[round(a + c["c0"] - c0, 4), round(b + c["c0"] - c0, 4)] for a, b in c["gaps"]]
                e["words"] = [[t, round(a + c["c0"] - c0, 4), round(b + c["c0"] - c0, 4)] for t, a, b in c["words"]]
            e["sha256"] = None
            e["source_url"] = AUDIO_URL.format(m=c["meeting"], ch=c["headset"])
            e["license"] = LICENSE
            clips.append(e)
    return clips


# -------------------------------------------------------------------- pipeline


SELECTION_FIELDS = ("path", "category", "meeting", "speaker", "agent", "headset", "gender", "gender_source", "text", "token",
                    "dialogue_act", "dialogue_acts", "speech_start_s", "speech_end_s", "others_talking",
                    "duration_s", "source_start_sample", "source_end_sample", "annotation_id", "gaps", "words",
                    "source_url")


def source_path(url: str) -> Path:
    return CACHE / url.rsplit("/", 1)[1]


def check_clip(e) -> str | None:
    p = HERE.parent.parent / e["path"]
    if not p.exists():
        return "missing"
    info = sf.info(str(p))
    if (info.samplerate, info.channels, info.subtype) != (SR, 1, "PCM_16"):
        return f"format {info.samplerate} Hz, {info.channels} ch, {info.subtype}"
    if abs(info.frames / SR - e["duration_s"]) > 1e-3:
        return f"duration {info.frames / SR} != {e['duration_s']}"
    if e["sha256"] and sha256_file(p) != e["sha256"]:
        return "sha256 mismatch"
    return None


def verify(manifest) -> int:
    bad = 0
    for e in manifest["clips"]:
        err = check_clip(e)
        if err:
            bad += 1
            print(f"  FAIL {e['path']}: {err}")
    n = len(manifest["clips"])
    print(f"verify: {n - bad}/{n} clips ok")
    return bad


def build(select_mode: bool) -> int:
    t_start = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
    downloaded = download(ANNOT_URL, CACHE / "ami_public_manual_1.6.2.zip", ANNOT_SHA256)
    ann = Annotations(CACHE / "ami_public_manual_1.6.2.zip")
    clips = select(ann)

    if not select_mode:
        if manifest is None:
            raise SystemExit("no manifest; run with --select first")
        old = manifest["clips"]
        same = len(old) == len(clips) and all(
            {k: a.get(k) for k in SELECTION_FIELDS} == {k: b.get(k) for k in SELECTION_FIELDS} for a, b in zip(old, clips))
        if not same:
            raise SystemExit("selection derived from the annotations differs from backchannels.json")
        clips = old
        sources = {s["url"]: s for s in manifest["sources"]}
    else:
        sources = {s["url"]: s for s in (manifest or {}).get("sources", [])}

    urls = sorted({e["source_url"] for e in clips})
    for url in urls:
        known = sources.get(url, {}).get("sha256")
        if not select_mode and not known:
            raise SystemExit(f"no sha256 for {url} in manifest")
        downloaded += download(url, source_path(url), known)
        if select_mode and not known:
            p = source_path(url)
            sources[url] = {"url": url, "sha256": sha256_file(p), "bytes": p.stat().st_size}

    by_src = collections.defaultdict(list)
    for e in clips:
        by_src[e["source_url"]].append(e)
    for url, es in by_src.items():
        with sf.SoundFile(str(source_path(url))) as f:
            if (f.samplerate, f.channels, f.subtype) != (SR, 1, "PCM_16"):
                raise SystemExit(f"unexpected format in {url}: {f.samplerate} {f.channels} {f.subtype}")
            for e in es:
                p = HERE.parent.parent / e["path"]
                if p.exists() and e["sha256"] and sha256_file(p) == e["sha256"]:
                    continue
                if e["source_end_sample"] > f.frames:
                    raise SystemExit(f"{e['path']}: beyond end of {url}")
                f.seek(e["source_start_sample"])
                pcm = f.read(e["source_end_sample"] - e["source_start_sample"], dtype="int16")
                write_wav(p, pcm)
                if select_mode:
                    e["sha256"] = sha256_file(p)

    if select_mode:
        manifest = {
            "description": "Backchannels, turn starts and hesitant turns cut from AMI Meeting Corpus individual "
                           "headsets; times in seconds relative to the clip start. Rebuild with "
                           "python tests/audio/fetch_backchannels.py",
            "corpus": CORPUS_URL,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "attribution": ATTRIBUTION,
            "annotations": {"url": ANNOT_URL, "sha256": ANNOT_SHA256},
            "sources": [sources[u] for u in urls],
            "clips": clips,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.name}: {len(clips)} clips")

    bad = verify(manifest)
    mb = sum((HERE.parent.parent / e["path"]).stat().st_size for e in clips if (HERE.parent.parent / e["path"]).exists()) / 1e6
    print(f"clips {mb:.1f} MB; downloaded {downloaded / 1e6:.1f} MB; {time.time() - t_start:.0f} s")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only check existing clips against the manifest")
    ap.add_argument("--select", action="store_true", help="maintainer: re-select clips and rewrite the manifest")
    a = ap.parse_args()
    if a.verify:
        if not MANIFEST.exists():
            raise SystemExit("no manifest")
        return 1 if verify(json.loads(MANIFEST.read_text(encoding="utf-8"))) else 0
    return build(a.select)


if __name__ == "__main__":
    sys.exit(main())
