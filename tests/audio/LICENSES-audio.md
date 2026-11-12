# Test audio: sources and licenses

The clips under `tests/audio/data/{speech,noise,music,nonverbal,whisper}/` are not in git.
`python tests/audio/fetch_audio.py` downloads and rebuilds them, and checks every download
and every clip against the sha256 values in `manifest.json`. The manifest lists each clip with
its source file, cut, license and attribution.

Rule used: only CC0 / public domain, CC BY, CC BY-SA, MIT, Apache or BSD, checked on the
dataset's own page (dates: November 2026). Nothing with NC, ND, "research only" or an unclear
license was used. The share-alike licenses (CC BY-SA) apply to these clips if they are passed
on: give the attribution below, name the license, and keep derived audio under the same license.

All clips are 16 kHz, mono, 16-bit PCM WAV. Processing: cut (`offset_s`/`length_s`), average
the channels to mono, resample to 16 kHz with `scipy.signal.resample_poly`, and quantise float
samples to 16 bit. Levels are never changed.

---

## FLEURS (category `speech`)

- **Page:** https://huggingface.co/datasets/google/fleurs (dataset card: `license: cc-by-4.0`;
  "All datasets are licensed under the Creative Commons license (CC-BY)")
- **License:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- **Attribution:** FLEURS: A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia,
  J. Riesa, C. Rivera, A. Bapna, "FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech", 2022, https://arxiv.org/abs/2205.12446, Google. CC BY 4.0.
- **Downloaded from:** `https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/<lang>/audio/{dev,train}.tar.gz`
  (pinned revision). Only the start of each archive is streamed, up to the last member needed.
- **What was taken:** 80 read sentences, 10 each in English (en_us), German (de_de),
  Spanish (es_419), French (fr_fr), Mandarin (cmn_hans_cn), Japanese (ja_jp), Hindi (hi_in) and
  Russian (ru_ru); 5 to 15 s each, whole utterances (not cut), with the FLEURS transcript and
  gender. FLEURS gives no speaker ids. In each language three clips are recordings of the same
  sentence, which FLEURS has read by three different native speakers ("three recordings by
  three different native speakers", paper), so each language has at least three speakers. The
  German dev and test sets contain only male speakers; the four German female clips come from
  the train set.

## MS-SNSD noise (category `noise`)

- **Page:** https://github.com/microsoft/MS-SNSD (README, "Dataset licenses": noise from
  Freesound, "Only files with CC0 licenses were selected", and from DEMAND under CC BY-SA 3.0;
  repository code under MIT)
- **Downloaded from:** `https://raw.githubusercontent.com/microsoft/MS-SNSD/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/noise_{train,test}/<file>.wav`
  (pinned commit), one file per download.
- **What was taken:** 18 clips of up to 40 s: babble (3), street (3), fan / air conditioner (3),
  keyboard typing (3), door shutting (3), kitchen (1), household: washing machine, vacuum
  cleaner (2). The 16 kHz mono 16-bit source samples are copied unchanged.

### DEMAND (the 300 s MS-SNSD files)

The seven MS-SNSD files that are exactly 300 s long are single channels of DEMAND recordings
(Cafe_1 = SCAFE, Restaurant_1 = PRESTO, Traffic_1 = STRAFFIC, Square_1 = SPSQUARE,
Station_1 = PSTATION, Kitchen_1 = DKITCHEN, Washing_1 = DWASHING; the mapping is inferred
from the names and the 300 s DEMAND length and is recorded as `demand_environment` in the
manifest).

- **Page:** https://zenodo.org/records/1227121 ("This work, the audio data and the document
  describing it, is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported
  License"; the Zenodo metadata field says CC BY 4.0, the stricter CC BY-SA 3.0 is applied)
- **License:** CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/
- **Attribution:** DEMAND: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments
  Multi-channel Acoustic Noise Database (DEMAND)", 2013, https://zenodo.org/records/1227121.
  CC BY-SA 3.0. 16 kHz single-channel copy from MS-SNSD (Microsoft).

### Freesound (the other MS-SNSD files)

- **License:** CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/
- **Attribution (courtesy, not required):** MS-SNSD, Microsoft: C. K. A. Reddy, E. Beyrami,
  J. Pool, R. Cutler, S. Srinivasan, J. Gehrke, "A Scalable Noisy Speech Dataset and Online
  Subjective Test Framework", Interspeech 2019; noise clips from Freesound.

## Wikimedia Commons (category `music`)

Each license was read on the file's own description page.

| clip | file page | license | attribution |
|---|---|---|---|
| `instrumental_piano` (50 s from 20 s) | https://commons.wikimedia.org/wiki/File:Frederic_Chopin_Piano_Sonata_No.2_in_B_flat_minor_Op35_-_IV_Finale_Presto.ogg | CC0 1.0 | Frédéric Chopin, Piano Sonata No. 2 Op. 35, IV Finale Presto; recording from Musopen |
| `instrumental_band` (50 s from 30 s) | https://commons.wikimedia.org/wiki/File:United_States_Navy_Band_-_Inno_e_Marcia_Pontificale.ogg | Public domain (U.S. federal government work) | "Inno e Marcia Pontificale" (Charles Gounod), United States Navy Band |
| `singing_solo` (50 s from 10 s) | https://commons.wikimedia.org/wiki/File:Twinkle_Twinkle_Little_Star_-_sung_with_full_lyrics.ogg | CC0 1.0 | "Twinkle Twinkle Little Star", sung by Dcoetzee (solo voice, unaccompanied) |
| `singing_choir` (50 s from 20 s) | https://commons.wikimedia.org/wiki/File:Shenandoah.ogg | Public domain (U.S. federal government work) | "Shenandoah" (traditional, arr. Floyd Werle and Michael Davis), Singing Sergeants, United States Air Force Band (a cappella choir) |

Downloaded from `https://upload.wikimedia.org/wikipedia/commons/...` (Ogg Vorbis, decoded by
libsndfile through soundfile, no ffmpeg).

## VocalSound (category `nonverbal`)

- **Page:** https://github.com/YuanGongND/vocalsound (README: "The VocalSound dataset is
  licensed under a Creative Commons Attribution-ShareAlike 4.0 International License")
- **License:** CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/
- **Attribution:** VocalSound: Y. Gong, J. Yu, J. Glass, "Vocalsound: A Dataset for Improving
  Human Vocal Sounds Recognition", ICASSP 2022, https://github.com/YuanGongND/vocalsound.
  CC BY-SA 4.0.
- **Downloaded from:** the 16 kHz WebDataset copy on Zenodo,
  https://doi.org/10.5281/zenodo.14650192, shards `wds-audio-valid-000000.tar` and
  `wds-audio-valid-000001.tar` (13 MB each). The official download is on Dropbox, which is not
  a stable direct host; the Zenodo copy's own metadata says CC BY 4.0, but it cannot relicense
  the data, so the original CC BY-SA 4.0 applies.
- **What was taken:** 50 whole clips, 10 each of cough, laughter, sigh (for breath/sigh),
  throat clearing and sneeze; 5 female and 5 male speakers per class (sneeze: 4 and 6), each
  speaker at most once per class. Selected automatically: at least 0.3 s quiet before and
  after the sound (frame energy more than 30 dB below the clip's maximum), and no sample at or
  near full scale. Each clip holds one vocalisation episode, which may contain several bursts
  (for example two coughs).

## Claris-Whispered-Hindi (category `whisper`)

- **Page:** https://huggingface.co/datasets/neilshahnms/Claris-Whispered-Hindi (dataset card:
  `license: mit`; the card has no further text and names no copyright holder)
- **License:** MIT, https://opensource.org/license/mit
- **Attribution:** Claris-Whispered-Hindi by neilshahnms (Hugging Face),
  https://huggingface.co/datasets/neilshahnms/Claris-Whispered-Hindi. MIT License.
- **Downloaded from:** `https://huggingface.co/datasets/neilshahnms/Claris-Whispered-Hindi/resolve/97dcd64e27c2a008bd832f6b80df168b994b50b1/<file>.wav`
  (pinned revision), one file per download; the transcripts come from the matching `.txt` files.
- **What was taken:** 10 whole whispered Hindi utterances (5 to 12 s) from 7 speakers (sp1, sp2,
  sp3, sp5, sp7, sp9, sp10), gender unknown. Recordings at 22.05 kHz or 44.1 kHz stereo are
  averaged to mono and resampled. Two speakers (sp4, sp8) were left out because their
  recordings looked voiced rather than whispered in a periodicity check.
