# Licenses: word-timing benchmark audio

The clips of `benchmarks/word_timing.py` are not stored in this repository.
`python tests/audio/fetch_word_timing.py` downloads the sources into
`tests/audio/_cache/` and cuts 7,080 clips into `tests/audio/data/word_timing/`
(both git-ignored). Every download is checked against the sha256 values in
`tests/audio/word_timing.json`; the clip list with all word times and clip
checksums is checked against `clips_sha256` there.

All three sources are CC BY 4.0, checked on their own pages in November 2026.

## AMI Meeting Corpus

- **Page:** https://groups.inf.ed.ac.uk/ami/corpus/
- **License:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ ("The AMI corpus
  and its annotations are released under the Creative Commons Attribution 4.0 license
  agreement", https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- **Audio:** the four individual headset recordings and the first microphone of the
  table array (`Array1-01`) of 18 meetings, 16 kHz mono 16-bit,
  `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.<channel>.wav`
- **Meetings:** ES2002a, ES2004c, ES2006d, ES2008a, ES2010a, ES2013a, ES2015a, IS1001a,
  IS1004a, IS1006a, IS1008a, IS1009b, TS3003a, TS3005a, TS3006b, TS3008a, TS3010a,
  TS3012a: one meeting of each of 18 participant groups, 72 speakers, three rooms.
- **Word times:** AMI manual annotations v1.6.2 (manual transcript, forced-aligned word
  times), https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
- **Attribution:** Contains audio and annotations from the AMI Meeting Corpus
  (https://groups.inf.ed.ac.uk/ami/corpus/), licensed under CC BY 4.0. J. Carletta et al.,
  "The AMI Meeting Corpus: A Pre-Announcement", MLMI 2005, LNCS 3869, Springer, 2006.

## LibriSpeech

- **Page:** https://www.openslr.org/12 (states "License: CC BY 4.0")
- **License:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- **Audio:** `dev-clean`, `test-clean`, `dev-other`, `test-other` (146 speakers), FLAC,
  16 kHz mono, from the mirror `https://openslr.trmal.net/resources/12/<name>.tar.gz`.
  The archives' MD5 sums match `md5sum.txt` on that page.
- **Attribution:** V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR
  corpus based on public domain audio books", ICASSP 2015. Read from LibriVox
  recordings of public-domain books. CC BY 4.0.

## LibriSpeech Alignments

- **Page:** https://zenodo.org/records/2619474 (DOI 10.5281/zenodo.2619474, version 1.0,
  2019)
- **License:** CC BY 4.0 (stated on the record)
- **What it is:** word and phoneme alignments for all of LibriSpeech as Praat TextGrid
  files, made with the Montreal Forced Aligner and its pre-trained LibriSpeech acoustic
  model. Only the word tier of the four subsets above is used. The file's MD5 matches the
  one on the record (2bab567d0ace651a4ba254e813629f46).
- **Attribution:** Loren Lugosch (Mila), "LibriSpeech Alignments", Zenodo, 2019,
  https://doi.org/10.5281/zenodo.2619474. CC BY 4.0.

## Changes made

Clips are cut sample for sample from the sources (LibriSpeech: after decoding the FLAC
to 16-bit PCM). Nothing else is changed: no resampling, no level or gain change, no
filtering, no mixing. Word times are shifted to the clip start and rounded to 1 ms.

Which clip comes from where (source file, first and last sample, speaker, words with
times) is in `tests/audio/data/word_timing/clips.json`, written by the fetch script.
