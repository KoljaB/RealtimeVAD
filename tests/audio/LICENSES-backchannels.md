# Licenses: AMI backchannel test audio

The clips listed in `tests/audio/backchannels.json` are not stored in this
repository. `tests/audio/fetch_backchannels.py` downloads the sources and cuts
them into the git-ignored `tests/audio/data/` (categories `backchannel`,
`turn_start`, `hesitation`).

## Source

- **Corpus:** AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ (stated on
  https://groups.inf.ed.ac.uk/ami/corpus/license.shtml for the corpus and its annotations)
- **Audio:** individual headset recordings, 16 kHz mono 16-bit,
  `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.Headset-<n>.wav`
- **Annotations:** AMI manual annotations v1.6.2 (words with forced-aligned
  times, dialogue acts, `meetings.xml`, `participants.xml`),
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip

## Meetings used

ES2004c, ES2006d, IS1009b, TS3006b: all four individual headsets of each
(16 speakers). Each clip comes from the headset of the speaker it is labelled
with. Gender comes from `participants.xml`; TS-site participants are missing
there, so for them it is taken from the first letter of the AMI participant ID
(`gender_source` in the manifest).

## Changes made

Clips are cut sample for sample from the original files. Nothing else is
changed: no resampling, no level or gain change, no filtering.

## Required attribution

> Contains audio and annotations from the AMI Meeting Corpus
> (https://groups.inf.ed.ac.uk/ami/corpus/), licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/); clips cut from the individual
> headset recordings, not otherwise modified.

Corpus reference: J. Carletta et al., "The AMI Meeting Corpus: A
Pre-Announcement", Machine Learning for Multimodal Interaction (MLMI 2005),
LNCS 3869, Springer, 2006, pp. 28-39.
