# RealtimeVAD 0.0.5 against word timings, 214 speakers

`benchmarks/word_timing.py` checks RealtimeVAD's `start`, `pause` and `end` against
word start and end times on 7,080 turn-like clips (10 hours) from 214 speakers. The
library is measured as it is (0.0.5, defaults); nothing in it was changed.

```bash
python tests/audio/fetch_word_timing.py        # 5.7 GB of downloads, checked by sha256; cuts 7,080 clips
python benchmarks/word_timing.py --examples 5  # 6 low-priority processes by default; 2 minutes with --processes 16
python benchmarks/word_timing.py --clip libri_clean/7021-85628-0003_1   # one clip, event by event
```

Everything from "Tables" on is the output of the second command. Sources and
licenses: `tests/audio/LICENSES-word-timing.md`.

## The short version

| Question | Answer |
|---|---|
| Does the speech-off lead hold for other speakers? | Yes, on close microphones. Median 86 – 102 ms before Silero's probability falls below 0.5, earlier in 77 – 86 % of clips. Half of the speakers lie between 70 and 108 ms. 15 of 191 speakers with enough clips have no lead: 5 with a DC offset in the recording, 7 with a room only 11 – 22 dB under their voice, 2 very quiet ones (−51 and −57 dBFS), 1 with breath or other voices right after the words. On a distant table microphone there is none |
| Does the `end` lead hold? | Only partly. `end` is earlier than Silero alone in 51 – 61 % of clips (README: 82 %), by a median 8 – 48 ms (README: 70 ms). It was later in 0 of 5,540 clips |
| Does a speech-off land inside a word? | Yes. The last `pause` before the `end` comes more than 100 ms before the last word's labelled end in 7.5 – 8.9 % of LibriSpeech clips, and in 6.2 – 7.6 % there is still audible sound of that word to come. Silero's own speech-off does that in 0.2 – 0.3 % |
| Does `end` come while the speaker goes on? | About as often as with Silero alone, plus 11 – 13 % more: 432 ends in pauses of roughly 250 to 350 ms that Silero alone rides through. In LibriSpeech no speech is lost by that. For quiet AMI speakers 83 clips (2.9 %) lose 100 ms or more of audible speech that Silero alone keeps |
| `pause` false alarms | 1.4 – 1.8 per second of speech against 0.27 – 0.48 for Silero (README: 1.5 against 0.14). 88 – 98 % of them lie inside a labelled word |
| Would a short confirmation delay help? | It trades one for the other, there is no cheap point. 30 ms removes 45 % of the false alarms (1.64 to 0.89 per second on LibriSpeech clean) and costs 36 ms of the 88 ms lead; 60 ms removes 71 % and leaves 16 ms |
| Start | `start` is decided a median 152 – 162 ms after the first word's labelled start, 64 ms after Silero reaches 0.5. LibriSpeech: every utterance gets a segment. AMI headsets (quiet, many one-word utterances): 180 of 2,819 get none, 94 with Silero at 0.5 |

## Data and what the labels are good for

| Condition | Clips | Speakers | What it is |
|---|---|---|---|
| LibriSpeech clean | 1,550 | 79 | read English, dev-clean and test-clean, close microphones, no noise suppression |
| LibriSpeech other | 2,270 | 64 | dev-other and test-other: noisier rooms, more accents |
| AMI headset, alone | 441 | 67 | meeting speech on the speaker's own headset, nobody else speaking in the clip |
| AMI headset, others talk | 2,378 | 71 | the same, with other participants audible somewhere in the clip |
| AMI distant mic, alone | 441 | 67 | the "alone" clips from one table microphone (Array1-01): −53 dBFS, 8 dB above the room |

A clip is one stretch of words with its real silence around it. AMI: a run of words
without a gap of 1 s, with up to 1 s on both sides. LibriSpeech: the words between two
pauses of 0.5 s or more. LibriSpeech files end about 0.5 s after the last word, so the
trailing silence there is short (0.47 s at least): 43 clips end before any detector
could close the segment and are left out of the end comparisons.

The labels are forced alignments, and they are not equally good:

- **LibriSpeech (Montreal Forced Aligner).** Against the nearest edge of the sound, the
  labelled end of the last word is off by a median 0 ms, −90 to +60 ms for 80 % of clips.
  Pauses shorter than about 0.3 s are sometimes not marked at all: the word before simply
  gets longer.
- **AMI.** Words of one run are labelled back to back. A pause inside a run is part of a
  word. The last word's labelled end is a median 50 – 70 ms before the sound stops.

How this was handled:

- "Inside a word" is only read on LibriSpeech, and always at several depths (50, 100,
  200 ms), next to the same count for Silero on the same frames.
- Every early speech-off is checked against the audio: how many 10 ms blocks between the
  `pause` and the labelled end are still audible (within 25 dB of the speaker's level and
  6 dB above the room).
- AMI is used for what its labels can carry: leads, ends, starts, lost speech. Lost speech
  counts only audible blocks inside labelled words.
- The 20 worst cases were looked at as level curves instead of trusting the label:
  - the 10 quiet-clock ends with the most labelled speech inside the 0.24 s wait (up to
    280 ms of it): all ten are label errors. The audio has a real pause of 280 to 330 ms
    at the room's level that the aligner did not mark ("hot | divide", "energy | shall
    remain", "well | how are you"). None is soft speech taken for silence.
  - the 10 speech-offs furthest before the labelled end (280 to 320 ms): all ten are
    real. The `pause` fires at a dip and the word goes on at 15 to 25 dB under the
    speaker's level: the /f/ of "knife", the last syllable of "Rodolfo", "Cornwall",
    "Arthur", "thyself", "unhappy", the release of "moved", "asked", "withhold", "right".
    In all ten the `end` still came after the word; the count had started again.

## Reading of the tables

**Lead.** The speech-off lead is a property of the method and carries over: 88 and 86 ms
on LibriSpeech, 102 and 100 ms on AMI headsets, per-speaker medians between 70 and 108 ms
for the middle half. Against `VADIterator`'s end it is 222 ms (LibriSpeech) and 240 ms
(AMI). `benchmarks/speech_off_lead.py` reports that end one frame too early (it lets the
100 ms run from the start of the first low frame; `silero_vad` counts from its end), so
the README's 206 ms is 32 ms too small.

**The `end` lead depends on how quiet the pause is.** The `end` comes from the quiet
clock in only 53 – 62 % of clips. The table "by how far the audio after the last word lies
under the quiet floor" shows why: with the pause 12 dB or more under the floor the clock
makes 91 % of the ends and leads by 58 ms (LibriSpeech) and 86 ms (AMI); within 3 dB of
the floor it makes 24 – 44 % and the median lead is 0. The speech-off lead is unaffected
until the pause is above the floor. The README's 70 ms comes from noise-suppressed
browser audio, where pauses are near digital silence. Raw microphones are not like that:
in 45 % of the LibriSpeech clips the pause is less than 3 dB under the floor or above it.

Where the `end` is left to Silero (LibriSpeech / AMI headsets, share of all clips):

| What happened after the last word | LibriSpeech | AMI headsets |
|---|---|---|
| quiet clock made the `end` | 54.5 % | 51.5 % |
| the count was broken twice or more by single louder 10 ms blocks | 23.2 % | 17.5 % |
| nothing counted as quiet (speech-off came from Silero) | 12.2 % | 18.7 % |
| the count was broken once | 7.5 % | 8.7 % |
| unbroken, but it started later than Silero's own count allows | 2.6 % | 3.6 % |

In the traces looked at, the threshold sits at the fixed floor (−54 dBFS) when the count
runs, also in rooms at −55 dBFS: the level it is capped by (`speech_db − 15 dB`) follows
the word's fade, 1 dB per frame, while Silero is still confident, so the room-noise margin
is gone by the time the word is over. Room noise then crosses the threshold every few
blocks and each crossing starts the 0.24 s again.

A DC offset is part of it: 218 LibriSpeech clips (5.8 %, 14 speakers) have an offset
above `quiet_rms`, and they have no lead at all. Without every clip with an offset the
`end` lead is still only 24 ms.

**Speech-off inside words.** By design most `pause` events are inside words (stop
closures, dips): 82 – 88 % of all pauses on LibriSpeech, 40 % of them followed by sound
again within 100 ms. What the README left open is the last one. It is early in 7.5 – 8.9 %
of turns by more than 100 ms, in 0.7 – 1.4 % by more than 200 ms, and almost always there
is still sound of the word to come. Whoever transcribes at `pause` gets a clipped last
word in about one turn of twelve; the `end` segment has it.

**Ends while the speaker goes on.** A third of the clips have an `end` before the last
word with any detector, because a 0.24 s wait ends at every breath pause. On the close
microphones RealtimeVAD makes 4,067 such ends, Silero alone 3,620 (all of which
RealtimeVAD makes too), `VADIterator` 4,968. After the 432 that are RealtimeVAD's own, Silero hears speech again a
median 70 ms later. By label and by level these are real pauses of a quarter second;
the quiet clock is not taking speech for silence there (Silero stayed above 0.5 during
the whole wait in 22 of 5,799 quiet-clock ends, 17 of them on AMI headsets with other
people talking). But it makes the effective end silence about 0.1 s shorter than the
same `end_s` gives with Silero alone.

**Confirmation delay.** After a false quiet-clock `pause` the audio stays quiet for 10 ms
in 24 – 29 % of cases, for 30 ms or less in 50 – 55 %, and the rest falls off slowly. The
histogram has no knee, and every 10 ms of delay costs 10 to 14 ms of lead.

**Start.** The 64 ms after Silero's 0.5 are the two-frame debounce at 0.9. The spread
across speakers is small (LibriSpeech medians 149 – 174 ms for the middle half). Missed
utterances are short and quiet: on AMI headsets 155 of the 180 are shorter than 0.8 s.
`normalize_to_dbfs=-30` brings back 22 of them. On the distant microphone 58 % of the
utterances get no segment (Silero at 0.5: 44 %) and the option does not help.

## Failure patterns, most important first

Example ids are in "Example clips" at the end; `--clip <id>` prints one event by event
next to the labels and the level.

1. **The early `end` is lost when the pause is not far under the quiet floor**
   (31 % of LibriSpeech clips, 26 % of AMI headset clips: count broken at least once).
   Single 10 ms blocks of room tone, a lip smack or the release of a final stop restart
   the 0.24 s; Silero's count finishes first. `libri_other/1686-142278-0049_0` (12 breaks,
   room −56 dBFS), `libri_clean/2094-142345-0054_0`, `ami_headset/TS3006b_D_025484`.
2. **Nothing counts as quiet although the room is far below the speaker** (12 % and 19 %).
   A loud speaker in a normal room: `ami_headset_xtalk/IS1009b_A_070764` (speech −17,
   room −54 dBFS, 36 dB apart, no early `pause`). A DC offset: speakers ls1988, ls4198,
   ls2830. A breath or another voice right after the word on AMI headsets.
3. **Speech-off inside a soft word ending** (6 – 8 % of turns by more than 100 ms).
   `libri_clean/7021-85628-0003_1` ("knife", 320 ms early), `libri_clean/5639-40744-0003_0`
   ("Rodolfo"), `libri_other/3660-172182-0030_0` ("Cornwall").
4. **An `end` in a quarter-second pause that Silero alone rides through** (432 ends, 11 –
   13 % more mid-utterance ends). `libri_clean/4970-29095-0004_0` (Silero hears speech
   6 ms after the `end`), `libri_other/1255-138279-0000_0`.
5. **Quiet speakers lose speech that Silero alone keeps** (83 AMI clips, 36 speakers,
   median level −50 dBFS; no LibriSpeech clip, and never the other way round).
   `ami_headset_xtalk/TS3003a_A_059254`: speech at −62 dBFS, room at −75 dBFS, the quiet
   clock ends the segment at a dip of 10 to 15 dB and the next 0.4 s are in no segment.
   The threshold there is 15 dB under the latest speech level, not the 20 dB under the
   speaker that SPEC.md states.
6. **Short quiet utterances get no segment although Silero at 0.5 hears them** (86 more
   than Silero at 0.5 on AMI headsets). One-word backchannels whose probability peaks at
   0.86 – 0.89: `ami_headset_xtalk/IS1009b_B_109888`, `ami_headset_xtalk/IS1001a_B_027271`.
7. **Silero's probabilities differ with and without the quiet clock** in 5 clips; SPEC.md
   says they do not depend on it. `libri_other/3528-168669-0082_0` (75 frames).
8. **Distant microphone at −53 dBFS:** no lead, more than half of the utterances missed.
   This is mostly Silero at that level; listed last because nothing in the quiet clock
   causes it.

## What the existing settings do to it

Every third clip, close microphones only (`--limit 3 --options '{...}'`). No setting
gives the `end` lead back for free; each one that helps also moves speech-off further
into words and adds ends that Silero alone does not make.

| Setting | `end` by the quiet clock: LibriSpeech / AMI | `end` lead, median ms | speech-off lead, median ms | false `pause` per s | own mid-utterance ends | speech-off > 100 ms early with sound to come |
|---|---|---|---|---|---|---|
| defaults | 54 % / 52 % | 14 / 28 | 86 / 100 | 1.54 / 1.83 | 61 / 79 | 6.4 % / 11.3 % |
| `noisy_extra_s=0` | 58 % / 53 % | 42 / 32 | 86 / 100 | 1.54 / 1.83 | 87 / 82 | 6.4 % / 11.0 % |
| `quiet_rms=0.004` | 61 % / 53 % | 42 / 32 | 94 / 102 | 1.67 / 1.87 | 107 / 81 | 8.5 % / 11.6 % |
| `speech_margin_db=8` | 72 % / 66 % | 43 / 64 | 104 / 110 | 1.83 / 1.91 | 136 / 108 | 13.4 % / 13.9 % |
| `noise_margin_db=6` | 49 % / 47 % | 0 / 0 | 78 / 94 | 1.26 / 1.71 | 49 / 67 | 3.0 % / 9.9 % |
| `rearm_s=0.03` | 54 % / 52 % | 16 / 30 | 78 / 92 | 1.81 / 2.15 | 61 / 79 | 2.8 % / 8.6 % |

## Limits of this benchmark

- English only. Read speech and meeting speech, no phone audio, no noise-suppressed
  capture (the README's own condition is not in here).
- Every clip starts a fresh detector with 0.25 to 1 s of room before the speech. Level
  learning across turns, echo handling and long streams are not exercised.
- AMI "alone" has only 441 clips (6.6 per speaker): in meetings somebody else almost
  always speaks within a second. Per-speaker spreads for AMI therefore come mostly from
  the clips with other people audible.
- "Audible" and "under the floor" use levels measured here on the labelled words, not the
  library's internal ones.

# Tables

### Data

| Condition | clips | speakers | labelled speech | speech level: median (p10 – p90 of clips) | room level | speech above room |
|---|---|---|---|---|---|---|
| LibriSpeech clean | 1550 | 79 | 2.0 h | -29 dBFS (-36 – -24) | -59 dBFS | 29 dB |
| LibriSpeech other | 2270 | 64 | 2.3 h | -29 dBFS (-35 – -24) | -63 dBFS | 33 dB |
| AMI headset, alone | 441 | 67 | 0.4 h | -46 dBFS (-60 – -33) | -76 dBFS | 29 dB |
| AMI headset, others talk | 2378 | 71 | 1.9 h | -44 dBFS (-58 – -28) | -75 dBFS | 28 dB |
| AMI distant mic, alone | 441 | 67 | 0.4 h | -53 dBFS (-58 – -40) | -62 dBFS | 8 dB |

Checks: the delay replay reproduces the library's `pause` events in 7080 of 7080 clips. Silero's probabilities differ between RealtimeVAD and Silero alone in 5 clips (ES2002a_D_057022, ES2008a_C_100135, IS1008a_D_025707, TS3005a_A_092664, 3528-168669-0082_0).

### Labels against the audio

| Condition | last word: labelled end minus end of the sound, median (p10 – p90) ms | label > 50 ms after the sound | label > 50 ms before it | first word: start of the sound minus labelled start, ms |
|---|---|---|---|---|
| LibriSpeech clean | 0 (-90 – 60) | 14.2 % | 14.8 % | 40 (-40 – 60) |
| LibriSpeech other | 0 (-80 – 60) | 13.3 % | 16.1 % | 40 (-30 – 60) |
| AMI headset, alone | -50 (-330 – 0) | 2.3 % | 51.5 % | 0 (-160 – 30) |
| AMI headset, others talk | -70 (-390 – 0) | 3.8 % | 54.3 % | -10 (-273 – 30) |
| AMI distant mic, alone | -20 (-221 – 90) | 18.9 % | 34.8 % | 0 (-170 – 60) |

### Where `pause` falls

RealtimeVAD `pause`:

| Condition | events | per s of speech | inside a labelled word | … ≥ 50 ms from both its edges | … ms before the word's end: median (p25 – p75) | … and quiet for < 100 ms | in a labelled gap | after the last word |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 13467 | 1.86 | 87.8 % | 44.6 % | 180 (90 – 330) | 40.2 % | 6.5 % | 5.7 % |
| LibriSpeech other | 14077 | 1.71 | 81.7 % | 41.4 % | 170 (90 – 310) | 36.3 % | 8.8 % | 9.4 % |
| AMI headset, alone | 2887 | 1.98 | 90.9 % | 59.5 % | 250 (120 – 500) | 41.9 % | 0.4 % | 8.7 % |
| AMI headset, others talk | 14343 | 2.10 | 86.0 % | 54.4 % | 240 (120 – 460) | 37.8 % | 0.4 % | 13.6 % |
| AMI distant mic, alone | 842 | 0.58 | 80.0 % | 58.8 % | 306 (141 – 533) | 20.5 % | 0.6 % | 19.4 % |

Silero p < 0.5:

| Condition | events | per s of speech | inside a labelled word | … ≥ 50 ms from both its edges | … ms before the word's end: median (p25 – p75) | … and quiet for < 100 ms | in a labelled gap | after the last word |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 3549 | 0.49 | 19.0 % | 7.2 % | 178 (104 – 294) | 5.6 % | 37.0 % | 44.0 % |
| LibriSpeech other | 4701 | 0.57 | 17.1 % | 7.0 % | 188 (111 – 333) | 5.1 % | 34.1 % | 48.7 % |
| AMI headset, alone | 1123 | 0.77 | 67.1 % | 54.7 % | 322 (150 – 575) | 8.0 % | 1.0 % | 31.9 % |
| AMI headset, others talk | 5580 | 0.82 | 60.0 % | 48.0 % | 302 (158 – 561) | 6.6 % | 1.1 % | 38.9 % |
| AMI distant mic, alone | 1105 | 0.76 | 78.7 % | 58.3 % | 275 (146 – 526) | 12.9 % | 0.5 % | 20.7 % |

### Speech-off against the end of the last word

| Condition | speech-off of | clips | relative to the labelled end: median (p10 – p90) ms | > 50 ms before it | > 100 ms before it | … with ≥ 30 ms of audible sound still to come | … with ≥ 100 ms | > 200 ms before it |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | RealtimeVAD | 1519 | -10 (-90 – 124) | 23.2 % | 8.9 % | 7.6 % | 2.3 % | 1.4 % |
| LibriSpeech clean | Silero p < 0.5 | 1504 | 72 (24 – 144) | 0.9 % | 0.3 % | 0.2 % | 0.1 % | 0.0 % |
| LibriSpeech other | RealtimeVAD | 2239 | -10 (-80 – 150) | 18.0 % | 7.5 % | 6.2 % | 2.2 % | 0.7 % |
| LibriSpeech other | Silero p < 0.5 | 2215 | 82 (28 – 158) | 0.7 % | 0.2 % | 0.2 % | 0.0 % | 0.0 % |
| AMI headset, alone | RealtimeVAD | 368 | 0 (-110 – 160) | 19.0 % | 10.9 % | 10.3 % | 6.2 % | 3.5 % |
| AMI headset, alone | Silero p < 0.5 | 363 | 108 (-10 – 238) | 7.2 % | 3.6 % | 3.3 % | 1.4 % | 0.6 % |
| AMI headset, others talk | RealtimeVAD | 1789 | 10 (-120 – 335) | 21.1 % | 12.1 % | 11.1 % | 8.3 % | 3.2 % |
| AMI headset, others talk | Silero p < 0.5 | 1615 | 120 (-28 – 594) | 8.0 % | 5.3 % | 4.8 % | 3.9 % | 0.6 % |
| AMI distant mic, alone | RealtimeVAD | 154 | 114 (-22 – 268) | 6.5 % | 3.2 % | 2.6 % | 0.6 % | 0.6 % |
| AMI distant mic, alone | Silero p < 0.5 | 146 | 146 (-6 – 863) | 6.8 % | 2.7 % | 2.7 % | 2.1 % | 0.0 % |

### Lead over Silero

| Condition | clips | speech-off lead over p < 0.5: median (p25 – p75) ms | earlier in | over p < 0.35, median ms | speakers with ≥ 8 clips | their medians: p25 / median / p75 ms | speakers with no lead |
|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 1504 | 88 (62 – 116) | 83.6 % | 98 | 57 | 74 / 90 / 98 (min 0, max 126) | 5 |
| LibriSpeech other | 2215 | 86 (48 – 110) | 82.2 % | 94 | 53 | 70 / 86 / 96 (min 0, max 146) | 5 |
| AMI headset, alone | 363 | 102 (63 – 130) | 85.7 % | 116 | 19 | 89 / 100 / 110 (min 69, max 131) | 0 |
| AMI headset, others talk | 1615 | 100 (15 – 132) | 76.5 % | 118 | 62 | 79 / 100 / 108 (min 0, max 168) | 5 |
| AMI distant mic, alone | 146 | 0 (-32 – 432) | 29.5 % | 32 | 4 | 0 / 0 / 88 (min 0, max 352) | 3 |

| Condition | clips where both closed the same segments | last `end` made by the quiet clock | `end` earlier than Silero alone: median (p25 – p75) ms | earlier in | later in (clips) | speakers with ≥ 8 clips | their medians: p25 / median / p75 ms | speakers with median 0 | `end` after the labelled end of the last word, median ms | VADIterator's end after RealtimeVAD's `end`, median ms |
|---|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 1439 | 52.7 % | 8 (0 – 52) | 51.9 % | 0 | 55 | 0 / 28 / 46 (min 0, max 78) | 20 | 270 | -64 |
| LibriSpeech other | 2133 | 55.7 % | 18 (0 – 60) | 54.9 % | 0 | 52 | 0 / 29 / 56 (min 0, max 94) | 19 | 270 | -64 |
| AMI headset, alone | 331 | 62.0 % | 48 (0 – 92) | 61.3 % | 0 | 16 | 28 / 63 / 81 (min 0, max 90) | 4 | 260 | -26 |
| AMI headset, others talk | 1487 | 49.4 % | 12 (0 – 88) | 50.8 % | 0 | 56 | 0 / 13 / 62 (min 0, max 96) | 27 | 270 | -32 |
| AMI distant mic, alone | 150 | 0.0 % | 0 (0 – 0) | 0.0 % | 0 | 4 | 0 / 0 / 0 (min 0, max 0) | 4 | 340 | -64 |

The `end` lead by how far the audio after the last word (50 to 300 ms after its labelled end) lies under the quiet floor (−54 dBFS, or 20 dB under the speaker when that is lower):

LibriSpeech (both):

| Audio after the last word | clips | share | `end` made by the quiet clock | `end` lead, median ms | speech-off lead, median ms | breaks of the quiet count, median |
|---|---|---|---|---|---|---|
| ≥ 12 dB under | 967 | 25.7 % | 91.4 % | 58 | 87 | 0 |
| 6 – 12 dB under | 724 | 19.3 % | 84.9 % | 42 | 94 | 0 |
| 3 – 6 dB under | 360 | 9.6 % | 63.6 % | 26 | 100 | 1 |
| 0 – 3 dB under | 386 | 10.3 % | 43.5 % | 0 | 96 | 1 |
| 0 – 3 dB over | 402 | 10.7 % | 23.9 % | 0 | 94 | 2 |
| 3 – 6 dB over | 334 | 8.9 % | 11.1 % | 0 | 63 | 2 |
| > 6 dB over | 585 | 15.6 % | 3.2 % | 0 | 0 | 1 |

AMI headset (both):

| Audio after the last word | clips | share | `end` made by the quiet clock | `end` lead, median ms | speech-off lead, median ms | breaks of the quiet count, median |
|---|---|---|---|---|---|---|
| ≥ 12 dB under | 351 | 16.3 % | 88.6 % | 86 | 106 | 0 |
| 6 – 12 dB under | 385 | 17.8 % | 83.1 % | 84 | 110 | 0 |
| 3 – 6 dB under | 214 | 9.9 % | 65.0 % | 66 | 108 | 0 |
| 0 – 3 dB under | 199 | 9.2 % | 43.7 % | 0 | 106 | 1 |
| 0 – 3 dB over | 280 | 13.0 % | 37.9 % | 0 | 90 | 1 |
| 3 – 6 dB over | 200 | 9.3 % | 28.5 % | 0 | 80 | 1 |
| > 6 dB over | 528 | 24.5 % | 17.2 % | 0 | 30 | 1 |

LibriSpeech by the DC offset of the clip (mean sample value):

| DC offset | clips | speakers | room level as fed, dBFS | room level without the offset, dBFS | `end` made by the quiet clock | speech-off lead, median ms | `end` lead, median ms |
|---|---|---|---|---|---|---|---|
| under 0.0003 (−70 dBFS) | 3256 | 134 | -62 | -62 | 58.7 % | 88 | 24 |
| 0.0003 – 0.001 | 210 | 27 | -56 | -57 | 44.3 % | 76 | 0 |
| 0.001 – 0.002 | 74 | 7 | -61 | -65 | 36.5 % | 100 | 0 |
| 0.002 – 0.004 (quiet_rms is 0.002) | 123 | 10 | -50 | -60 | 12.2 % | 13 | 0 |
| over 0.004 (−48 dBFS) | 95 | 8 | -38 | -56 | 3.2 % | 0 | 0 |

### `end` while the speaker goes on

| Condition | clips | mid-utterance ends (clips with one): RealtimeVAD | Silero alone | VADIterator | RealtimeVAD ends that Silero alone does not make | … Silero hears speech again after, median ms | … within 100 ms | quiet-clock ends during which Silero never fell below 0.5 | clips with ≥ 100 ms of audible labelled speech in no segment: RealtimeVAD | Silero alone | VADIterator |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 1550 | 850 (35.8 %) | 761 (33.1 %) | 1068 (42.1 %) | 89 | 68 | 61 | 4 of 1332 | 1 | 1 | 52 |
| LibriSpeech other | 2270 | 1088 (33.0 %) | 984 (31.0 %) | 1358 (38.5 %) | 104 | 72 | 91 | 0 of 1929 | 2 | 2 | 90 |
| AMI headset, alone | 441 | 386 (37.9 %) | 330 (35.6 %) | 442 (41.7 %) | 53 | 70 | 39 | 1 of 488 | 121 | 117 | 200 |
| AMI headset, others talk | 2378 | 1743 (33.1 %) | 1545 (30.9 %) | 2100 (39.4 %) | 186 | 72 | 138 | 17 of 2050 | 456 | 412 | 815 |
| AMI distant mic, alone | 441 | 270 (22.7 %) | 270 (22.7 %) | 420 (30.2 %) | 0 | – | 0 | 0 of 0 | 341 | 341 | 371 |

### `pause` false alarms and a confirmation delay

| Condition | false `pause` per s of speech | Silero p < 0.5 followed by speech, per s | clips without a false `pause` | per speaker: p25 / median / p75 | from the quiet clock | inside a labelled word |
|---|---|---|---|---|---|---|
| LibriSpeech clean | 1.64 | 0.27 | 6.5 % | 1.29 / 1.63 / 1.86 (max 2.36) | 97.8 % | 91.9 % |
| LibriSpeech other | 1.42 | 0.29 | 10.5 % | 1.00 / 1.39 / 1.68 (max 2.78) | 97.4 % | 88.1 % |
| AMI headset, alone | 1.75 | 0.48 | 19.0 % | 1.49 / 1.72 / 1.95 (max 2.36) | 95.9 % | 98.5 % |
| AMI headset, others talk | 1.81 | 0.46 | 23.9 % | 1.57 / 1.78 / 1.95 (max 2.67) | 95.9 % | 96.8 % |
| AMI distant mic, alone | 0.60 | 0.64 | 20.1 % | 0.60 / 0.62 / 0.64 (max 0.64) | 29.0 % | 96.2 % |

How long the audio stayed quiet after a false quiet-clock `pause` (10 ms blocks until a louder one; the last column ran on into an `end`):

| Condition | false pauses | 10 ms | 20 | 30 | 40 | 50 | 60 | 70 – 80 | 90 – 100 | 110 – 150 | 160 – 240 | longer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 11264 | 28.6 % | 15.4 % | 11.5 % | 10.3 % | 8.4 % | 6.1 % | 7.4 % | 3.5 % | 3.2 % | 2.8 % | 2.6 % |
| LibriSpeech other | 11236 | 29.3 % | 15.2 % | 10.7 % | 8.8 % | 8.0 % | 5.8 % | 6.8 % | 4.1 % | 3.7 % | 5.5 % | 2.3 % |
| AMI headset, alone | 2288 | 24.0 % | 14.3 % | 12.3 % | 10.5 % | 7.0 % | 5.2 % | 7.0 % | 4.1 % | 3.9 % | 10.2 % | 1.4 % |
| AMI headset, others talk | 9962 | 23.5 % | 13.3 % | 12.4 % | 10.1 % | 7.6 % | 5.4 % | 7.8 % | 4.3 % | 4.4 % | 10.3 % | 0.9 % |
| AMI distant mic, alone | 168 | 63.1 % | 10.7 % | 16.1 % | 7.7 % | 0.6 % | 0.6 % | 0.6 % | 0.0 % | 0.6 % | 0.0 % | 0.0 % |

A `pause` that is reported only after the quiet has lasted this much longer (replayed, library unchanged):

| Delay | LibriSpeech clean: false per s | lead ms (earlier in) | LibriSpeech other: false per s | lead ms (earlier in) | AMI headset, alone: false per s | lead ms (earlier in) | AMI headset, others talk: false per s | lead ms (earlier in) | AMI distant mic, alone: false per s | lead ms (earlier in) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 1.64 | 88 (83.6 %) | 1.42 | 86 (82.2 %) | 1.75 | 102 (85.7 %) | 1.81 | 100 (76.5 %) | 0.60 | 0 (29.5 %) |
| 10 ms | 1.31 | 74 (80.5 %) | 1.13 | 72 (80.1 %) | 1.47 | 90 (83.5 %) | 1.52 | 88 (74.9 %) | 0.53 | 0 (28.8 %) |
| 20 ms | 1.07 | 62 (77.0 %) | 0.93 | 60 (77.1 %) | 1.25 | 78 (80.4 %) | 1.31 | 76 (73.1 %) | 0.51 | 0 (28.1 %) |
| 30 ms | 0.89 | 52 (73.9 %) | 0.78 | 48 (72.4 %) | 1.05 | 68 (77.7 %) | 1.10 | 66 (71.1 %) | 0.48 | 0 (27.4 %) |
| 40 ms | 0.71 | 40 (70.8 %) | 0.65 | 36 (67.9 %) | 0.88 | 58 (76.0 %) | 0.93 | 54 (68.8 %) | 0.46 | 0 (27.4 %) |
| 50 ms | 0.57 | 28 (66.7 %) | 0.53 | 24 (62.4 %) | 0.76 | 44 (72.7 %) | 0.79 | 42 (66.4 %) | 0.46 | 0 (26.7 %) |
| 60 ms | 0.47 | 16 (62.3 %) | 0.45 | 12 (57.0 %) | 0.67 | 32 (68.9 %) | 0.69 | 32 (63.5 %) | 0.46 | 0 (26.7 %) |
| 80 ms | 0.34 | 0 (43.5 %) | 0.35 | 0 (39.9 %) | 0.55 | 10 (55.9 %) | 0.55 | 10 (54.8 %) | 0.46 | 0 (26.7 %) |
| 100 ms | 0.29 | 0 (22.1 %) | 0.30 | 0 (19.3 %) | 0.49 | 0 (38.6 %) | 0.48 | 0 (39.6 %) | 0.46 | 0 (26.7 %) |

### Start

| Condition | clips | `start` decided after the first word's labelled start: median (p25 – p75, p90) ms | Silero p ≥ 0.5 (VADIterator's start) | RealtimeVAD after Silero, median ms | per-speaker medians: p25 / median / p75 ms | no segment: RealtimeVAD | Silero alone | Silero p ≥ 0.5 | RealtimeVAD with normalize_to_dbfs=-30 |
|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 1550 | 162 (140 – 186, p90 222) | 98 (82 – 120, p90 154) | 64 (p90 96) | 149 / 159 / 168 (min 132, max 192) | 0 (0.0 %) | 0 | 0 (0.0 %) | – |
| LibriSpeech other | 2270 | 162 (140 – 188, p90 226) | 98 (80 – 120, p90 152) | 64 (p90 96) | 150 / 162 / 174 (min 118, max 232) | 0 (0.0 %) | 0 | 0 (0.0 %) | – |
| AMI headset, alone | 441 | 154 (138 – 216, p90 376) | 88 (56 – 120, p90 216) | 64 (p90 96) | 152 / 167 / 184 (min 120, max 312) | 41 (9.3 %) | 41 | 26 (5.9 %) | 36 (8.2 %) |
| AMI headset, others talk | 2378 | 152 (88 – 184, p90 248) | 56 (56 – 88, p90 152) | 64 (p90 192) | 120 / 152 / 152 (min -680, max 536) | 139 (5.8 %) | 139 | 68 (2.9 %) | 122 (5.1 %) |
| AMI distant mic, alone | 441 | 409 (248 – 984, p90 2200) | 312 (152 – 712, p90 1765) | 128 (p90 790) | 344 / 568 / 764 (min 184, max 1576) | 257 (58.3 %) | 257 | 194 (44.0 %) | 254 (57.6 %) |

### Speakers with the least lead

| Corpus | speaker | clips | speech-off lead, median ms | `end` lead, median ms | `end` by the quiet clock | speech dBFS | room dBFS | after the last word dBFS | dB under the quiet floor |
|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech | ls1221 (F) | 12 | 0 | 0 | 0.0 % | -32 | -49 | -48 | -6 |
| LibriSpeech | ls1701 (M) | 62 | 0 | 0 | 11.3 % | -32 | -50 | -49 | -5 |
| LibriSpeech | ls1988 (F) | 22 | 0 | 0 | 36.4 % | -31 | -39 | -38 | -17 |
| LibriSpeech | ls1998 (F) | 47 | 0 | 0 | 0.0 % | -31 | -46 | -45 | -10 |
| LibriSpeech | ls2609 (M) | 46 | 0 | 0 | 6.5 % | -26 | -43 | -39 | -15 |
| LibriSpeech | ls2830 (M) | 14 | 0 | 0 | 7.1 % | -27 | -42 | -39 | -15 |
| LibriSpeech | all 109 speakers | 3719 | 86 | 16 | 54.9 % | -29 | -61 | -59 | 5 |
| AMI headset | FEE050 (F) | 10 | 0 | 0 | 20.0 % | -57 | -84 | -79 | 3 |
| AMI headset | FEE060 (F) | 8 | 0 | 0 | 12.5 % | -50 | -79 | -76 | 3 |
| AMI headset | FIO084 (F) | 10 | 0 | 0 | 10.0 % | -37 | -56 | -53 | -8 |
| AMI headset | MIO012 (M) | 17 | 0 | 0 | 0.0 % | -33 | -53 | -50 | -5 |
| AMI headset | FEE029 (F) | 26 | 4 | 0 | 26.9 % | -38 | -78 | -44 | -16 |
| AMI headset | FEE032 (F) | 20 | 5 | 0 | 15.0 % | -40 | -77 | -49 | -10 |
| AMI headset | all 64 speakers | 1978 | 100 | 28 | 53.5 % | -45 | -76 | -70 | 2 |

### Example clips

The quiet count broken again and again by the room (speech-off early, `end` left to Silero):

- `libri_other/1686-142278-0049_0` 12 breaks, room -56 dBFS, speech -30 dBFS
- `libri_clean/3853-163249-0016_0` 12 breaks, room -45 dBFS, speech -26 dBFS
- `ami_headset/TS3006b_D_025484` 12 breaks, room -69 dBFS, speech -37 dBFS
- `libri_other/1686-142278-0042_0` 11 breaks, room -55 dBFS, speech -33 dBFS
- `libri_clean/2094-142345-0054_0` 11 breaks, room -48 dBFS, speech -25 dBFS
- `libri_clean/422-122949-0009_0` 11 breaks, room -50 dBFS, speech -21 dBFS

Nothing counts as quiet after the last word (no early `pause`, no early `end`); the quietest rooms first:

- `ami_headset_xtalk/IS1009b_A_070764` after the word -53 dBFS, speech -17 dBFS, room -54 dBFS
- `ami_headset_xtalk/IS1009b_D_107283` after the word -52 dBFS, speech -21 dBFS, room -53 dBFS
- `libri_other/4570-24733-0002_0` after the word -51 dBFS, speech -21 dBFS, room -89 dBFS
- `libri_other/4570-24733-0005_4` after the word -48 dBFS, speech -20 dBFS, room -90 dBFS
- `libri_clean/6295-64301-0015_0` after the word -53 dBFS, speech -26 dBFS, room -63 dBFS
- `ami_headset_xtalk/IS1009b_A_129207` after the word -48 dBFS, speech -21 dBFS, room -53 dBFS

Speech-off inside a soft word ending (LibriSpeech):

- `libri_clean/7021-85628-0003_1` `pause` 320 ms before the labelled end, 120 ms of audible sound after it
- `libri_clean/5639-40744-0003_0` `pause` 300 ms before the labelled end, 120 ms of audible sound after it
- `libri_clean/5683-32879-0021_0` `pause` 300 ms before the labelled end, 220 ms of audible sound after it
- `libri_other/8288-274162-0035_0` `pause` 290 ms before the labelled end, 180 ms of audible sound after it
- `libri_clean/908-31957-0023_0` `pause` 280 ms before the labelled end, 80 ms of audible sound after it
- `libri_other/3660-172182-0030_0` `pause` 280 ms before the labelled end, 120 ms of audible sound after it

`end` in a short pause that Silero alone rides through (LibriSpeech):

- `libri_clean/4970-29095-0004_0` `end` at 5.69 s, Silero hears speech again 6 ms later
- `libri_other/1255-138279-0000_0` `end` at 3.60 s, Silero hears speech again 16 ms later
- `libri_other/8280-266249-0038_0` `end` at 3.15 s, Silero hears speech again 18 ms later
- `libri_other/8188-269290-0045_0` `end` at 1.48 s, Silero hears speech again 24 ms later
- `libri_clean/2035-147960-0010_0` `end` at 4.07 s, Silero hears speech again 26 ms later
- `libri_other/5849-50873-0000_0` `end` at 3.59 s, Silero hears speech again 26 ms later

Quiet-clock `end` while Silero stayed above 0.5 throughout:

- `ami_headset_xtalk/ES2004c_C_045166` `end` at 4.01 s, Silero's lowest probability in the wait 0.79
- `ami_headset_xtalk/ES2004c_C_112723` `end` at 5.79 s, Silero's lowest probability in the wait 0.72
- `ami_headset_xtalk/TS3012a_C_005329` `end` at 13.36 s, Silero's lowest probability in the wait 0.71
- `ami_headset_xtalk/ES2004c_C_040983` `end` at 7.28 s, Silero's lowest probability in the wait 0.69
- `ami_headset_xtalk/ES2004c_D_208130` `end` at 5.42 s, Silero's lowest probability in the wait 0.64
- `ami_headset/ES2008a_A_004513` `end` at 11.13 s, Silero's lowest probability in the wait 0.60

Audible speech in no segment that Silero alone keeps:

- `ami_headset_xtalk/ES2008a_B_062921` 540 ms against 80 ms, speech -44 dBFS, room -82 dBFS
- `ami_headset_xtalk/ES2010a_A_055686` 1580 ms against 1120 ms, speech -48 dBFS, room -78 dBFS
- `ami_headset_xtalk/ES2004c_C_100391` 770 ms against 380 ms, speech -52 dBFS, room -86 dBFS
- `ami_headset_xtalk/TS3003a_A_059254` 560 ms against 180 ms, speech -62 dBFS, room -75 dBFS
- `ami_headset_xtalk/ES2010a_A_048441` 920 ms against 600 ms, speech -51 dBFS, room -80 dBFS
- `ami_headset_xtalk/ES2015a_A_052308` 410 ms against 100 ms, speech -47 dBFS, room -80 dBFS

No segment although Silero at 0.5 hears the utterance:

- `ami_headset_xtalk/TS3006b_B_212314` 1 words, 420 ms, highest probability 0.89, speech -70 dBFS
- `ami_headset_xtalk/IS1009b_B_109888` 1 words, 410 ms, highest probability 0.89, speech -41 dBFS
- `ami_headset_xtalk/TS3010a_D_025457` 1 words, 200 ms, highest probability 0.89, speech -63 dBFS
- `ami_headset_xtalk/IS1001a_B_027271` 1 words, 410 ms, highest probability 0.88, speech -47 dBFS
- `ami_headset_xtalk/TS3012a_C_077897` 1 words, 260 ms, highest probability 0.87, speech -59 dBFS
- `ami_headset_xtalk/ES2006d_C_075702` 1 words, 340 ms, highest probability 0.86, speech -43 dBFS

Silero's probabilities differ with and without the quiet clock:

- `libri_other/3528-168669-0082_0` 75 frames
- `ami_headset_xtalk/IS1008a_D_025707` 18 frames
- `ami_headset_xtalk/TS3005a_A_092664` 12 frames
- `ami_headset_xtalk/ES2002a_D_057022` 10 frames
- `ami_headset_xtalk/ES2008a_C_100135` 1 frames

