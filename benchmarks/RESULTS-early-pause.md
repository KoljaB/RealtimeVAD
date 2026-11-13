# The early last `pause`: mechanism, features, rule candidates

`benchmarks/early_pause.py` on the close-microphone clips of the word-timing set (6,639
clips, 214 speakers), for 0.0.5 and for the head of `claude/core-quiet-clock` (commit
cdb66d3: DC high-pass and noise subtraction). Read-only: the library's per-block decisions
are recorded by wrapping the instance's `_block_start`; rules are replayed offline.

```bash
python benchmarks/early_pause.py --tag 0.0.5 --processes 10   # 1 minute; PYTHONPATH picks the library
python benchmarks/early_pause.py --report 0.0.5               # 20 s, one process
```

**Early** = the last `pause` before the last `end` is more than 100 ms before the last
word's labelled end and at least 30 ms of audible sound of that word are still to come.
**True stop** = at most 50 ms before the labelled end, or after it.

## Answers

**How many.** 0.0.5: 7.6 % (LibriSpeech clean), 6.2 % (other), 11.0 % (AMI headsets).
Core branch: 8.0 %, 6.5 %, 11.1 %. The branch does not change this failure.

**1. Mechanism: the sound comes back over the threshold, but no second `pause` is
reported.** In 87 % (clean), 93 % (other) and 69 % (AMI) of the early pauses some blocks
between the `pause` and the word's end are not quiet. The loud stretches are short: at
most 20 ms in 26 – 46 % of these turns, 30 to 60 ms in the rest, practically never the
64 ms the library needs to allow a new `pause`. The quiet count itself does start again
(which is why the `end` is right); only the event is not repeated. In the remaining 7 –
31 % the rest of the word stays under the threshold the whole time.

| The audible rest of the word (median) | above the learned room | under the speaker's level | relative to the quiet threshold |
|---|---|---|---|
| crosses the threshold (B) | 13 – 15 dB | 19 – 21 dB | at it (−1 to +2 dB under) |
| stays under it (A) | 8 – 10 dB | 23 – 25 dB | 5 – 6 dB under |

So the remaining sound is not near the room. It is a release, a fricative or a weak last
syllable about 20 dB under the speaker and more than 10 dB above the room.

**2. Nothing at the moment of the `pause` tells the two apart well.** The best single
feature has an AUC of 0.67 (time since the level was last within 10 dB of the speaker);
level above the room 0.60, under the speaker 0.61, slope 0.57 – 0.59, Silero's
probability 0.63, high-band share 0.54. An early pause and a true stop look the same
when they fire: a fall of 12 to 17 dB in 50 ms, Silero still at 1.0. Used as a gate
(report the `pause` at the first quiet block where the gate is open; 0.0.5, speakers
the thresholds were not chosen on, ungated lead 90 ms):

| Early pauses to remove | Best single feature | Best pair | Plain confirmation delay |
|---|---|---|---|
| 50 % | level ≥ 26.7 dB under the speaker: removes 54 %, delays 73 % of the true stops by a median 30 ms, lead left 62 ms | under the speaker ≥ 21.6 dB and Silero falling: 56 %, lead 58 ms | 40 ms: 62 %, lead 42 ms |
| 75 % | Silero's probability falling by 0.05 over 3 frames: 83 %, delays 87 % by 70 ms, lead 26 ms | ≤ 11.6 dB above the room and Silero falling: 80 %, lead 28 ms | 70 ms: 81 %, lead 6 ms |
| 90 % | none; waiting for Silero's own unvoiced frame removes 94 % and leaves no lead | – | – |

A gate costs 30 ms of lead for the first half of the early pauses and 60 ms for three
quarters. The features beat a plain delay by about 20 ms, not more.

**3. A second `pause` is much cheaper than a later first one.** Replayed with a model of
the pause and end logic that reproduces the library's own events in 99.7 % of the clips.
Core branch, clean / other / AMI:

| Rule | early (> 100 ms, sound to come) | speech-off lead, median ms | earlier than Silero | false `pause` per s |
|---|---|---|---|---|
| as it is | 8.0 / 6.5 / 11.1 % | 90 / 88 / 102 | 89 / 89 / 79 % | 1.66 / 1.43 / 1.81 |
| a: quiet = within 6 dB of the room, where the room is ≥ 25 dB under the speaker | 1.9 / 1.3 / 6.3 % | 64 / 50 / 38 | 79 / 74 / 59 % | 1.08 / 0.83 / 1.17 |
| a: within 9 dB | 4.3 / 2.8 / 7.4 % | 76 / 62 / 60 | 83 / 80 / 65 % | 1.26 / 0.98 / 1.29 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 2.2 / 1.1 / 5.7 % | 82 / 78 / 86 | 76 / 77 / 69 % | 2.14 / 1.86 / 2.41 |
| d: as c, only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 2.3 / 1.2 / 5.8 % | 84 / 82 / 90 | 83 / 84 / 71 % | 1.98 / 1.71 / 2.31 |
| d: ≥ 6 dB above the room, ≥ 10 ms | 1.1 / 0.5 / 4.7 % | 76 / 74 / 78 | 76 / 77 / 66 % | 2.33 / 2.02 / 2.73 |
| d: ≥ 10 dB above the room, ≥ 30 ms | 4.1 / 2.4 / 7.0 % | 88 / 86 / 94 | 88 / 87 / 75 % | 1.81 / 1.57 / 2.09 |
| b: second pause after 20 ms at ≥ 10 dB above the room, at the next quiet block within 10 dB of it | 1.6 / 0.7 / 2.5 % | 74 / 58 / 36 | 78 / 74 / 58 % | 2.22 / 2.07 / 2.75 |

For rules b, c and d the lead is that of the last `pause`; the first one still fires
where it does today and becomes one more false alarm (+0.3 per second for d at 20 ms).

**4. General, not a few speakers** (0.0.5; the branch is the same within a point). 60 – 78 % of the speakers have at least one early
pause. Per speaker with 8 turns or more the rate is a median 7.1 % / 5.0 % / 9.1 %,
the upper quartile 11 – 14 %, the maximum 30 – 39 %. The top tenth of the speakers
have 33 – 48 % of the cases. What decides is the word: last words spelled with a final
stop are early in 14 – 16 % of turns, with a final fricative in 5 – 15 %, with a nasal
or liquid in 2 – 7 %.

## Recommendation, in this order

1. **Report a `pause` again when the quiet count starts again after sound that is clearly
   above the room** (rule d: 20 ms of blocks over the quiet threshold and at least 6 dB
   above the learned room, instead of 64 ms). It attacks the mechanism behind 69 – 93 %
   of the cases, cuts the early share to a third on LibriSpeech (8.0 to 2.3 %, 6.5 to
   1.2 %) and to half on AMI (11.1 to 5.8 %), and costs 6 to 12 ms of median lead and
   0.3 false pauses per second. Without the room condition (rule c) room noise that
   merely crosses the threshold also asks for a second pause, and the share earlier than
   Silero drops by another 7 points on LibriSpeech.
2. **Do not gate or delay the first `pause`.** Every causal feature tried costs 30 ms of
   lead for half of the cases, because early pauses and true stops look alike when they
   fire.
3. **"Quiet = near the room" (rule a) only as an option.** It removes the most (down to
   1.3 – 1.9 % on LibriSpeech) and also a third of the false alarms, but it takes 26 to
   64 ms of lead and on quiet AMI speakers leaves the lead at 38 ms. It also changes
   which blocks end a segment; that side (the `end` lead, ends inside words) was not
   evaluated here.
4. What is left after rule d is mechanism A (the rest of the word under the threshold
   all the time, 31 % of the AMI cases) and Silero itself stopping early. Rule b reaches
   A (AMI down to 2.5 %) at the price of most of the lead there.

## Limits

- The labelled end of a word is a forced alignment (LibriSpeech: ±60 – 90 ms for 80 % of
  clips; AMI: a median 50 – 70 ms before the sound stops). "Early" therefore needs both
  the label and audible sound after the pause; AMI numbers are softer than LibriSpeech's.
- The rules are replayed, not implemented: the model ignores soft starts (it takes the
  library's own start times) and matches the library exactly in 99.7 % of the clips.
- Gate thresholds were chosen on half of the speakers and reported on the other half;
  rule parameters were not tuned at all (a small grid, all of it shown below).
- "The room" and "the speaker" in the rules are the library's own learned levels
  (`_noise_db`, `_level_db`) at that block.

# Tables

## Library: 0.0.5

6639 clips. The replay model with the library's own block verdicts gives the library's `pause` and `end` events exactly in 6622 clips (99.7 %).

### How many

| Condition | turns with a speech-off | early | … from the quiet clock | true stops | in between | early: ms before the labelled end, median | audible sound to come, median ms |
|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 1519 | 115 (7.6 %) | 100.0 % | 1166 (76.8 %) | 238 | 140 | 70 |
| LibriSpeech other | 2239 | 139 (6.2 %) | 98.6 % | 1837 (82.0 %) | 263 | 140 | 80 |
| AMI headset | 2157 | 237 (11.0 %) | 82.3 % | 1709 (79.2 %) | 211 | 170 | 130 |

### 1. What the quiet clock saw between an early `pause` and the end of the word

| Condition | early pauses (quiet clock) | A: every block up to the word's end counted as quiet | B: some blocks were not quiet, yet no new `pause` | B: longest loud stretch ≤ 20 ms | 30 – 60 ms | ≥ 70 ms | B: loud blocks in all, median ms | B: clock re-armed on the way |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 115 | 13.0 % | 87.0 % | 37.0 % | 63.0 % | 0.0 % | 40 | 0.0 % |
| LibriSpeech other | 137 | 6.6 % | 93.4 % | 25.8 % | 74.2 % | 0.0 % | 50 | 0.0 % |
| AMI headset | 195 | 30.8 % | 69.2 % | 45.9 % | 53.3 % | 0.7 % | 30 | 0.7 % |

The audible part of the sound that was still to come (median per turn; then median and quartiles over turns):

| Condition |  | turns | pause to labelled end, ms | audible, ms | above the learned room, dB | under the learned speaker level, dB | under the quiet threshold, dB (negative: over it) | above the clip's room (10th percentile), dB | under the clip's speech level, dB | share above 3 kHz |
|---|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | A | 15 | 140 | 80 | 8 (7 – 9) | 23 (21 – 25) | 6 (4 – 7) | 10 (8 – 12) | 20 (18 – 22) | 0.27 |
| LibriSpeech clean | B | 100 | 145 | 70 | 13 (10 – 16) | 19 (16 – 21) | -0 (-3 – 1) | 14 (10 – 17) | 17 (14 – 20) | 0.22 |
| LibriSpeech other | A | 9 | 110 | 50 | 8 (7 – 10) | 23 (21 – 27) | 5 (4 – 7) | 10 (8 – 11) | 19 (15 – 23) | 0.15 |
| LibriSpeech other | B | 128 | 140 | 80 | 15 (11 – 20) | 19 (17 – 21) | -1 (-4 – 1) | 15 (11 – 21) | 18 (15 – 20) | 0.31 |
| AMI headset | A | 60 | 150 | 105 | 10 (8 – 12) | 25 (23 – 27) | 6 (4 – 7) | 12 (10 – 15) | 14 (9 – 21) | 0.03 |
| AMI headset | B | 135 | 190 | 130 | 13 (10 – 17) | 21 (18 – 24) | 2 (-0 – 4) | 14 (12 – 18) | 13 (7 – 19) | 0.02 |

### 2. Features at the moment the last `pause` fires

| Feature at the pause's block | early: median (p25 – p75) | true stop: median (p25 – p75) | separation (AUC, 0.5 = none) |
|---|---|---|---|
| level above the learned room, dB | 13 (10 – 16) | 11 (5 – 15) | 0.60 |
| level under the speaker's level, dB | 21 (19 – 25) | 23 (21 – 27) | 0.61 |
| level under the quiet threshold, dB | 2 (1 – 4) | 2 (1 – 4) | 0.50 |
| level change over the last 30 ms, dB | -14 (-18 – -9) | -12 (-16 – -7) | 0.57 |
| level change over the last 50 ms, dB | -17 (-22 – -14) | -15 (-20 – -10) | 0.59 |
| Silero's probability (latest frame) | 1.00 (1.00 – 1.00) | 1.00 (0.97 – 1.00) | 0.63 |
| its change over the last 3 frames | -0.00 (-0.00 – 0.00) | -0.00 (-0.01 – -0.00) | 0.64 |
| share of the energy above 3 kHz (20 ms) | 0.01 (0.00 – 0.06) | 0.01 (0.00 – 0.10) | 0.54 |
| ms since Silero's last frame at 0.9 or more | 16 (8 – 24) | 20 (10 – 30) | 0.58 |
| ms since the level was last within 10 dB of the speaker | 30 (20 – 40) | 40 (30 – 80) | 0.67 |

A gate that never opens leaves the speech-off to Silero's first unvoiced frame. That is the ceiling: it removes 94.0 % of the early pauses (in the rest Silero itself stops before the labelled end) and leaves a lead of -4 ms.

Gates chosen to remove 50 % of the early pauses on half of the speakers; numbers from the other half (ungated there: lead 90 ms, earlier than Silero in 82.7 %). Features that cannot reach the target are missing:

| `pause` is reported at the first quiet block where | early pauses removed | true stops delayed | … by, median ms | mean delay over all true stops, ms | speech-off lead left, median ms | earlier than Silero in | LibriSpeech clean: removed / lead ms | LibriSpeech other: removed / lead ms | AMI headset: removed / lead ms |
|---|---|---|---|---|---|---|---|---|---|
| best pair: level above the learned room, dB ≤ 12.9 and ms since the level was last within 10 dB of the speaker ≥ 60 | 56.0 % | 74.1 % | 30 | 25 | 62 | 79.0 % | 49.1 % / 60 | 64.5 % / 64 | 54.2 % / 62 |
| level under the speaker's level, dB ≥ 26.7 | 53.5 % | 72.5 % | 30 | 32 | 62 | 69.9 % | 61.8 % / 58 | 48.4 % / 70 | 51.8 % / 52 |
| level above the learned room, dB ≤ 7.18 | 52.0 % | 65.8 % | 30 | 29 | 60 | 72.8 % | 60.0 % / 62 | 48.4 % / 62 | 49.4 % / 46 |
| its change over the last 3 frames ≤ -0.00131 | 57.0 % | 64.0 % | 50 | 31 | 58 | 81.8 % | 74.5 % / 50 | 59.7 % / 58 | 43.4 % / 62 |
| level under the quiet threshold, dB ≥ 7.24 | 61.0 % | 94.2 % | 30 | 37 | 54 | 68.8 % | 70.9 % / 50 | 61.3 % / 54 | 54.2 % / 60 |
| ms since the level was last within 10 dB of the speaker ≥ 90 | 66.5 % | 76.5 % | 50 | 39 | 44 | 79.2 % | 70.9 % / 40 | 69.4 % / 48 | 61.4 % / 42 |
| quiet blocks in a row so far (a plain confirmation delay) ≥ 5 | 61.5 % | 99.9 % | 40 | 46 | 42 | 71.6 % | 63.6 % / 37 | 77.4 % / 38 | 48.2 % / 53 |
| level change over the last 50 ms, dB ≥ -7.25 | 56.5 % | 85.3 % | 50 | 45 | 38 | 75.4 % | 56.4 % / 32 | 75.8 % / 34 | 42.2 % / 48 |
| level change over the last 30 ms, dB ≥ -2.38 | 55.0 % | 96.6 % | 40 | 46 | 38 | 72.6 % | 58.2 % / 30 | 75.8 % / 32 | 37.3 % / 52 |
| Silero's probability (latest frame) ≤ 0.99 | 70.0 % | 70.3 % | 60 | 43 | 30 | 79.3 % | 90.9 % / 28 | 77.4 % / 30 | 50.6 % / 58 |
| ms since Silero's last frame at 0.9 or more ≥ 34 | 83.5 % | 81.5 % | 80 | 64 | 22 | 61.9 % | 96.4 % / 10 | 90.3 % / 22 | 69.9 % / 26 |
| share of the energy above 3 kHz (20 ms) ≥ 0.115 | 55.0 % | 77.0 % | 80 | 56 | 0 | 49.0 % | 50.9 % / 0 | 61.3 % / 0 | 53.0 % / 0 |

Gates chosen to remove 75 % of the early pauses on half of the speakers; numbers from the other half (ungated there: lead 90 ms, earlier than Silero in 82.7 %). Features that cannot reach the target are missing:

| `pause` is reported at the first quiet block where | early pauses removed | true stops delayed | … by, median ms | mean delay over all true stops, ms | speech-off lead left, median ms | earlier than Silero in | LibriSpeech clean: removed / lead ms | LibriSpeech other: removed / lead ms | AMI headset: removed / lead ms |
|---|---|---|---|---|---|---|---|---|---|
| best pair: level above the learned room, dB ≤ 8.65 and its change over the last 3 frames ≤ -0.00425 | 77.0 % | 84.4 % | 60 | 51 | 28 | 72.6 % | 92.7 % / 28 | 75.8 % / 28 | 67.5 % / 30 |
| its change over the last 3 frames ≤ -0.0522 | 82.5 % | 86.9 % | 70 | 61 | 26 | 68.9 % | 92.7 % / 24 | 87.1 % / 26 | 72.3 % / 28 |
| Silero's probability (latest frame) ≤ 0.886 | 84.5 % | 81.7 % | 80 | 64 | 24 | 60.8 % | 96.4 % / 5 | 90.3 % / 24 | 72.3 % / 28 |
| ms since Silero's last frame at 0.9 or more ≥ 34 | 83.5 % | 81.5 % | 80 | 64 | 22 | 61.9 % | 96.4 % / 10 | 90.3 % / 22 | 69.9 % / 26 |
| quiet blocks in a row so far (a plain confirmation delay) ≥ 8 | 81.0 % | 99.9 % | 70 | 70 | 6 | 57.7 % | 83.6 % / 0 | 87.1 % / 6 | 74.7 % / 18 |
| ms since the level was last within 10 dB of the speaker ≥ 130 | 82.0 % | 85.0 % | 80 | 66 | 6 | 56.9 % | 92.7 % / 2 | 85.5 % / 10 | 72.3 % / 4 |
| level above the learned room, dB ≤ 1.27 | 79.5 % | 83.1 % | 70 | 58 | 0 | 52.8 % | 83.6 % / 24 | 82.3 % / 10 | 74.7 % / -0 |
| level under the speaker's level, dB ≥ 33 | 81.5 % | 94.9 % | 60 | 61 | 0 | 47.8 % | 89.1 % / 0 | 77.4 % / 12 | 79.5 % / -0 |
| level under the quiet threshold, dB ≥ 12.3 | 79.5 % | 99.6 % | 60 | 66 | 0 | 46.4 % | 89.1 % / 0 | 79.0 % / 0 | 73.5 % / 0 |
| level change over the last 50 ms, dB ≥ 0.754 | 81.5 % | 100.0 % | 80 | 77 | -0 | 39.5 % | 90.9 % / -0 | 87.1 % / -2 | 71.1 % / 0 |
| level change over the last 30 ms, dB ≥ 3.43 | 82.0 % | 100.0 % | 80 | 81 | -2 | 31.2 % | 94.5 % / -2 | 87.1 % / -4 | 69.9 % / -2 |
| share of the energy above 3 kHz (20 ms) ≤ 0.000423 | 83.0 % | 93.3 % | 90 | 86 | -4 | 16.6 % | 90.9 % / -4 | 82.3 % / -4 | 78.3 % / -4 |

No gate reaches 90 % on the speakers it was chosen on.

### 3. Rule candidates, replayed

LibriSpeech clean:

| Rule | turns | speech-off > 50 ms before the labelled end | > 100 ms | > 100 ms with sound to come | > 200 ms | speech-off lead, median ms | earlier than Silero in | false `pause` per s of speech |
|---|---|---|---|---|---|---|---|---|
| library (measured) | 1519 | 23.2 % | 8.9 % | 7.6 % | 1.4 % | 88 | 83.6 % | 1.64 |
| replay model, no rule | 1519 | 23.2 % | 8.9 % | 7.6 % | 1.4 % | 88 | 83.6 % | 1.64 |
| a: quiet = within 6 dB of the room where it is ≥ 20 dB under the speaker | 1518 | 10.3 % | 3.5 % | 2.5 % | 0.5 % | 66 | 77.2 % | 1.08 |
| a: quiet = within 9 dB of the room where it is ≥ 20 dB under the speaker | 1521 | 18.1 % | 7.0 % | 6.0 % | 1.1 % | 78 | 81.3 % | 1.30 |
| a: quiet = within 12 dB of the room where it is ≥ 20 dB under the speaker | 1523 | 25.9 % | 10.8 % | 9.7 % | 2.2 % | 90 | 84.0 % | 1.53 |
| a: quiet = within 6 dB of the room where it is ≥ 25 dB under the speaker | 1518 | 8.8 % | 3.0 % | 2.2 % | 0.5 % | 62 | 74.7 % | 1.09 |
| a: quiet = within 9 dB of the room where it is ≥ 25 dB under the speaker | 1519 | 14.4 % | 5.2 % | 4.3 % | 0.9 % | 74 | 78.3 % | 1.26 |
| a: quiet = within 12 dB of the room where it is ≥ 25 dB under the speaker | 1521 | 19.9 % | 7.3 % | 6.2 % | 1.2 % | 82 | 80.9 % | 1.46 |
| a: quiet = within 6 dB of the room where it is ≥ 30 dB under the speaker | 1518 | 10.5 % | 4.0 % | 3.2 % | 0.7 % | 62 | 74.5 % | 1.19 |
| a: quiet = within 9 dB of the room where it is ≥ 30 dB under the speaker | 1518 | 13.6 % | 5.2 % | 4.3 % | 0.7 % | 72 | 77.5 % | 1.30 |
| a: quiet = within 12 dB of the room where it is ≥ 30 dB under the speaker | 1518 | 17.3 % | 6.7 % | 5.5 % | 0.9 % | 80 | 79.8 % | 1.43 |
| b: second pause after 10 ms at ≥ 6 dB above the room | 1519 | 2.8 % | 0.6 % | 0.1 % | 0.0 % | 0 | 49.1 % | 2.78 |
| b: second pause after 20 ms at ≥ 6 dB above the room | 1519 | 4.3 % | 1.1 % | 0.3 % | 0.2 % | 42 | 61.5 % | 2.43 |
| b: second pause after 30 ms at ≥ 6 dB above the room | 1519 | 5.7 % | 1.7 % | 0.9 % | 0.3 % | 54 | 68.0 % | 2.29 |
| b: second pause after 10 ms at ≥ 10 dB above the room | 1519 | 7.2 % | 1.9 % | 1.1 % | 0.2 % | 58 | 65.9 % | 2.43 |
| b: second pause after 20 ms at ≥ 10 dB above the room | 1519 | 9.4 % | 2.8 % | 1.8 % | 0.4 % | 68 | 72.2 % | 2.18 |
| b: second pause after 30 ms at ≥ 10 dB above the room | 1519 | 12.0 % | 3.7 % | 2.6 % | 0.5 % | 72 | 75.5 % | 2.07 |
| b: second pause after 10 ms at ≥ 15 dB above the room | 1519 | 13.0 % | 3.9 % | 3.0 % | 0.5 % | 76 | 76.4 % | 2.08 |
| b: second pause after 20 ms at ≥ 15 dB above the room | 1519 | 15.5 % | 4.9 % | 3.8 % | 0.5 % | 80 | 78.6 % | 1.90 |
| b: second pause after 30 ms at ≥ 15 dB above the room | 1519 | 17.4 % | 5.8 % | 4.6 % | 0.7 % | 82 | 79.6 % | 1.83 |
| c: second pause when the count starts again after ≥ 10 ms over the threshold | 1519 | 10.3 % | 2.2 % | 1.1 % | 0.1 % | 62 | 59.1 % | 2.71 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 1519 | 13.3 % | 3.4 % | 2.2 % | 0.1 % | 74 | 67.4 % | 2.17 |
| c: second pause when the count starts again after ≥ 30 ms over the threshold | 1519 | 16.5 % | 5.0 % | 3.6 % | 0.4 % | 82 | 74.7 % | 1.94 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 10 ms | 1519 | 10.8 % | 2.4 % | 1.1 % | 0.1 % | 70 | 67.2 % | 2.36 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 1519 | 14.4 % | 3.8 % | 2.4 % | 0.3 % | 80 | 76.4 % | 1.99 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 30 ms | 1519 | 17.4 % | 5.3 % | 3.9 % | 0.5 % | 84 | 80.3 % | 1.83 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 10 ms | 1519 | 12.1 % | 2.9 % | 1.6 % | 0.3 % | 76 | 74.1 % | 2.19 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 20 ms | 1519 | 15.3 % | 4.2 % | 2.8 % | 0.5 % | 82 | 79.5 % | 1.91 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 30 ms | 1519 | 18.4 % | 5.6 % | 4.2 % | 0.6 % | 84 | 81.8 % | 1.79 |

LibriSpeech other:

| Rule | turns | speech-off > 50 ms before the labelled end | > 100 ms | > 100 ms with sound to come | > 200 ms | speech-off lead, median ms | earlier than Silero in | false `pause` per s of speech |
|---|---|---|---|---|---|---|---|---|
| library (measured) | 2239 | 18.0 % | 7.5 % | 6.2 % | 0.7 % | 86 | 82.2 % | 1.42 |
| replay model, no rule | 2239 | 18.0 % | 7.5 % | 6.3 % | 0.7 % | 86 | 82.2 % | 1.42 |
| a: quiet = within 6 dB of the room where it is ≥ 20 dB under the speaker | 2237 | 7.2 % | 2.7 % | 1.9 % | 0.2 % | 46 | 70.6 % | 0.89 |
| a: quiet = within 9 dB of the room where it is ≥ 20 dB under the speaker | 2238 | 11.3 % | 5.0 % | 4.2 % | 0.6 % | 62 | 76.8 % | 1.07 |
| a: quiet = within 12 dB of the room where it is ≥ 20 dB under the speaker | 2238 | 17.4 % | 7.6 % | 6.7 % | 1.5 % | 74 | 80.4 % | 1.26 |
| a: quiet = within 6 dB of the room where it is ≥ 25 dB under the speaker | 2237 | 6.2 % | 2.0 % | 1.4 % | 0.1 % | 42 | 68.4 % | 0.89 |
| a: quiet = within 9 dB of the room where it is ≥ 25 dB under the speaker | 2237 | 9.0 % | 3.4 % | 2.7 % | 0.2 % | 58 | 74.4 % | 1.03 |
| a: quiet = within 12 dB of the room where it is ≥ 25 dB under the speaker | 2237 | 13.5 % | 5.4 % | 4.5 % | 0.6 % | 70 | 77.9 % | 1.19 |
| a: quiet = within 6 dB of the room where it is ≥ 30 dB under the speaker | 2237 | 7.4 % | 2.9 % | 2.3 % | 0.2 % | 42 | 68.7 % | 0.96 |
| a: quiet = within 9 dB of the room where it is ≥ 30 dB under the speaker | 2237 | 8.9 % | 3.4 % | 2.6 % | 0.3 % | 56 | 74.3 % | 1.06 |
| a: quiet = within 12 dB of the room where it is ≥ 30 dB under the speaker | 2237 | 12.0 % | 4.7 % | 3.9 % | 0.4 % | 68 | 77.5 % | 1.16 |
| b: second pause after 10 ms at ≥ 6 dB above the room | 2239 | 2.0 % | 0.4 % | 0.0 % | 0.0 % | 0 | 44.6 % | 2.57 |
| b: second pause after 20 ms at ≥ 6 dB above the room | 2239 | 3.3 % | 0.8 % | 0.2 % | 0.0 % | 18 | 56.0 % | 2.24 |
| b: second pause after 30 ms at ≥ 6 dB above the room | 2239 | 4.5 % | 1.1 % | 0.3 % | 0.0 % | 30 | 61.3 % | 2.11 |
| b: second pause after 10 ms at ≥ 10 dB above the room | 2239 | 5.3 % | 0.9 % | 0.4 % | 0.0 % | 40 | 63.5 % | 2.23 |
| b: second pause after 20 ms at ≥ 10 dB above the room | 2239 | 6.7 % | 1.5 % | 0.9 % | 0.1 % | 54 | 69.8 % | 2.02 |
| b: second pause after 30 ms at ≥ 10 dB above the room | 2239 | 7.6 % | 1.9 % | 1.3 % | 0.1 % | 58 | 71.9 % | 1.94 |
| b: second pause after 10 ms at ≥ 15 dB above the room | 2239 | 9.3 % | 2.1 % | 1.4 % | 0.1 % | 70 | 74.6 % | 1.88 |
| b: second pause after 20 ms at ≥ 15 dB above the room | 2239 | 11.0 % | 2.9 % | 2.3 % | 0.3 % | 72 | 76.8 % | 1.73 |
| b: second pause after 30 ms at ≥ 15 dB above the room | 2239 | 12.1 % | 3.7 % | 3.0 % | 0.4 % | 74 | 77.7 % | 1.67 |
| c: second pause when the count starts again after ≥ 10 ms over the threshold | 2239 | 8.3 % | 0.9 % | 0.3 % | 0.0 % | 60 | 62.4 % | 2.38 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 2239 | 10.1 % | 1.8 % | 1.0 % | 0.1 % | 70 | 67.9 % | 1.93 |
| c: second pause when the count starts again after ≥ 30 ms over the threshold | 2239 | 12.0 % | 2.9 % | 1.9 % | 0.2 % | 76 | 73.4 % | 1.71 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 10 ms | 2239 | 8.7 % | 1.2 % | 0.3 % | 0.0 % | 66 | 70.1 % | 2.03 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 2239 | 11.2 % | 2.2 % | 1.2 % | 0.1 % | 76 | 77.2 % | 1.72 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 30 ms | 2239 | 13.1 % | 3.2 % | 2.1 % | 0.2 % | 82 | 80.1 % | 1.59 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 10 ms | 2239 | 10.0 % | 1.5 % | 0.6 % | 0.0 % | 74 | 76.2 % | 1.87 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 20 ms | 2239 | 12.1 % | 2.6 % | 1.6 % | 0.2 % | 78 | 79.4 % | 1.65 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 30 ms | 2239 | 13.6 % | 3.6 % | 2.5 % | 0.3 % | 82 | 80.9 % | 1.55 |

AMI headset:

| Rule | turns | speech-off > 50 ms before the labelled end | > 100 ms | > 100 ms with sound to come | > 200 ms | speech-off lead, median ms | earlier than Silero in | false `pause` per s of speech |
|---|---|---|---|---|---|---|---|---|
| library (measured) | 2157 | 20.8 % | 11.9 % | 11.0 % | 3.3 % | 100 | 78.2 % | 1.79 |
| replay model, no rule | 2157 | 20.8 % | 11.9 % | 11.0 % | 3.3 % | 100 | 78.2 % | 1.79 |
| a: quiet = within 6 dB of the room where it is ≥ 20 dB under the speaker | 2179 | 12.3 % | 7.7 % | 6.3 % | 1.4 % | 32 | 57.5 % | 1.14 |
| a: quiet = within 9 dB of the room where it is ≥ 20 dB under the speaker | 2179 | 15.2 % | 9.5 % | 8.2 % | 1.9 % | 64 | 65.4 % | 1.28 |
| a: quiet = within 12 dB of the room where it is ≥ 20 dB under the speaker | 2181 | 17.9 % | 11.4 % | 10.3 % | 2.6 % | 86 | 71.7 % | 1.45 |
| a: quiet = within 6 dB of the room where it is ≥ 25 dB under the speaker | 2178 | 12.2 % | 7.6 % | 6.2 % | 1.4 % | 30 | 56.9 % | 1.14 |
| a: quiet = within 9 dB of the room where it is ≥ 25 dB under the speaker | 2178 | 14.4 % | 8.9 % | 7.6 % | 1.8 % | 58 | 63.7 % | 1.28 |
| a: quiet = within 12 dB of the room where it is ≥ 25 dB under the speaker | 2178 | 16.7 % | 10.2 % | 9.1 % | 2.2 % | 80 | 69.4 % | 1.44 |
| a: quiet = within 6 dB of the room where it is ≥ 30 dB under the speaker | 2174 | 13.2 % | 8.3 % | 7.0 % | 1.4 % | 36 | 58.4 % | 1.19 |
| a: quiet = within 9 dB of the room where it is ≥ 30 dB under the speaker | 2176 | 14.6 % | 9.2 % | 8.0 % | 1.9 % | 58 | 63.5 % | 1.30 |
| a: quiet = within 12 dB of the room where it is ≥ 30 dB under the speaker | 2173 | 16.2 % | 9.8 % | 8.7 % | 2.3 % | 78 | 68.5 % | 1.43 |
| b: second pause after 10 ms at ≥ 6 dB above the room | 2157 | 3.6 % | 1.7 % | 1.0 % | 0.4 % | -20 | 36.8 % | 3.36 |
| b: second pause after 20 ms at ≥ 6 dB above the room | 2157 | 4.4 % | 2.3 % | 1.4 % | 0.6 % | 0 | 43.2 % | 2.98 |
| b: second pause after 30 ms at ≥ 6 dB above the room | 2157 | 5.3 % | 3.0 % | 1.9 % | 0.7 % | 0 | 47.1 % | 2.83 |
| b: second pause after 10 ms at ≥ 10 dB above the room | 2157 | 5.8 % | 2.8 % | 2.0 % | 0.6 % | 18 | 53.5 % | 3.04 |
| b: second pause after 20 ms at ≥ 10 dB above the room | 2157 | 7.3 % | 3.6 % | 2.6 % | 0.9 % | 34 | 57.8 % | 2.75 |
| b: second pause after 30 ms at ≥ 10 dB above the room | 2157 | 9.1 % | 4.6 % | 3.4 % | 1.0 % | 44 | 60.3 % | 2.61 |
| b: second pause after 10 ms at ≥ 15 dB above the room | 2157 | 10.3 % | 5.7 % | 4.8 % | 1.4 % | 74 | 66.3 % | 2.63 |
| b: second pause after 20 ms at ≥ 15 dB above the room | 2157 | 12.1 % | 7.0 % | 6.0 % | 1.7 % | 82 | 69.4 % | 2.38 |
| b: second pause after 30 ms at ≥ 15 dB above the room | 2157 | 13.8 % | 7.9 % | 6.8 % | 1.8 % | 86 | 71.4 % | 2.25 |
| c: second pause when the count starts again after ≥ 10 ms over the threshold | 2157 | 10.5 % | 5.0 % | 4.6 % | 1.3 % | 72 | 62.5 % | 2.95 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 2157 | 12.5 % | 6.5 % | 5.7 % | 1.6 % | 86 | 68.0 % | 2.41 |
| c: second pause when the count starts again after ≥ 30 ms over the threshold | 2157 | 14.5 % | 7.6 % | 6.7 % | 2.0 % | 92 | 71.6 % | 2.16 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 10 ms | 2157 | 10.7 % | 5.2 % | 4.6 % | 1.3 % | 76 | 65.4 % | 2.75 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 2157 | 12.8 % | 6.7 % | 5.8 % | 1.7 % | 88 | 70.5 % | 2.31 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 30 ms | 2157 | 14.9 % | 7.9 % | 6.9 % | 2.1 % | 92 | 73.4 % | 2.10 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 10 ms | 2157 | 11.1 % | 5.5 % | 4.9 % | 1.4 % | 80 | 68.1 % | 2.63 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 20 ms | 2157 | 13.4 % | 7.0 % | 6.1 % | 1.9 % | 90 | 71.6 % | 2.26 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 30 ms | 2157 | 15.7 % | 8.2 % | 7.2 % | 2.2 % | 94 | 73.9 % | 2.07 |

### 4. A few speakers or general

| Condition | speakers | with at least one early pause | speakers with ≥ 8 turns | their early rate, %: p25 / median / p75 | share of all early pauses from the top tenth of speakers |
|---|---|---|---|---|---|
| LibriSpeech clean | 79 | 59.5 % | 57 | 0.0 / 7.1 / 11.1 (max 30.0) | 38.3 % |
| LibriSpeech other | 64 | 62.5 % | 53 | 0.0 / 5.0 / 9.1 (max 25.0) | 48.2 % |
| AMI headset | 71 | 77.5 % | 66 | 4.0 / 9.1 / 13.7 (max 38.7) | 32.9 % |

Early rate by how the last word is spelled at its end:

| Last word ends in | LibriSpeech clean | LibriSpeech other | AMI headset |
|---|---|---|---|
| fricative (s, z, f, th, sh, ch, x, ce, se, ge) | 7.5 % of 388 | 4.5 % of 486 | 15.0 % of 293 |
| nasal or liquid (m, n, ng, l, r) | 2.8 % of 461 | 2.4 % of 670 | 6.6 % of 593 |
| stop (t, d, k, p, g, b, ed, te, de, ke) | 14.0 % of 422 | 14.2 % of 684 | 16.3 % of 453 |
| vowel or other | 5.6 % of 248 | 1.0 % of 399 | 9.8 % of 818 |

### Example clips

A: the rest of the word stayed under the quiet threshold (LibriSpeech):

- `libri_other/1686-142278-0054_0` "yes": `pause` 190 ms before the labelled end, 170 ms audible after it, 7 dB above the room, 24 dB under the speaker
- `libri_clean/1089-134691-0018_1` "again": `pause` 200 ms before the labelled end, 160 ms audible after it, 9 dB above the room, 24 dB under the speaker
- `libri_clean/1188-133604-0023_7` "earth": `pause` 180 ms before the labelled end, 150 ms audible after it, 11 dB above the room, 20 dB under the speaker
- `libri_clean/3575-170457-0046_0` "haworth": `pause` 170 ms before the labelled end, 140 ms audible after it, 9 dB above the room, 16 dB under the speaker
- `libri_clean/1988-24833-0010_0` "cares": `pause` 170 ms before the labelled end, 130 ms audible after it, 6 dB above the room, 28 dB under the speaker
- `libri_clean/174-84280-0013_1` "of": `pause` 130 ms before the labelled end, 100 ms audible after it, 8 dB above the room, 24 dB under the speaker
- `libri_clean/5105-28240-0013_0` "yourself": `pause` 100 ms before the labelled end, 100 ms audible after it, 7 dB above the room, 19 dB under the speaker
- `libri_clean/8842-302203-0011_0` "now": `pause` 120 ms before the labelled end, 90 ms audible after it, 2 dB above the room, 30 dB under the speaker
- `libri_other/1686-142278-0059_1` "margaret": `pause` 110 ms before the labelled end, 80 ms audible after it, 6 dB above the room, 21 dB under the speaker
- `libri_clean/2902-9008-0009_0` "earth": `pause` 110 ms before the labelled end, 80 ms audible after it, 13 dB above the room, 23 dB under the speaker

A: the rest of the word stayed under the quiet threshold (AMI headset):

- `ami_headset_xtalk/TS3012a_A_004223` "yeah": `pause` 210 ms before the labelled end, 210 ms audible after it, 11 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3008a_D_010724` "up": `pause` 190 ms before the labelled end, 190 ms audible after it, 9 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3012a_C_050774` "stripes": `pause` 230 ms before the labelled end, 190 ms audible after it, 9 dB above the room, 13 dB under the speaker
- `ami_headset_xtalk/ES2015a_D_049187` "shapely": `pause` 210 ms before the labelled end, 180 ms audible after it, 1 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3010a_A_066835` "that": `pause` 180 ms before the labelled end, 180 ms audible after it, 12 dB above the room, 30 dB under the speaker
- `ami_headset_xtalk/ES2015a_A_028462` "control": `pause` 230 ms before the labelled end, 170 ms audible after it, -0 dB above the room, 27 dB under the speaker
- `ami_headset_xtalk/TS3005a_C_073542` "world": `pause` 210 ms before the labelled end, 170 ms audible after it, 15 dB above the room, 22 dB under the speaker
- `ami_headset_xtalk/TS3008a_A_072976` "next": `pause` 190 ms before the labelled end, 170 ms audible after it, 9 dB above the room, 23 dB under the speaker
- `ami_headset_xtalk/ES2002a_B_052321` "um": `pause` 210 ms before the labelled end, 160 ms audible after it, 19 dB above the room, 27 dB under the speaker
- `ami_headset_xtalk/ES2006d_A_123844` "uh": `pause` 170 ms before the labelled end, 160 ms audible after it, 10 dB above the room, 31 dB under the speaker

B: the rest of the word crossed the threshold, no new `pause` (LibriSpeech):

- `libri_clean/5683-32879-0021_0` "moved": `pause` 300 ms before the labelled end, 220 ms audible after it, 14 dB above the room, 20 dB under the speaker
- `libri_clean/2035-147961-0027_0` "yes": `pause` 210 ms before the labelled end, 210 ms audible after it, 13 dB above the room, 17 dB under the speaker
- `libri_other/2414-159411-0010_1` "judgment": `pause` 200 ms before the labelled end, 200 ms audible after it, 16 dB above the room, 24 dB under the speaker
- `libri_clean/2035-147961-0025_0` "asked": `pause` 270 ms before the labelled end, 190 ms audible after it, 12 dB above the room, 21 dB under the speaker
- `libri_clean/2035-147961-0028_0` "enough": `pause` 210 ms before the labelled end, 190 ms audible after it, 15 dB above the room, 22 dB under the speaker
- `libri_clean/1995-1826-0023_0` "yams": `pause` 210 ms before the labelled end, 180 ms audible after it, 11 dB above the room, 23 dB under the speaker
- `libri_other/3660-172182-0001_0` "arthur": `pause` 260 ms before the labelled end, 180 ms audible after it, 13 dB above the room, 17 dB under the speaker
- `libri_clean/3729-6852-0003_1` "life": `pause` 210 ms before the labelled end, 180 ms audible after it, 12 dB above the room, 20 dB under the speaker
- `libri_clean/6241-61943-0027_1` "unhappy": `pause` 270 ms before the labelled end, 180 ms audible after it, 10 dB above the room, 18 dB under the speaker
- `libri_other/8288-274162-0035_0` "withhold": `pause` 290 ms before the labelled end, 180 ms audible after it, 16 dB above the room, 18 dB under the speaker

B: the rest of the word crossed the threshold, no new `pause` (AMI headset):

- `ami_headset_xtalk/ES2010a_B_043002` "everything": `pause` 330 ms before the labelled end, 330 ms audible after it, 20 dB above the room, 21 dB under the speaker
- `ami_headset/TS3010a_A_016208` "colour": `pause` 280 ms before the labelled end, 280 ms audible after it, 14 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/ES2006d_C_047726` "mm-hmm": `pause` 270 ms before the labelled end, 270 ms audible after it, 25 dB above the room, 21 dB under the speaker
- `ami_headset_xtalk/TS3006b_D_041230` "a": `pause` 270 ms before the labelled end, 270 ms audible after it, 13 dB above the room, 29 dB under the speaker
- `ami_headset_xtalk/ES2015a_B_109194` "yeah": `pause` 260 ms before the labelled end, 260 ms audible after it, 11 dB above the room, 29 dB under the speaker
- `ami_headset/TS3006b_D_207414` "yeah": `pause` 260 ms before the labelled end, 260 ms audible after it, 17 dB above the room, 25 dB under the speaker
- `ami_headset_xtalk/ES2004c_C_116288` "yeah": `pause` 250 ms before the labelled end, 250 ms audible after it, 21 dB above the room, 19 dB under the speaker
- `ami_headset_xtalk/TS3006b_D_121574` "for": `pause` 260 ms before the labelled end, 250 ms audible after it, 7 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3006b_D_191901` "so": `pause` 250 ms before the labelled end, 250 ms audible after it, 10 dB above the room, 28 dB under the speaker
- `ami_headset_xtalk/TS3008a_B_030071` "yes": `pause` 250 ms before the labelled end, 250 ms audible after it, 16 dB above the room, 19 dB under the speaker

## Library: core-cdb66d3

6639 clips. The replay model with the library's own block verdicts gives the library's `pause` and `end` events exactly in 6621 clips (99.7 %).

### How many

| Condition | turns with a speech-off | early | … from the quiet clock | true stops | in between | early: ms before the labelled end, median | audible sound to come, median ms |
|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 1518 | 121 (8.0 %) | 100.0 % | 1151 (75.8 %) | 246 | 150 | 70 |
| LibriSpeech other | 2239 | 146 (6.5 %) | 98.6 % | 1797 (80.3 %) | 296 | 140 | 80 |
| AMI headset | 2157 | 239 (11.1 %) | 82.0 % | 1700 (78.8 %) | 218 | 160 | 130 |

### 1. What the quiet clock saw between an early `pause` and the end of the word

| Condition | early pauses (quiet clock) | A: every block up to the word's end counted as quiet | B: some blocks were not quiet, yet no new `pause` | B: longest loud stretch ≤ 20 ms | 30 – 60 ms | ≥ 70 ms | B: loud blocks in all, median ms | B: clock re-armed on the way |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 121 | 9.9 % | 90.1 % | 37.6 % | 62.4 % | 0.0 % | 40 | 0.0 % |
| LibriSpeech other | 144 | 6.9 % | 93.1 % | 26.9 % | 73.1 % | 0.0 % | 50 | 0.0 % |
| AMI headset | 196 | 31.1 % | 68.9 % | 45.2 % | 54.1 % | 0.7 % | 30 | 0.7 % |

The audible part of the sound that was still to come (median per turn; then median and quartiles over turns):

| Condition |  | turns | pause to labelled end, ms | audible, ms | above the learned room, dB | under the learned speaker level, dB | under the quiet threshold, dB (negative: over it) | above the clip's room (10th percentile), dB | under the clip's speech level, dB | share above 3 kHz |
|---|---|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | A | 12 | 155 | 90 | 8 (8 – 9) | 23 (22 – 24) | 5 (3 – 7) | 10 (8 – 12) | 20 (18 – 22) | 0.37 |
| LibriSpeech clean | B | 109 | 150 | 70 | 14 (10 – 17) | 19 (17 – 22) | -0 (-3 – 2) | 13 (9 – 17) | 17 (14 – 20) | 0.19 |
| LibriSpeech other | A | 10 | 110 | 50 | 8 (7 – 10) | 24 (22 – 26) | 6 (4 – 7) | 9 (8 – 11) | 19 (16 – 22) | 0.14 |
| LibriSpeech other | B | 134 | 140 | 80 | 15 (11 – 23) | 19 (16 – 22) | -1 (-4 – 1) | 15 (10 – 21) | 17 (14 – 20) | 0.28 |
| AMI headset | A | 61 | 150 | 110 | 10 (8 – 12) | 25 (24 – 28) | 6 (5 – 8) | 12 (10 – 15) | 14 (9 – 21) | 0.03 |
| AMI headset | B | 135 | 190 | 130 | 13 (11 – 17) | 21 (18 – 24) | 2 (-0 – 4) | 14 (12 – 18) | 13 (7 – 19) | 0.02 |

### 2. Features at the moment the last `pause` fires

| Feature at the pause's block | early: median (p25 – p75) | true stop: median (p25 – p75) | separation (AUC, 0.5 = none) |
|---|---|---|---|
| level above the learned room, dB | 13 (10 – 17) | 11 (5 – 15) | 0.61 |
| level under the speaker's level, dB | 22 (19 – 25) | 23 (21 – 27) | 0.59 |
| level under the quiet threshold, dB | 2 (1 – 4) | 2 (1 – 4) | 0.50 |
| level change over the last 30 ms, dB | -14 (-18 – -8) | -12 (-16 – -8) | 0.56 |
| level change over the last 50 ms, dB | -17 (-22 – -13) | -16 (-20 – -10) | 0.58 |
| Silero's probability (latest frame) | 1.00 (1.00 – 1.00) | 1.00 (0.98 – 1.00) | 0.62 |
| its change over the last 3 frames | -0.00 (-0.00 – 0.00) | -0.00 (-0.01 – -0.00) | 0.63 |
| share of the energy above 3 kHz (20 ms) | 0.01 (0.00 – 0.05) | 0.01 (0.00 – 0.10) | 0.54 |
| ms since Silero's last frame at 0.9 or more | 16 (8 – 24) | 18 (10 – 28) | 0.56 |
| ms since the level was last within 10 dB of the speaker | 30 (20 – 40) | 40 (30 – 80) | 0.68 |

A gate that never opens leaves the speech-off to Silero's first unvoiced frame. That is the ceiling: it removes 94.2 % of the early pauses (in the rest Silero itself stops before the labelled end) and leaves a lead of -4 ms.

Gates chosen to remove 50 % of the early pauses on half of the speakers; numbers from the other half (ungated there: lead 94 ms, earlier than Silero in 87.9 %). Features that cannot reach the target are missing:

| `pause` is reported at the first quiet block where | early pauses removed | true stops delayed | … by, median ms | mean delay over all true stops, ms | speech-off lead left, median ms | earlier than Silero in | LibriSpeech clean: removed / lead ms | LibriSpeech other: removed / lead ms | AMI headset: removed / lead ms |
|---|---|---|---|---|---|---|---|---|---|
| best pair: level above the learned room, dB ≤ 8.9 and level under the speaker's level, dB ≥ 20.8 | 48.1 % | 75.2 % | 20 | 29 | 68 | 79.0 % | 50.9 % / 70 | 49.2 % / 68 | 45.2 % / 62 |
| level above the learned room, dB ≤ 8.07 | 47.6 % | 64.5 % | 30 | 25 | 68 | 79.6 % | 47.4 % / 70 | 47.7 % / 69 | 47.6 % / 60 |
| level under the speaker's level, dB ≥ 27.2 | 53.4 % | 75.3 % | 30 | 35 | 66 | 73.4 % | 57.9 % / 62 | 49.2 % / 72 | 53.6 % / 52 |
| level under the quiet threshold, dB ≥ 7.08 | 56.3 % | 93.3 % | 20 | 37 | 62 | 74.2 % | 63.2 % / 56 | 61.5 % / 64 | 47.6 % / 66 |
| its change over the last 3 frames ≤ -0.000966 | 53.9 % | 64.4 % | 40 | 29 | 60 | 87.1 % | 66.7 % / 58 | 60.0 % / 60 | 40.5 % / 64 |
| ms since the level was last within 10 dB of the speaker ≥ 90 | 64.1 % | 78.8 % | 50 | 40 | 48 | 84.7 % | 64.9 % / 44 | 69.2 % / 52 | 59.5 % / 43 |
| level change over the last 50 ms, dB ≥ -8.54 | 50.0 % | 81.9 % | 50 | 42 | 48 | 81.8 % | 45.6 % / 44 | 69.2 % / 46 | 38.1 % / 52 |
| quiet blocks in a row so far (a plain confirmation delay) ≥ 5 | 59.2 % | 99.9 % | 40 | 46 | 46 | 78.0 % | 56.1 % / 42 | 75.4 % / 44 | 48.8 % / 54 |
| level change over the last 30 ms, dB ≥ -2.61 | 54.4 % | 96.2 % | 40 | 46 | 42 | 78.6 % | 54.4 % / 36 | 73.8 % / 42 | 39.3 % / 54 |
| ms since Silero's last frame at 0.9 or more ≥ 30 | 59.7 % | 78.7 % | 60 | 46 | 34 | 81.8 % | 68.4 % / 32 | 67.7 % / 34 | 47.6 % / 34 |
| Silero's probability (latest frame) ≤ 0.992 | 68.0 % | 71.3 % | 60 | 42 | 32 | 85.4 % | 84.2 % / 30 | 78.5 % / 32 | 48.8 % / 60 |
| share of the energy above 3 kHz (20 ms) ≥ 0.116 | 52.4 % | 75.5 % | 80 | 56 | 0 | 53.4 % | 43.9 % / 30 | 58.5 % / 24 | 53.6 % / 0 |

Gates chosen to remove 75 % of the early pauses on half of the speakers; numbers from the other half (ungated there: lead 94 ms, earlier than Silero in 87.9 %). Features that cannot reach the target are missing:

| `pause` is reported at the first quiet block where | early pauses removed | true stops delayed | … by, median ms | mean delay over all true stops, ms | speech-off lead left, median ms | earlier than Silero in | LibriSpeech clean: removed / lead ms | LibriSpeech other: removed / lead ms | AMI headset: removed / lead ms |
|---|---|---|---|---|---|---|---|---|---|
| best pair: level above the learned room, dB ≤ 8.9 and its change over the last 3 frames ≤ -0.00435 | 76.2 % | 87.0 % | 60 | 52 | 30 | 78.1 % | 84.2 % / 30 | 78.5 % / 30 | 69.0 % / 30 |
| its change over the last 3 frames ≤ -0.0347 | 80.1 % | 86.9 % | 70 | 59 | 28 | 77.6 % | 87.7 % / 26 | 87.7 % / 28 | 69.0 % / 30 |
| level above the learned room, dB ≤ 2.17 | 74.3 % | 84.3 % | 60 | 53 | 26 | 63.0 % | 77.2 % / 40 | 73.8 % / 34 | 72.6 % / 0 |
| Silero's probability (latest frame) ≤ 0.915 | 82.0 % | 84.0 % | 80 | 64 | 26 | 69.2 % | 91.2 % / 24 | 90.8 % / 26 | 69.0 % / 29 |
| ms since Silero's last frame at 0.9 or more ≥ 44 | 85.9 % | 85.8 % | 80 | 73 | 14 | 65.6 % | 94.7 % / 12 | 96.9 % / 14 | 71.4 % / 16 |
| quiet blocks in a row so far (a plain confirmation delay) ≥ 8 | 80.6 % | 99.9 % | 70 | 72 | 12 | 63.4 % | 82.5 % / 8 | 86.2 % / 12 | 75.0 % / 20 |
| ms since the level was last within 10 dB of the speaker ≥ 130 | 81.6 % | 87.0 % | 80 | 69 | 8 | 60.4 % | 89.5 % / 6 | 86.2 % / 14 | 72.6 % / 4 |
| level under the speaker's level, dB ≥ 32.6 | 79.6 % | 94.2 % | 60 | 61 | 0 | 53.4 % | 86.0 % / 2 | 75.4 % / 34 | 78.6 % / 0 |
| level under the quiet threshold, dB ≥ 12 | 77.2 % | 99.5 % | 60 | 66 | 0 | 52.1 % | 86.0 % / 0 | 76.9 % / 2 | 71.4 % / 0 |
| level change over the last 50 ms, dB ≥ 0.165 | 78.2 % | 99.7 % | 80 | 78 | 0 | 46.4 % | 77.2 % / -0 | 90.8 % / -0 | 69.0 % / 0 |
| level change over the last 30 ms, dB ≥ 2.26 | 74.3 % | 100.0 % | 80 | 77 | -0 | 43.0 % | 75.4 % / -0 | 83.1 % / -2 | 66.7 % / 0 |
| share of the energy above 3 kHz (20 ms) ≤ 0.00116 | 69.4 % | 84.5 % | 100 | 79 | -4 | 26.1 % | 75.4 % / -4 | 69.2 % / -4 | 65.5 % / -4 |

Gates chosen to remove 90 % of the early pauses on half of the speakers; numbers from the other half (ungated there: lead 94 ms, earlier than Silero in 87.9 %). Features that cannot reach the target are missing:

| `pause` is reported at the first quiet block where | early pauses removed | true stops delayed | … by, median ms | mean delay over all true stops, ms | speech-off lead left, median ms | earlier than Silero in | LibriSpeech clean: removed / lead ms | LibriSpeech other: removed / lead ms | AMI headset: removed / lead ms |
|---|---|---|---|---|---|---|---|---|---|
| best pair: level under the quiet threshold, dB ≥ 15.4 and ms since the level was last within 10 dB of the speaker ≥ 130 | 94.2 % | 100.0 % | 90 | 92 | -4 | 24.5 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| its change over the last 3 frames ≤ -0.505 | 94.2 % | 98.6 % | 100 | 94 | -4 | 11.7 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| ms since the level was last within 10 dB of the speaker ≥ 270 | 93.7 % | 96.4 % | 100 | 94 | -4 | 13.9 % | 98.2 % / -4 | 98.5 % / -4 | 86.9 % / -4 |
| Silero's probability (latest frame) ≤ 0.463 | 92.7 % | 93.1 % | 100 | 94 | -4 | 11.4 % | 96.5 % / -4 | 100.0 % / -4 | 84.5 % / -4 |
| quiet blocks in a row so far (a plain confirmation delay) ≥ 17 | 94.2 % | 99.9 % | 100 | 96 | -4 | 12.4 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| level under the quiet threshold, dB ≤ -4.61 | 94.2 % | 98.5 % | 100 | 96 | -4 | 11.1 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| level change over the last 50 ms, dB ≥ 14.7 | 94.2 % | 100.0 % | 100 | 96 | -4 | 10.7 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| share of the energy above 3 kHz (20 ms) ≤ 0 | 94.2 % | 100.0 % | 100 | 97 | -4 | 10.3 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| level above the learned room, dB ≤ -69.4 | 94.2 % | 100.0 % | 100 | 97 | -4 | 10.2 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| level under the speaker's level, dB ≥ 76.5 | 94.2 % | 100.0 % | 100 | 97 | -4 | 10.2 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |
| level change over the last 30 ms, dB ≤ -57.4 | 94.2 % | 100.0 % | 100 | 97 | -4 | 10.2 % | 98.2 % / -4 | 100.0 % / -4 | 86.9 % / -4 |

### 3. Rule candidates, replayed

LibriSpeech clean:

| Rule | turns | speech-off > 50 ms before the labelled end | > 100 ms | > 100 ms with sound to come | > 200 ms | speech-off lead, median ms | earlier than Silero in | false `pause` per s of speech |
|---|---|---|---|---|---|---|---|---|
| library (measured) | 1518 | 24.2 % | 9.5 % | 8.0 % | 1.6 % | 90 | 89.2 % | 1.66 |
| replay model, no rule | 1518 | 24.2 % | 9.5 % | 8.0 % | 1.6 % | 90 | 89.2 % | 1.66 |
| a: quiet = within 6 dB of the room where it is ≥ 20 dB under the speaker | 1517 | 10.2 % | 3.5 % | 2.4 % | 0.5 % | 66 | 80.8 % | 1.07 |
| a: quiet = within 9 dB of the room where it is ≥ 20 dB under the speaker | 1520 | 18.0 % | 7.0 % | 5.7 % | 1.2 % | 80 | 85.0 % | 1.28 |
| a: quiet = within 12 dB of the room where it is ≥ 20 dB under the speaker | 1522 | 25.8 % | 10.7 % | 9.4 % | 2.4 % | 92 | 88.5 % | 1.51 |
| a: quiet = within 6 dB of the room where it is ≥ 25 dB under the speaker | 1517 | 9.2 % | 2.9 % | 1.9 % | 0.4 % | 64 | 79.0 % | 1.08 |
| a: quiet = within 9 dB of the room where it is ≥ 25 dB under the speaker | 1518 | 14.8 % | 5.5 % | 4.3 % | 0.8 % | 76 | 83.0 % | 1.26 |
| a: quiet = within 12 dB of the room where it is ≥ 25 dB under the speaker | 1519 | 20.7 % | 7.8 % | 6.4 % | 1.3 % | 86 | 86.1 % | 1.45 |
| a: quiet = within 6 dB of the room where it is ≥ 30 dB under the speaker | 1517 | 11.4 % | 4.0 % | 3.1 % | 0.7 % | 66 | 79.2 % | 1.18 |
| a: quiet = within 9 dB of the room where it is ≥ 30 dB under the speaker | 1517 | 14.2 % | 5.3 % | 4.2 % | 0.8 % | 76 | 82.5 % | 1.30 |
| a: quiet = within 12 dB of the room where it is ≥ 30 dB under the speaker | 1517 | 17.9 % | 6.5 % | 5.2 % | 1.0 % | 82 | 85.3 % | 1.43 |
| b: second pause after 10 ms at ≥ 6 dB above the room | 1518 | 3.0 % | 0.6 % | 0.1 % | 0.0 % | 22 | 55.8 % | 2.80 |
| b: second pause after 20 ms at ≥ 6 dB above the room | 1518 | 4.5 % | 1.0 % | 0.3 % | 0.1 % | 52 | 67.8 % | 2.46 |
| b: second pause after 30 ms at ≥ 6 dB above the room | 1518 | 5.9 % | 1.4 % | 0.6 % | 0.1 % | 58 | 73.5 % | 2.34 |
| b: second pause after 10 ms at ≥ 10 dB above the room | 1518 | 7.8 % | 1.9 % | 0.9 % | 0.1 % | 66 | 73.0 % | 2.47 |
| b: second pause after 20 ms at ≥ 10 dB above the room | 1518 | 10.1 % | 2.8 % | 1.6 % | 0.3 % | 74 | 78.2 % | 2.22 |
| b: second pause after 30 ms at ≥ 10 dB above the room | 1518 | 12.6 % | 3.8 % | 2.5 % | 0.5 % | 76 | 81.2 % | 2.13 |
| b: second pause after 10 ms at ≥ 15 dB above the room | 1518 | 13.6 % | 3.8 % | 2.6 % | 0.5 % | 80 | 81.8 % | 2.12 |
| b: second pause after 20 ms at ≥ 15 dB above the room | 1518 | 16.3 % | 5.1 % | 3.7 % | 0.7 % | 82 | 84.3 % | 1.93 |
| b: second pause after 30 ms at ≥ 15 dB above the room | 1518 | 18.4 % | 6.0 % | 4.6 % | 0.8 % | 84 | 85.5 % | 1.86 |
| c: second pause when the count starts again after ≥ 10 ms over the threshold | 1518 | 11.3 % | 2.4 % | 1.1 % | 0.1 % | 72 | 67.7 % | 2.64 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 1518 | 14.4 % | 3.7 % | 2.2 % | 0.1 % | 82 | 76.0 % | 2.14 |
| c: second pause when the count starts again after ≥ 30 ms over the threshold | 1518 | 17.8 % | 5.3 % | 3.7 % | 0.5 % | 84 | 81.5 % | 1.93 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 10 ms | 1518 | 11.7 % | 2.5 % | 1.1 % | 0.1 % | 76 | 76.0 % | 2.33 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 1518 | 15.0 % | 3.9 % | 2.3 % | 0.1 % | 84 | 83.3 % | 1.98 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 30 ms | 1518 | 18.5 % | 5.3 % | 3.8 % | 0.5 % | 86 | 86.8 % | 1.84 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 10 ms | 1518 | 12.8 % | 3.0 % | 1.5 % | 0.2 % | 82 | 81.8 % | 2.20 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 20 ms | 1518 | 16.3 % | 4.3 % | 2.7 % | 0.5 % | 86 | 85.9 % | 1.93 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 30 ms | 1518 | 19.4 % | 5.7 % | 4.1 % | 0.7 % | 88 | 87.9 % | 1.81 |

LibriSpeech other:

| Rule | turns | speech-off > 50 ms before the labelled end | > 100 ms | > 100 ms with sound to come | > 200 ms | speech-off lead, median ms | earlier than Silero in | false `pause` per s of speech |
|---|---|---|---|---|---|---|---|---|
| library (measured) | 2239 | 19.7 % | 8.1 % | 6.5 % | 0.8 % | 88 | 88.5 % | 1.43 |
| replay model, no rule | 2239 | 19.8 % | 8.2 % | 6.6 % | 0.8 % | 89 | 88.5 % | 1.43 |
| a: quiet = within 6 dB of the room where it is ≥ 20 dB under the speaker | 2236 | 6.2 % | 2.4 % | 1.5 % | 0.2 % | 52 | 74.8 % | 0.82 |
| a: quiet = within 9 dB of the room where it is ≥ 20 dB under the speaker | 2236 | 10.9 % | 4.6 % | 3.6 % | 0.4 % | 66 | 81.5 % | 1.01 |
| a: quiet = within 12 dB of the room where it is ≥ 20 dB under the speaker | 2237 | 16.5 % | 7.5 % | 6.3 % | 1.3 % | 78 | 85.4 % | 1.20 |
| a: quiet = within 6 dB of the room where it is ≥ 25 dB under the speaker | 2236 | 5.8 % | 2.1 % | 1.3 % | 0.1 % | 50 | 73.7 % | 0.83 |
| a: quiet = within 9 dB of the room where it is ≥ 25 dB under the speaker | 2236 | 9.3 % | 3.7 % | 2.8 % | 0.3 % | 62 | 80.2 % | 0.98 |
| a: quiet = within 12 dB of the room where it is ≥ 25 dB under the speaker | 2237 | 14.0 % | 5.8 % | 4.6 % | 0.7 % | 74 | 84.0 % | 1.15 |
| a: quiet = within 6 dB of the room where it is ≥ 30 dB under the speaker | 2236 | 6.9 % | 2.7 % | 2.1 % | 0.3 % | 50 | 73.8 % | 0.90 |
| a: quiet = within 9 dB of the room where it is ≥ 30 dB under the speaker | 2236 | 9.0 % | 3.5 % | 2.6 % | 0.3 % | 62 | 80.1 % | 1.01 |
| a: quiet = within 12 dB of the room where it is ≥ 30 dB under the speaker | 2236 | 12.3 % | 5.0 % | 3.8 % | 0.4 % | 72 | 83.5 % | 1.13 |
| b: second pause after 10 ms at ≥ 6 dB above the room | 2239 | 1.5 % | 0.3 % | 0.0 % | 0.0 % | 2 | 50.2 % | 2.60 |
| b: second pause after 20 ms at ≥ 6 dB above the room | 2239 | 2.6 % | 0.6 % | 0.1 % | 0.0 % | 28 | 60.5 % | 2.28 |
| b: second pause after 30 ms at ≥ 6 dB above the room | 2239 | 3.5 % | 1.0 % | 0.3 % | 0.0 % | 38 | 65.5 % | 2.17 |
| b: second pause after 10 ms at ≥ 10 dB above the room | 2239 | 4.6 % | 0.9 % | 0.3 % | 0.0 % | 46 | 67.5 % | 2.28 |
| b: second pause after 20 ms at ≥ 10 dB above the room | 2239 | 5.9 % | 1.3 % | 0.7 % | 0.1 % | 58 | 73.5 % | 2.07 |
| b: second pause after 30 ms at ≥ 10 dB above the room | 2239 | 7.1 % | 1.8 % | 1.2 % | 0.1 % | 62 | 76.8 % | 1.98 |
| b: second pause after 10 ms at ≥ 15 dB above the room | 2239 | 9.1 % | 2.1 % | 1.3 % | 0.1 % | 72 | 78.7 % | 1.93 |
| b: second pause after 20 ms at ≥ 15 dB above the room | 2239 | 11.0 % | 2.9 % | 2.1 % | 0.3 % | 76 | 82.0 % | 1.77 |
| b: second pause after 30 ms at ≥ 15 dB above the room | 2239 | 12.4 % | 3.6 % | 2.8 % | 0.4 % | 78 | 83.2 % | 1.70 |
| c: second pause when the count starts again after ≥ 10 ms over the threshold | 2239 | 9.3 % | 1.3 % | 0.5 % | 0.0 % | 68 | 69.1 % | 2.29 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 2239 | 11.3 % | 2.1 % | 1.1 % | 0.1 % | 78 | 76.8 % | 1.86 |
| c: second pause when the count starts again after ≥ 30 ms over the threshold | 2239 | 13.7 % | 3.2 % | 2.0 % | 0.2 % | 82 | 81.1 % | 1.67 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 10 ms | 2239 | 9.5 % | 1.3 % | 0.5 % | 0.0 % | 74 | 77.0 % | 2.02 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 2239 | 11.8 % | 2.2 % | 1.2 % | 0.1 % | 82 | 84.1 % | 1.71 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 30 ms | 2239 | 14.2 % | 3.3 % | 2.1 % | 0.2 % | 84 | 86.3 % | 1.59 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 10 ms | 2239 | 10.4 % | 1.7 % | 0.7 % | 0.0 % | 78 | 81.8 % | 1.89 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 20 ms | 2239 | 12.5 % | 2.5 % | 1.5 % | 0.2 % | 82 | 85.4 % | 1.66 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 30 ms | 2239 | 14.6 % | 3.6 % | 2.4 % | 0.2 % | 86 | 86.8 % | 1.57 |

AMI headset:

| Rule | turns | speech-off > 50 ms before the labelled end | > 100 ms | > 100 ms with sound to come | > 200 ms | speech-off lead, median ms | earlier than Silero in | false `pause` per s of speech |
|---|---|---|---|---|---|---|---|---|
| library (measured) | 2157 | 21.2 % | 12.0 % | 11.1 % | 3.2 % | 102 | 78.9 % | 1.81 |
| replay model, no rule | 2157 | 21.2 % | 12.0 % | 11.1 % | 3.2 % | 102 | 78.9 % | 1.81 |
| a: quiet = within 6 dB of the room where it is ≥ 20 dB under the speaker | 2180 | 12.7 % | 7.8 % | 6.4 % | 1.4 % | 40 | 59.3 % | 1.16 |
| a: quiet = within 9 dB of the room where it is ≥ 20 dB under the speaker | 2179 | 15.1 % | 9.3 % | 8.0 % | 1.7 % | 66 | 66.4 % | 1.29 |
| a: quiet = within 12 dB of the room where it is ≥ 20 dB under the speaker | 2181 | 17.9 % | 11.3 % | 10.1 % | 2.5 % | 84 | 71.9 % | 1.44 |
| a: quiet = within 6 dB of the room where it is ≥ 25 dB under the speaker | 2180 | 12.5 % | 7.8 % | 6.3 % | 1.3 % | 38 | 58.5 % | 1.17 |
| a: quiet = within 9 dB of the room where it is ≥ 25 dB under the speaker | 2177 | 14.4 % | 8.7 % | 7.4 % | 1.6 % | 60 | 64.5 % | 1.29 |
| a: quiet = within 12 dB of the room where it is ≥ 25 dB under the speaker | 2178 | 16.9 % | 10.2 % | 9.0 % | 2.2 % | 80 | 69.7 % | 1.44 |
| a: quiet = within 6 dB of the room where it is ≥ 30 dB under the speaker | 2175 | 13.5 % | 8.4 % | 7.0 % | 1.3 % | 46 | 60.2 % | 1.21 |
| a: quiet = within 9 dB of the room where it is ≥ 30 dB under the speaker | 2177 | 14.7 % | 8.9 % | 7.7 % | 1.6 % | 60 | 64.5 % | 1.31 |
| a: quiet = within 12 dB of the room where it is ≥ 30 dB under the speaker | 2176 | 16.5 % | 9.9 % | 8.7 % | 2.2 % | 76 | 68.9 % | 1.43 |
| b: second pause after 10 ms at ≥ 6 dB above the room | 2157 | 3.8 % | 2.0 % | 1.2 % | 0.4 % | -14 | 38.6 % | 3.33 |
| b: second pause after 20 ms at ≥ 6 dB above the room | 2157 | 4.6 % | 2.5 % | 1.6 % | 0.6 % | 0 | 44.7 % | 2.97 |
| b: second pause after 30 ms at ≥ 6 dB above the room | 2157 | 5.3 % | 2.8 % | 1.8 % | 0.6 % | 0 | 48.7 % | 2.82 |
| b: second pause after 10 ms at ≥ 10 dB above the room | 2157 | 5.8 % | 2.9 % | 2.0 % | 0.6 % | 22 | 54.3 % | 3.04 |
| b: second pause after 20 ms at ≥ 10 dB above the room | 2157 | 7.2 % | 3.5 % | 2.5 % | 0.9 % | 36 | 58.0 % | 2.75 |
| b: second pause after 30 ms at ≥ 10 dB above the room | 2157 | 8.8 % | 4.5 % | 3.2 % | 1.1 % | 50 | 60.7 % | 2.61 |
| b: second pause after 10 ms at ≥ 15 dB above the room | 2157 | 10.2 % | 5.4 % | 4.6 % | 1.3 % | 72 | 66.2 % | 2.66 |
| b: second pause after 20 ms at ≥ 15 dB above the room | 2157 | 11.9 % | 6.6 % | 5.6 % | 1.5 % | 80 | 69.2 % | 2.41 |
| b: second pause after 30 ms at ≥ 15 dB above the room | 2157 | 13.6 % | 7.6 % | 6.4 % | 1.7 % | 86 | 71.2 % | 2.28 |
| c: second pause when the count starts again after ≥ 10 ms over the threshold | 2157 | 10.7 % | 5.1 % | 4.6 % | 1.2 % | 74 | 63.6 % | 2.93 |
| c: second pause when the count starts again after ≥ 20 ms over the threshold | 2157 | 12.7 % | 6.5 % | 5.7 % | 1.7 % | 86 | 68.5 % | 2.41 |
| c: second pause when the count starts again after ≥ 30 ms over the threshold | 2157 | 14.6 % | 7.6 % | 6.6 % | 1.9 % | 92 | 71.9 % | 2.17 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 10 ms | 2157 | 10.9 % | 5.3 % | 4.7 % | 1.3 % | 78 | 66.4 % | 2.73 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 20 ms | 2157 | 13.0 % | 6.7 % | 5.8 % | 1.7 % | 90 | 71.2 % | 2.31 |
| d: as c, counting only blocks that are also ≥ 6 dB above the room, ≥ 30 ms | 2157 | 15.0 % | 7.8 % | 6.8 % | 2.0 % | 94 | 73.9 % | 2.12 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 10 ms | 2157 | 11.3 % | 5.6 % | 4.9 % | 1.4 % | 82 | 69.0 % | 2.64 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 20 ms | 2157 | 13.6 % | 6.9 % | 6.0 % | 1.9 % | 90 | 72.4 % | 2.27 |
| d: as c, counting only blocks that are also ≥ 10 dB above the room, ≥ 30 ms | 2157 | 15.7 % | 8.1 % | 7.0 % | 2.1 % | 94 | 74.6 % | 2.09 |

### 4. A few speakers or general

| Condition | speakers | with at least one early pause | speakers with ≥ 8 turns | their early rate, %: p25 / median / p75 | share of all early pauses from the top tenth of speakers |
|---|---|---|---|---|---|
| LibriSpeech clean | 79 | 63.3 % | 57 | 4.2 / 8.0 / 11.5 (max 30.0) | 33.9 % |
| LibriSpeech other | 64 | 65.6 % | 53 | 1.1 / 5.9 / 8.7 (max 26.8) | 46.6 % |
| AMI headset | 71 | 78.9 % | 66 | 4.0 / 9.1 / 14.0 (max 36.4) | 32.2 % |

Early rate by how the last word is spelled at its end:

| Last word ends in | LibriSpeech clean | LibriSpeech other | AMI headset |
|---|---|---|---|
| fricative (s, z, f, th, sh, ch, x, ce, se, ge) | 8.0 % of 388 | 4.3 % of 486 | 15.0 % of 293 |
| nasal or liquid (m, n, ng, l, r) | 2.0 % of 460 | 2.4 % of 670 | 6.4 % of 593 |
| stop (t, d, k, p, g, b, ed, te, de, ke) | 16.1 % of 422 | 15.2 % of 684 | 17.0 % of 452 |
| vowel or other | 5.2 % of 248 | 1.3 % of 399 | 9.8 % of 819 |

### Example clips

A: the rest of the word stayed under the quiet threshold (LibriSpeech):

- `libri_other/1686-142278-0054_0` "yes": `pause` 200 ms before the labelled end, 180 ms audible after it, 8 dB above the room, 25 dB under the speaker
- `libri_clean/1089-134691-0018_1` "again": `pause` 200 ms before the labelled end, 160 ms audible after it, 9 dB above the room, 25 dB under the speaker
- `libri_clean/3575-170457-0046_0` "haworth": `pause` 190 ms before the labelled end, 160 ms audible after it, 9 dB above the room, 17 dB under the speaker
- `libri_clean/1188-133604-0023_7` "earth": `pause` 180 ms before the labelled end, 150 ms audible after it, 11 dB above the room, 22 dB under the speaker
- `libri_clean/1988-24833-0010_0` "cares": `pause` 170 ms before the labelled end, 130 ms audible after it, 7 dB above the room, 28 dB under the speaker
- `libri_clean/5105-28240-0013_0` "yourself": `pause` 110 ms before the labelled end, 110 ms audible after it, 8 dB above the room, 20 dB under the speaker
- `libri_clean/174-84280-0013_1` "of": `pause` 130 ms before the labelled end, 100 ms audible after it, 8 dB above the room, 24 dB under the speaker
- `libri_other/1686-142278-0059_1` "margaret": `pause` 110 ms before the labelled end, 80 ms audible after it, 5 dB above the room, 24 dB under the speaker
- `libri_clean/2902-9008-0009_0` "earth": `pause` 110 ms before the labelled end, 80 ms audible after it, 13 dB above the room, 23 dB under the speaker
- `libri_clean/8842-302203-0004_0` "death": `pause` 180 ms before the labelled end, 80 ms audible after it, 8 dB above the room, 23 dB under the speaker

A: the rest of the word stayed under the quiet threshold (AMI headset):

- `ami_headset_xtalk/TS3012a_A_004223` "yeah": `pause` 210 ms before the labelled end, 210 ms audible after it, 11 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3008a_D_010724` "up": `pause` 190 ms before the labelled end, 190 ms audible after it, 9 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3012a_C_050774` "stripes": `pause` 230 ms before the labelled end, 190 ms audible after it, 9 dB above the room, 13 dB under the speaker
- `ami_headset_xtalk/ES2015a_D_049187` "shapely": `pause` 210 ms before the labelled end, 180 ms audible after it, 1 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3010a_A_066835` "that": `pause` 180 ms before the labelled end, 180 ms audible after it, 12 dB above the room, 30 dB under the speaker
- `ami_headset_xtalk/ES2015a_A_028462` "control": `pause` 230 ms before the labelled end, 170 ms audible after it, -0 dB above the room, 27 dB under the speaker
- `ami_headset_xtalk/TS3008a_A_072976` "next": `pause` 190 ms before the labelled end, 170 ms audible after it, 9 dB above the room, 23 dB under the speaker
- `ami_headset_xtalk/ES2002a_B_052321` "um": `pause` 210 ms before the labelled end, 160 ms audible after it, 19 dB above the room, 28 dB under the speaker
- `ami_headset_xtalk/ES2006d_A_123844` "uh": `pause` 170 ms before the labelled end, 160 ms audible after it, 10 dB above the room, 32 dB under the speaker
- `ami_headset_xtalk/TS3005a_C_073542` "world": `pause` 200 ms before the labelled end, 160 ms audible after it, 15 dB above the room, 23 dB under the speaker

B: the rest of the word crossed the threshold, no new `pause` (LibriSpeech):

- `libri_other/3528-168669-0114_0` "possible": `pause` 230 ms before the labelled end, 230 ms audible after it, 31 dB above the room, 20 dB under the speaker
- `libri_clean/5683-32879-0021_0` "moved": `pause` 300 ms before the labelled end, 220 ms audible after it, 15 dB above the room, 20 dB under the speaker
- `libri_clean/2035-147961-0027_0` "yes": `pause` 210 ms before the labelled end, 210 ms audible after it, 13 dB above the room, 17 dB under the speaker
- `libri_other/1686-142278-0007_0` "evening": `pause` 310 ms before the labelled end, 200 ms audible after it, 13 dB above the room, 16 dB under the speaker
- `libri_other/2414-159411-0010_1` "judgment": `pause` 200 ms before the labelled end, 200 ms audible after it, 16 dB above the room, 24 dB under the speaker
- `libri_clean/2035-147961-0025_0` "asked": `pause` 270 ms before the labelled end, 190 ms audible after it, 12 dB above the room, 21 dB under the speaker
- `libri_clean/2035-147961-0028_0` "enough": `pause` 210 ms before the labelled end, 190 ms audible after it, 15 dB above the room, 22 dB under the speaker
- `libri_other/3660-172182-0001_0` "arthur": `pause` 270 ms before the labelled end, 190 ms audible after it, 13 dB above the room, 17 dB under the speaker
- `libri_clean/6241-61943-0027_1` "unhappy": `pause` 280 ms before the labelled end, 190 ms audible after it, 10 dB above the room, 19 dB under the speaker
- `libri_clean/1995-1826-0023_0` "yams": `pause` 210 ms before the labelled end, 180 ms audible after it, 12 dB above the room, 23 dB under the speaker

B: the rest of the word crossed the threshold, no new `pause` (AMI headset):

- `ami_headset_xtalk/ES2010a_B_043002` "everything": `pause` 330 ms before the labelled end, 330 ms audible after it, 21 dB above the room, 21 dB under the speaker
- `ami_headset_xtalk/TS3008a_B_030071` "yes": `pause` 300 ms before the labelled end, 300 ms audible after it, 16 dB above the room, 19 dB under the speaker
- `ami_headset/TS3010a_A_016208` "colour": `pause` 280 ms before the labelled end, 280 ms audible after it, 14 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/ES2006d_C_047726` "mm-hmm": `pause` 270 ms before the labelled end, 270 ms audible after it, 26 dB above the room, 21 dB under the speaker
- `ami_headset_xtalk/TS3006b_D_041230` "a": `pause` 270 ms before the labelled end, 270 ms audible after it, 13 dB above the room, 29 dB under the speaker
- `ami_headset_xtalk/ES2015a_B_109194` "yeah": `pause` 260 ms before the labelled end, 260 ms audible after it, 10 dB above the room, 30 dB under the speaker
- `ami_headset/TS3006b_D_207414` "yeah": `pause` 260 ms before the labelled end, 260 ms audible after it, 17 dB above the room, 25 dB under the speaker
- `ami_headset_xtalk/ES2004c_C_116288` "yeah": `pause` 250 ms before the labelled end, 250 ms audible after it, 22 dB above the room, 19 dB under the speaker
- `ami_headset_xtalk/TS3006b_D_121574` "for": `pause` 260 ms before the labelled end, 250 ms audible after it, 8 dB above the room, 24 dB under the speaker
- `ami_headset_xtalk/TS3006b_D_191901` "so": `pause` 250 ms before the labelled end, 250 ms audible after it, 11 dB above the room, 28 dB under the speaker

