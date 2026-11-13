# Quiet, noisy and phone input

What happens when the speaker is quiet, the line is narrowband or noise sits
close under the speech: where `pause` still leads, where it does not, and
what to set. The summary is in the [README](../README.md); the other
measurements are in [results.md](results.md).

- [The quiet floor follows the speaker](#the-quiet-floor-follows-the-speaker) (since 0.0.5)
- [The same rows now](#the-same-rows-now)
- [Steady noise close under the speaker](#steady-noise-close-under-the-speaker-after-005)
- [Amplifying quiet input for Silero](#amplifying-quiet-input-for-silero-optional)
- [Short backchannels](#short-backchannels)
- [DC offset](#dc-offset)

## The quiet floor follows the speaker

Up to 0.0.4, "quiet" had a fixed floor (`quiet_rms`, about −54 dBFS). For a
speaker around −32 dBFS that is fine. For a quiet one (a phone line recorded at
source level, a distant microphone) the floor sits only a few dB under the
speech, their soft word endings counted as silence, and 0.24 s later the
segment ended while they were still talking. Since 0.0.5 the floor follows
the speaker: it is `quiet_rms` or 20 dB below the speaker's own level,
whichever is lower (`floor_margin_db`).

The same 3,401 turns, attenuated and sent through a simulated phone line
(300–3400 Hz, 8 kHz, μ-law, line noise). "Ended while talking" counts turns
with an end that the same detector does not make at the recorded level and
that is followed by more speech. "Silero alone" is the same VAD with the quiet
clock off, on the same audio.

| Speech level | Ended while talking: 0.0.4 | 0.0.5 | Silero alone | Lost ≥ 100 ms of speech: 0.0.4 | 0.0.5 | Silero alone | Speech-off lead 0.0.5 |
|---|---|---|---|---|---|---|---|
| −32 dBFS (as recorded) | – | – | – | 0 | 0 | 0 | 100 ms |
| −38 dBFS | 361 (11 %) | 36 | 58 | 13 | 8 | 8 | 108 ms |
| −42 dBFS | 609 (18 %) | 37 | 97 | 26 | 10 | 10 | 116 ms |
| −52 dBFS | 1,473 (43 %) | 215 | 155 | 868 | 26 | 23 | 132 ms |
| −62 dBFS | 1,832 (54 %) | 269 | 303 | 1,778 | 419 | 415 | 124 ms |
| −47 dBFS in room noise at −62 dBFS | 746 (22 %) | 125 | 312 | 295 | 93 | 93 | 0 ms |
| phone line, −38 dBFS | 285 | 416 | 391 | 266 | 234 | 229 | 112 ms |
| phone line, −48 dBFS | 667 | 265 | 275 | 311 | 301 | 294 | 0 ms |

What this says, and what it doesn't:

- **Quiet speakers are no longer cut.** Ends while the speaker talks and lost
  speech are now where Silero alone has them. In all of these runs RealtimeVAD's
  last end was never later than Silero's.
- **A noisy quiet line had no lead in 0.0.5.** When the noise is less than
  about 20 dB below the speaker (the room-noise row, and the quiet phone line
  with its line noise), no level can be called quiet without calling soft
  speech quiet. 0.0.5 then ends where Silero does: correct, and no faster.
  The next section is about that.
- **Very quiet and narrowband speech is a Silero limit.** At −62 dBFS, and on
  the phone line, Silero itself is unsure. Speech it never scores 0.9 used to
  start no segment at all (476 of the phone-line turns, 1,264 at −62 dBFS).
  The soft start halves that (186 and 380); Silero at its own 0.5 threshold
  still hears more (139 and 353). More turns found also means more ends counted
  in the phone rows. Amplifying such input before the VAD helps Silero most.
- **Borderline endings.** Ends with speech of the original recording inside
  the 0.24 s that counted as quiet are rarer than in 0.0.4 by a factor of 10
  at −52 dBFS (1,970 turns then, 182 now), but restarting the count after a
  click added some back (141 without it). Ends followed by more speech and
  lost speech did not rise with it.
- The phone path is simulated, and the reference for "still talking" is the
  original wideband recording. One speaker, one microphone.

### The same rows now

After 0.0.5: `end` is Silero's by default, levels are measured without a DC
offset, steady noise is subtracted (next section), and a `pause` is repeated
when sound came back. Same 3,401 turns; "early end" is `VAD(early_end=True)`.

| Speech level | Ended while talking: default = Silero alone | early end | Lost ≥ 100 ms: default = Silero alone | Speech-off lead | earlier than Silero in | `pause` followed by more speech, per s | `end` earlier with early end, median |
|---|---|---|---|---|---|---|---|
| −32 dBFS (as recorded) | – | – | 0 | 90 ms | 90 % | 1.85 | 70 ms |
| −38 dBFS | 58 | 34 | 8 | 100 ms | 91 % | 1.93 | 86 ms |
| −42 dBFS | 97 | 34 | 10 | 104 ms | 92 % | 1.95 | 88 ms |
| −52 dBFS | 155 | 177 | 23 | 122 ms | 95 % | 1.99 | 96 ms |
| −62 dBFS | 303 | 268 | 415 | 98 ms | 76 % | 2.12 | 56 ms |
| −47 dBFS in room noise at −62 dBFS | 304 | 123 | 106 | 84 ms | 75 % | 1.71 | 0 ms |
| phone line, −38 dBFS | 392 | 416 | 227 | 92 ms | 86 % | 2.16 | 66 ms |
| phone line, −48 dBFS | 275 | 255 | 294 | 114 ms | 87 % | 1.86 | 36 ms |
| phone line, −58 dBFS | 312 | 263 | 910 | 98 ms | 79 % | 1.80 | 0 ms |

"Ended while talking" counts ends that the same detector does not make at the
recorded level. With the early end many ends move, so the count can be lower
than Silero's; it does not mean fewer ends inside utterances (see
[results.md](results.md#end)).

### Steady noise close under the speaker (after 0.0.5)

Where steady noise is too close under the speaker for a level threshold,
RealtimeVAD now subtracts the room's spectrum (learned from the frames Silero
does not call speech) from each 10 ms block's and asks whether what is left
is below the quiet floor. A soft word ending stands out of the noise in a few
bands and stays; the noise alone leaves next to nothing. It is used only where
the noise is within 3 dB of the quiet floor or above it, so clean input is
decided by the level exactly as before. The same 3,401 turns, before (0.0.5
plus the DC fix) and after, measured with the early end on and before the
repeated `pause` (the table above has today's defaults):

| Speech level | Speech-off lead | earlier than Silero in | `end` earlier than Silero alone in | Ended while talking: before / after / Silero alone | Lost ≥ 100 ms: before / after / Silero alone |
|---|---|---|---|---|---|
| −47 dBFS in room noise at −62 dBFS | 0 → 90 ms | 25 → 77 % | 12 % | 124 / 123 / 304 | 106 / 105 / 106 |
| phone line, −38 dBFS | 112 → 116 ms | 88 → 92 % | 80 % | 416 / 416 / 392 | 231 / 231 / 227 |
| phone line, −48 dBFS | 0 → 118 ms | 39 → 89 % | 65 % (median 36 ms) | 263 / 255 / 275 | 301 / 301 / 294 |
| phone line, −58 dBFS | 0 → 100 ms | 29 → 79 % | 25 % | 271 / 263 / 312 | 912 / 912 / 910 |

The clean rows of the table above did not move (at most 9 of 3,401 turns per
row end differently), and no end was later than Silero's. What it does not
do:

- The lead is in the speech-off (`pause`). In room noise the `end` itself is
  still mostly Silero's: single noise blocks keep breaking the 0.28 s count.
- Borderline endings rose with it: on the −48 dBFS phone line, ends with
  speech of the original recording inside the wait went from 486 to 750 turns,
  and Silero was confident again within 0.3 s after an end in 247 turns
  instead of 82. Ends followed by more speech and lost speech did not rise.
- It needs noise that is steady and at least about 11 dB under the speaker.
  Babble, music and street noise are not subtracted away; nothing changes
  there.
- The room noise and the phone line are simulated (pink noise, white line
  noise). All of the changes after 0.0.5 together cost 1.3 ms of CPU per
  second of audio: 7.0 ms before, 8.3 ms now (one thread, Silero included,
  idle machine).

What to set:

| If … | then |
|---|---|
| your input is quiet and clean | nothing; the lead stays |
| you would rather keep the old fixed floor | `VAD(floor_margin_db=None)` |
| soft starts open segments you don't want (background voices) | `VAD(soft_start_s=None)` |
| you want the `end` sooner and accept more ends inside an utterance | `VAD(early_end=True)` |
| you want `pause` earlier and accept that it is more often inside the last word | `VAD(pause_mode="balanced")` or `"fast"`, see the [README](../README.md#earlier-or-more-accurate-pause_mode) |
| you want one `pause` per stretch of speech, as in 0.0.5 | `VAD(repause_s=None)` |
| you want Silero only, no quiet clock at all | `VAD(quiet_rms=0)` |
| you want the level threshold only, as in 0.0.5 | `VAD(noise_subtract_db=None)` |
| your input is quiet or a phone line, and clean | try `VAD(normalize_to_dbfs=-30)`, see below |

```bash
python benchmarks/quiet_input.py <wav dir> out.json
```

### Amplifying quiet input for Silero (optional)

Silero is less sure about quiet speech, most of all on a phone line: it finds
fewer turns and drops out inside them. `VAD(normalize_to_dbfs=-30)` amplifies
the frames Silero sees, by up to 20 dB and never into clipping, until speech
sits at −30 dBFS. Levels, the quiet clock and the audio you get back use the
input as it came. Same turns as above:

| Speech level | No segment at all: off | on | Lost ≥ 100 ms of speech: off | on |
|---|---|---|---|---|
| −32 dBFS (as recorded) | 0 | 0 | 0 | 2 |
| −52 dBFS | 138 | 127 | 26 | 5 |
| −62 dBFS | 380 | 357 | 419 | 40 |
| −47 dBFS in room noise | 220 | 215 | 93 | 21 |
| phone line, −38 dBFS | 186 | 142 | 234 | 125 |
| phone line, −48 dBFS | 542 | 499 | 301 | 265 |
| phone line, −58 dBFS | 1,569 | 1,203 | 910 | 411 |

It is off by default because it is not free at normal level: 2 of 3,401 turns
lost more than 100 ms, speech-off was earlier than Silero in 96 % instead of
98 % of turns, and in one phone-line turn the end came later than Silero
alone. It needs Silero to score the speech 0.5 at least once before it acts,
so the first quiet words of a stream can still be missed. On noisy quiet input
the ends moved by more than the benchmark's "ended while talking" count can
judge, so that count is not given here; borderline endings did not rise.

### Short backchannels

"mm-hmm", "yeah", "okay": 200 real ones from meeting recordings (AMI corpus,
close microphone, 0.15–0.8 s long). Up to 0.0.4, 50 of them were not reported
as speech at all: Silero scored 35 between 0.5 and 0.9, below the start
threshold, and 15 below 0.5. With the soft start 25 are missed, 15 of them
Silero's own. A `start` comes a median 160 ms after the word begins. Whether
such a sound is a backchannel, a turn or an interruption is not a VAD
question and not answered here.

### DC offset

A DC offset in the input no longer hides the silence from the quiet clock.
Up to 0.0.5 it did: no early `pause`, no early end. Levels are now measured
behind a 20 Hz high-pass; the audio you get back and what Silero sees keep
the offset.

On the 3,401 turns with a constant offset of −40 dBFS added, the speech-off
lead was 0 ms (earlier than Silero in 6 % of turns) and is now 102 ms (99 %),
against 100 ms (98 %) without the offset. These three figures were measured
before the repeated `pause`.

What is left is Silero itself: it is a little less sure with an offset under
the speech, most of all under quiet speech (at −52 dBFS it finds no speech in
254 turns instead of 139). Remove a large offset before the VAD if you have
one.
