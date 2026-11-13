# Results

All measurements behind the numbers in the [README](../README.md): how early
`pause` comes, how often it is too early, what `pause_mode` and `early_end`
change, and how this version compares with 0.0.5. Quiet, noisy and phone
input has its own page: [quiet-and-noisy-input.md](quiet-and-noisy-input.md).

- [Speech-off (`pause`)](#speech-off-pause): one speaker, then 214 speakers
- [`end`](#end): the default and `early_end=True`

## Speech-off (`pause`)

One speaker: 3,153 turns from a live voice assistant (browser capture with
noise suppression), each with its trailing silence, default settings. Silero's
decisions are computed from the very same frame probabilities, so both see
identical input. "Speech-off" is RealtimeVAD's last `pause` before the turn
ended, against the first frame of Silero's final run below its threshold.

| RealtimeVAD speech-off earlier than … | median | p25 – p75 | earlier in |
|---|---|---|---|
| Silero, probability below 0.5 (its default threshold) | 92 ms | 72 – 112 ms | 92 % |
| Silero, probability below 0.35 (`VADIterator`'s end threshold) | 104 ms | 84 – 126 ms | 93 % |
| Silero `VADIterator` end, default 100 ms minimum silence | 238 ms | 214 – 294 ms | 99 % |

| Speech-off signals followed by more speech | per second of speech | turns without any |
|---|---|---|
| RealtimeVAD `pause` | 1.8 | 16 % |
| Silero, probability below 0.5 | 0.16 | 84 % |

These turns come from one capture chain: browser capture with noise
suppression and echo cancellation, mostly quiet rooms. In the 137 noisier
turns among them (room level above −70 dBFS) the median lead was 102–124 ms;
that sub-analysis is from the 3,118-turn run before the trailing-silence rule.

Many speakers, raw microphones: 7,080 clips from 214 speakers with word
timings (LibriSpeech read speech, AMI meeting headsets; no noise suppression),
see [benchmarks/RESULTS-word-timing.md](https://github.com/KoljaB/RealtimeVAD/blob/main/benchmarks/RESULTS-word-timing.md).
"Too early" means the last `pause` came more than 100 ms before the labelled
end of the last word while at least 30 ms of audible sound of that word were
still to come.

| Audio | Last `pause` too early: 0.0.5 | now | Median lead over Silero: 0.0.5 | now | Earlier than Silero in: 0.0.5 | now | `pause` followed by more speech, per s: 0.0.5 | now |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 7.6 % | 2.2 % | 88 ms | 84 ms | 84 % | 84 % | 1.64 | 1.96 |
| LibriSpeech other | 6.2 % | 1.3 % | 86 ms | 82 ms | 82 % | 85 % | 1.42 | 1.69 |
| AMI headset, speaker alone | 10.3 % | 5.7 % | 102 ms | 90 ms | 86 % | 80 % | 1.75 | 2.13 |
| AMI headset, others talking | 11.1 % | 8.9 % | 100 ms | 86 ms | 77 % | 69 % | 1.81 | 2.24 |
| AMI distant microphone | 2.6 % | 3.2 % | 0 ms | 0 ms | 30 % | 31 % | 0.60 | 0.85 |

What changed after 0.0.5, and what it cost:

- **A `pause` inside the last word is now followed by a better one.** The
  level often dips under the threshold before a final consonant or a weak
  last syllable and comes back for 20 to 60 ms. 0.0.5 reported nothing after
  that, so the only `pause` was early. Now 20 ms of sound that is clearly
  above the room lets the next quiet report a new `pause`. The first `pause`
  comes exactly as early as before.
- **The lead is measured on the last `pause`, so it fell:** from 100 ms
  (earlier than Silero in 98 % of turns) to 92 ms (92 %) on the one-speaker
  turns. Before the trailing-silence rule below it was 90 ms (90 %); the last
  `pause` moved later than 0.0.5's in 28 % of turns then, by a median 110 ms,
  and in a sample of those the sound that caused it was 20 to 38 dB above the
  room. Pauses followed by more speech rose from 1.4 to 1.8 per second.
- **Sound in the trailing silence takes no `pause` back.** A repeated `pause`
  (and a `resume`) needs sound while Silero still takes the audio for speech,
  or no longer ago than its lag (`repause_lag_s`, 0.128 s): a breath or a
  click once Silero has been counting silence for longer changes nothing, and
  the standing `pause` remains the last one. It halves the last pauses that
  land more than 96 ms after Silero's own speech-off (4.8 / 6.1 / 7.3 / 11.6 %
  of clips to 2.8 / 3.7 / 5.0 / 8.1 %) and cuts the clips whose last `pause`
  is taken back by a `resume` from 2.8 / 4.5 / 4.3 / 9.1 % to
  1.2 / 1.5 / 1.1 / 4.1 %. It costs 0.5 points of early last pauses on the
  AMI headsets with other people talking (8.4 to 8.9 %) and 0.7 on the distant
  microphone, and nothing on the other sets. `VAD(repause_lag_s=None)` is
  what 0.1.0 did before the rule.
- **What is left** are word endings that stay under the quiet threshold the
  whole time (about 40 % of the remaining early pauses on LibriSpeech, 70 % on
  the AMI headsets). On the headsets with other people talking, 9 % of the
  last pauses are still too early. Silero's own speech-off is that early in
  0.2 to 0.3 % of the read sentences: it is late, and rarely early.
- The 206 ms against `VADIterator` in earlier versions of this page was
  measured one 32 ms frame short; the same code gives 252 ms for 0.0.5 with
  the fixes on this page, 238 ms with the new `pause`.
- `VAD(repause_s=None)` gives one `pause` per stretch of speech, as in 0.0.5.

The optional `resume` event on the same clips (close microphones): 1.6 to 2.0
per second of speech; it comes a median 50 ms after its `pause` (p90 120 to
140 ms) and 20 ms after the first 10 ms block that was not quiet again (p90
50 ms). In 1.1 to 4.1 % of clips a `resume` follows the last `pause` and no
new `pause` comes before the `end`: the turn then ends with the speculative
work cancelled, and `end` is the signal to act on. Before the
trailing-silence rule it was 2.8 to 9.1 %.

The three `pause_mode` settings, and the one `pause` per stretch of 0.0.5 at
today's thresholds (`repause_s=None`), on the same clips and turns:

| | LibriSpeech clean | LibriSpeech other | AMI headset, alone | AMI headset, others talking | one speaker, 3,153 turns |
|---|---|---|---|---|---|
| **Median lead over Silero** | | | | | |
| `"accurate"` | 84 ms | 82 ms | 90 ms | 86 ms | 92 ms |
| `repause_s=None` * | 90 ms | 88 ms | 102 ms | 98 ms | 102 ms |
| `"balanced"` | 104 ms | 102 ms | 112 ms | 114 ms | 112 ms |
| `"fast"` | 120 ms | 118 ms | 124 ms | 126 ms | 130 ms |
| **Earlier than Silero in** | | | | | |
| `"accurate"` | 84 % | 85 % | 80 % | 69 % | 92 % |
| `repause_s=None` * | 89 % | 88 % | 85 % | 75 % | 98 % |
| `"balanced"` | 94 % | 93 % | 89 % | 83 % | 98 % |
| `"fast"` | 96 % | 95 % | 92 % | 87 % | 98 % |
| **Last `pause` too early** (more than 100 ms, sound to come) | | | | | |
| `"accurate"` | 2.2 % | 1.3 % | 5.7 % | 8.9 % | – |
| `repause_s=None` * | 8.0 % | 6.8 % | 11.4 % | 12.4 % | – |
| `"balanced"` | 13.9 % | 11.3 % | 14.4 % | 17.2 % | – |
| `"fast"` | 23.5 % | 17.7 % | 20.6 % | 21.4 % | – |
| **… more than 200 ms before the end of the word** | | | | | |
| `"accurate"` | 0.1 % | 0.1 % | 1.6 % | 3.4 % | – |
| `repause_s=None` * | 1.6 % | 0.8 % | 3.8 % | 4.5 % | – |
| `"balanced"` | 2.8 % | 1.7 % | 7.3 % | 6.9 % | – |
| `"fast"` | 5.5 % | 3.6 % | 10.3 % | 9.0 % | – |
| **`pause` followed by more speech, per s** | | | | | |
| `"accurate"` | 1.96 | 1.69 | 2.13 | 2.24 | 1.88 |
| `repause_s=None` * | 1.66 | 1.44 | 1.74 | 1.83 | 1.38 |
| `"balanced"` | 1.94 | 1.74 | 1.92 | 1.98 | 1.62 |
| `"fast"` | 2.23 | 2.02 | 2.05 | 2.12 | 1.70 |

\* The `repause_s=None` row is 0.0.5's behaviour at today's thresholds, measured
before the trailing-silence rule and without it, on the 3,118-turn set. The
quiet, noisy and phone-line figures in the bullets below and in
[quiet-and-noisy-input.md](quiet-and-noisy-input.md) are from before that rule too.

- The one-speaker turns have no word timings, so "too early" cannot be
  counted there.
- The one-speaker column counts the 2,675 turns that have both a last `pause`
  and a Silero speech-off, which is why its 1.88 pauses per second differ
  from the 1.8 (1.84) over all 3,153 turns in the first table.
- All of the lead that the default gave up after 0.0.5 went to the repeated
  `pause`: with `repause_s=None` the lead is that of 0.0.5 or 2 ms more. The
  noise subtraction and the DC fix cost none.
- The lead of the faster modes is partly the same thing as their price: a
  `pause` inside the last word is early against Silero too.
- Where noise or a phone line has taken lead away, the faster modes bring
  some back: at −47 dBFS in room noise 84 ms (earlier than Silero in 75 %)
  with `"accurate"`, 104 ms (82 %) with `"balanced"`, 130 ms (93 %) with
  `"fast"`; on the phone line at −38 dBFS 92, 134 and 152 ms.
- Ends while the speaker goes on, lost speech and missed starts are the same
  as with Silero alone in all three modes, on every row above and on the
  quiet, noisy and phone-line turns.
- A `resume` comes a median 110 to 130 ms after its `pause` in `"balanced"`
  and `"fast"` (50 ms in `"accurate"`), and takes back 89 to 95 % of the
  pauses that were followed by more speech in every mode.

How the modes work, and what they leave alone:

- **How it works:** `"accurate"` reports a new `pause` when sound came back
  after one, so the last one is rarely early. The other two report one
  `pause` per stretch of speech and count audio as quiet 3 dB (`"balanced"`)
  or 6 dB (`"fast"`) sooner, so the fading end of a word already counts.
- **What it does not change:** `start`, `end` and the segment's audio. `end`
  is where Silero alone ends the segment in every mode, and nothing more is
  lost or cut.
- **With `early_end=True`** the end follows the quiet clock and so moves with
  the mode. That combination ends more segments inside an utterance: with
  `"fast"` 35 to 41 % of the clips in the [`end`](#end) table, against 33 to
  38 % with `"accurate"` and 31 to 36 % for Silero alone. The segment's
  `speech_end` is the quiet clock's, so it is early as often as the `pause`.
- **With `resume_events`:** in `"balanced"` and `"fast"` a `resume` comes
  about 70 ms after sound is back, not 20 ms, because only sound long enough
  for a new `pause` gives one.
- **Detailed settings win.** The mode only sets `repause_s`, `quiet_rms`,
  `floor_margin_db`, `speech_margin_db` and `noise_margin_db`
  (`RealtimeVAD.vad.PAUSE_MODES`); any of them you pass yourself is used
  as given.

## `end`

By default `end` comes where Silero alone ends the segment (same thresholds,
same 0.24 s): on all of the audio above, and in every row of the tables in
[quiet-and-noisy-input.md](quiet-and-noisy-input.md), RealtimeVAD and Silero alone end the same number of turns while
the speaker goes on, lose the same speech, and no `end` is later. The
segment's `speech_end` is the quiet clock's, so you still learn where the
speech stopped.

Up to 0.0.5 the quiet clock also ended segments: 0.24 s after the audio went
quiet. Silero's probability lags about 0.1 s behind the audio, so that is an
end in pauses of 0.24 to 0.34 s that Silero alone rides through, the same
trade as a shorter silence setting. It is now the option
`VAD(early_end=True)`:

| `early_end=True` against Silero alone | `end` earlier in | median | p25 – p75 | later in | Clips with an `end` inside the utterance: early end | Silero alone |
|---|---|---|---|---|---|---|
| one speaker, 2,323 turns | 85 % | 70 ms | 45 – 94 ms | 0 | – | – |
| LibriSpeech clean | 60 % | 26 ms | 0 – 54 ms | 0 | 35.9 % | 33.1 % |
| LibriSpeech other | 62 % | 30 ms | 0 – 66 ms | 0 | 33.3 % | 31.0 % |
| AMI headset, speaker alone | 64 % | 48 ms | 0 – 92 ms | 0 | 38.1 % | 35.8 % |
| AMI headset, others talking | 52 % | 16 ms | 0 – 88 ms | 0 | 33.2 % | 30.9 % |

Silero's probabilities are the same with and without the quiet clock in all
7,080 clips, with the default `end` and with the early end, in all three
`pause_mode` settings. The clips are whole sentences cut at 0.24 s, so a third
of them contain a pause that long for any detector; the early end adds 2 to 3
points to that.

The one-speaker turns have no word timings; there the early end left speech
of the recording inside its 0.24 s wait in 49 of 3,401 turns. The one-speaker
row and this count were measured before two late fixes to the early end (see
the [changelog](../CHANGELOG.md)); the four rows under it were measured again
after them and did not move.

Reproduce with your own recordings (16 kHz mono WAVs, one turn each, with
trailing silence):

```bash
python benchmarks/speech_off_lead.py <wav dir> out.json
```
