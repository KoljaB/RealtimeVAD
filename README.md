# RealtimeVAD

*Know that the speaker stopped, about 90 ms before Silero does.*

[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) · [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) · [Live demo](https://github.com/KoljaB/RealtimeVAD/blob/main/demo/README.md) · [Spec](https://github.com/KoljaB/RealtimeVAD/blob/main/SPEC.md) · [LiveKit plugin](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/README.md)

A voice assistant can't do anything until it knows the user has stopped
talking. Speech recognition, the LLM call and the voice output all wait for
that one decision, and a normal voice activity detector takes a few hundred
milliseconds to make it.

RealtimeVAD makes it sooner. It keeps [Silero VAD](https://github.com/snakers4/silero-vad)
to decide what is speech, and adds a second, faster check: every 10 ms it
compares the audio level with the room's noise and with the speaker's own
level. The moment the audio goes quiet, you get an event, long before
Silero's probability has fallen.

- **Input:** 16 kHz mono audio in chunks of any size (float32, int16 or int16 bytes).
- **Output:** `start`, `pause` and `end` events, plus the audio of the running segment at any time.
- **Lead:** speech-off (`pause`) a median 90 ms before Silero's probability drops, and 238 ms before Silero's `VADIterator` reports the end, on 3,118 real voice-assistant turns from one speaker.
- **Other speakers, raw microphones:** 80 to 88 ms on 6,600 close-microphone clips of read and meeting speech from 214 speakers, earlier than Silero in 67 to 83 % of them; no lead on a distant table microphone. The last `pause` is more than 100 ms too early in 1 to 2 % of read sentences and 6 to 8 % of meeting clips.
- **`end`:** exactly where Silero alone ends the segment, with the time the speech really stopped. Ending sooner is an option (`early_end=True`).
- **Price:** `pause` also fires in short gaps between words, about twelve times as often as Silero.
- **Quiet input:** a quiet speaker's soft word endings are not silence; on a noisy quiet line the lead is gone instead ([details](#quiet-input-and-phone-audio)).
- **Runs on:** one CPU thread, far faster than realtime. Needs only numpy and onnxruntime; the Silero model is bundled, no torch.
- **Status:** alpha (0.0.5). The API may still change.

## Live demo

https://github.com/user-attachments/assets/d01900dd-9601-4f3d-8c88-db55e8bc83a1

The video was recorded with 0.0.5 and a synthetic voice and shows a median lead of 92 ms.

[demo/](https://github.com/KoljaB/RealtimeVAD/blob/main/demo/README.md) runs this page on your own microphone. It shows, turn by turn, how much sooner
RealtimeVAD knew that you stopped than Silero alone, on the same audio.

## Quick start

```bash
pip install realtimevad
```

```python
from RealtimeVAD import VAD

vad = VAD()
for chunk in microphone_chunks():
    for event in vad.feed(chunk):
        if event.kind == "start":
            stream_to_asr(vad.audio())       # includes the audio just before speech began
        elif event.kind == "pause":
            speculate(vad.audio())           # the speaker may be done; this isn't the end yet
        elif event.kind == "end":
            finalize(event.segment.audio)
```

## Events

| Event | Meaning | When |
|---|---|---|
| `start` | Speech has begun | After about 64 ms of confident speech, so clicks don't trigger it. Speech Silero is never confident about (a short "mm-hmm", a quiet phone voice) starts after about 0.13 s at half confidence, if it is clearly louder than the room |
| `pause` | The speaker may have stopped (speech-off) | Within about 10 ms of the audio going quiet. Again when sound came back and the audio goes quiet again |
| `end` | The segment is over | Where Silero alone ends it: after 0.24 s of Silero silence. With `VAD(early_end=True)` 0.24 s after speech-off if that is sooner. Never later than Silero alone |
| `resume` | The speech went on after a `pause` | About 20 ms after sound is back. Off by default: `VAD(resume_events=True)` |
| `onset` | Speech may have begun | At the first confident frame, before `start` confirms it. Off by default: `VAD(onset_events=True)` |

`vad.audio()` returns the running segment at any time, starting about 0.3 s
before the speech. `end` carries the finished segment with its audio, start
time, end time and the time the speech stopped (from the quiet clock, so
earlier and more exact than Silero's last speech frame).

If your assistant talks while the microphone is open, pass
`vad.feed(chunk, playback=True)` for those chunks so its own voice is handled
as possible echo. Every threshold and duration can be configured;
[SPEC.md](https://github.com/KoljaB/RealtimeVAD/blob/main/SPEC.md) lists what is guaranteed.

## How to use `pause`

`pause` is early and often wrong, on purpose. It fires on every breath, every
short gap between words and every stop consonant, so most `pause` events are
followed by more speech.

- **Use it to start work you can cancel:** finalize the transcript, call the
  LLM, prepare the voice output. Cancel when speech continues. A wrong `pause`
  then costs almost nothing, and a right one saves hundreds of milliseconds.
- **Start at `pause`, cancel at `resume`, redo at the next `pause`.** With
  `VAD(resume_events=True)` a `resume` tells you the speech went on: 20 ms
  after sound came back, a median 50 ms after its `pause`. It takes back 91
  to 95 % of the pauses that were followed by more speech (the rest are
  replaced by the next `pause` or closed by `end` without one).
- **Act on every `pause`; the latest one counts.** A `pause` can come inside
  the last word: the level dips before a final consonant or a weak last
  syllable and comes back. When it does, the next quiet gives a new `pause`.
  Redo the speculative work then (the audio is longer now) and drop the
  older result.
- **Without `resume_events`** a `pause` is open until the next `pause` or
  the `end`.
- **Never treat it as the end of the turn.** Only `end` closes a segment.
- **Can't cancel quickly?** Ignore `pause` and wait for `end`. The extra
  false alarms would cost you more than they save.

## Already using Silero?

Change the import:

```python
from silero_vad import load_silero_vad
from RealtimeVAD import VADIterator          # was: from silero_vad import VADIterator

vad_iterator = VADIterator(load_silero_vad(), sampling_rate=16000)
speech_dict = vad_iterator(chunk, return_seconds=True)   # {'start': ...} / {'end': ...} / None
```

The constructor and the results stay the same. `end` is decided where Silero's
own count decides it, and the sample it reports is where the audio went quiet,
not Silero's first low frame. `VADIterator(..., early_end=True)` also decides
it sooner, after quiet audio (see [Results](#results) for what that costs).
Without a model argument, it uses the bundled Silero model and doesn't need
torch.

## Using LiveKit Agents?

[livekit-plugins-realtimevad/](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/README.md)
is a VAD plugin: `realtimevad.VAD.load()` in place of `silero.VAD.load()`. Its
README shows how the events map and what it measured against the Silero plugin
on the same audio. In short: against the Silero plugin's defaults the end of
speech comes about 380 ms sooner, but most of that is the shorter silence
setting (0.25 s instead of 0.55 s), which Silero can be given too. At the
same setting RealtimeVAD's end is 65 to 110 ms earlier, and LiveKit's turn
detector starts 95 to 180 ms earlier. Inside a LiveKit session the turn was
committed about 150 ms sooner than with Silero's defaults and about 70 ms
sooner than with Silero at 0.25 s. These figures were measured with 0.0.5,
whose `end` was the early end; with the default `end` of this version the
plugin's end of speech comes where Silero's own count puts it.

## Results

### Speech-off (`pause`)

One speaker: 3,118 turns from a live voice assistant (browser capture with
noise suppression), each with its trailing silence, default settings. Silero's
decisions are computed from the very same frame probabilities, so both see
identical input. "Speech-off" is RealtimeVAD's last `pause` before the turn
ended, against the first frame of Silero's final run below its threshold.

| RealtimeVAD speech-off earlier than … | median | p25 – p75 | earlier in |
|---|---|---|---|
| Silero, probability below 0.5 (its default threshold) | 90 ms | 72 – 112 ms | 90 % |
| Silero, probability below 0.35 (`VADIterator`'s end threshold) | 102 ms | 82 – 122 ms | 92 % |
| Silero `VADIterator` end, default 100 ms minimum silence | 238 ms | 214 – 294 ms | 96 % |

| Speech-off signals followed by more speech | per second of speech | turns without any |
|---|---|---|
| RealtimeVAD `pause` | 1.9 | 16 % |
| Silero, probability below 0.5 | 0.15 | 84 % |

Many speakers, raw microphones: 7,080 clips from 214 speakers with word
timings (LibriSpeech read speech, AMI meeting headsets; no noise suppression),
see [benchmarks/RESULTS-word-timing.md](https://github.com/KoljaB/RealtimeVAD/blob/main/benchmarks/RESULTS-word-timing.md).
"Too early" means the last `pause` came more than 100 ms before the labelled
end of the last word while at least 30 ms of audible sound of that word were
still to come.

| Audio | Last `pause` too early: 0.0.5 | now | Median lead over Silero: 0.0.5 | now | Earlier than Silero in: 0.0.5 | now | `pause` followed by more speech, per s: 0.0.5 | now |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean | 7.6 % | 2.2 % | 88 ms | 84 ms | 84 % | 83 % | 1.64 | 1.97 |
| LibriSpeech other | 6.2 % | 1.3 % | 86 ms | 80 ms | 82 % | 83 % | 1.42 | 1.70 |
| AMI headset, speaker alone | 10.3 % | 5.7 % | 102 ms | 88 ms | 86 % | 79 % | 1.75 | 2.16 |
| AMI headset, others talking | 11.1 % | 8.4 % | 100 ms | 84 ms | 77 % | 67 % | 1.81 | 2.28 |
| AMI distant microphone | 2.6 % | 2.5 % | 0 ms | 0 ms | 30 % | 30 % | 0.60 | 0.88 |

What changed after 0.0.5, and what it cost:

- **A `pause` inside the last word is now followed by a better one.** The
  level often dips under the threshold before a final consonant or a weak
  last syllable and comes back for 20 to 60 ms. 0.0.5 reported nothing after
  that, so the only `pause` was early. Now 20 ms of sound that is clearly
  above the room lets the next quiet report a new `pause`. The first `pause`
  comes exactly as early as before.
- **The lead is measured on the last `pause`, so it fell:** from 100 ms
  (earlier than Silero in 98 % of turns) to 90 ms (90 %) on the one-speaker
  turns. There the last `pause` moved later in 28 % of turns, by a median
  110 ms; in a sample of those, the sound that caused it was 20 to 38 dB above
  the room. Pauses followed by more speech rose from 1.4 to 1.9 per second.
- **What is left** are word endings that stay under the quiet threshold the
  whole time (about 40 % of the remaining early pauses on LibriSpeech, 70 % on
  the AMI headsets). On the headsets with other people talking, 8 % of the
  last pauses are still too early. Silero's own speech-off is that early in
  0.2 to 0.3 % of the read sentences: it is late, and rarely early.
- The 206 ms against `VADIterator` in earlier versions of this page was
  measured one 32 ms frame short; the same code gives 252 ms for 0.0.5 with
  the fixes on this page, 238 ms with the new `pause`.
- `VAD(repause_s=None)` gives one `pause` per stretch of speech, as in 0.0.5.

The optional `resume` event on the same clips (close microphones): 1.6 to 2.1
per second of speech; it comes a median 50 ms after its `pause` (p90 120 to
150 ms) and 20 ms after the first 10 ms block that was not quiet again (p90
50 ms). In 3 to 9 % of clips a `resume` follows the last `pause` and no new
`pause` comes before the `end`: the turn then ends with the speculative work
cancelled, and `end` is the signal to act on.

### `end`

By default `end` comes where Silero alone ends the segment (same thresholds,
same 0.24 s): on all of the audio above, and in every row of the quiet-input
table below, RealtimeVAD and Silero alone end the same number of turns while
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
| AMI headset, others talking | 52 % | 16 ms | 0 – 88 ms | 1 clip | 33.2 % | 30.9 % |

The one later `end` (32 ms): with the early end, Silero's recovery from a
long pause came on other frames than without it in 7 of 7,080 clips, so its
probabilities differed there. With the default `end` they are identical in
all clips. The clips are whole sentences cut at 0.24 s, so a third of them contain a
pause that long for any detector; the early end adds 2 to 3 points to that.
The one-speaker turns have no word timings; there the early end left speech
of the recording inside its 0.24 s wait in 49 of 3,401 turns.

Reproduce with your own recordings (16 kHz mono WAVs, one turn each, with
trailing silence):

```bash
python benchmarks/speech_off_lead.py <wav dir> out.json
```

## Quiet input and phone audio

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
[Results](#end)).

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

## Limits

- The numbers come from one speaker and one capture chain: browser capture
  with noise suppression and echo cancellation, mostly quiet rooms.
- The lead needs audio that actually goes quiet after a word, and noise
  suppression helps that. In the 137 noisier turns (room level above −70 dBFS)
  the median lead was 102–124 ms. Without noise suppression in a loud room,
  expect less; with steady noise within about 20 dB of the speaker the
  speech-off still leads (the early end mostly does not, see above), and
  within about 11 dB there is no lead.
- Earlier is not the same as correct. Against word timings the last `pause`
  is still more than 100 ms early, with sound of the word to come, in 1 to 2 %
  of read sentences and 6 to 8 % of meeting clips ([Results](#results)).
- The lead is for speech-off. `end` is Silero's by default and has no lead;
  with `early_end=True` it gains a median 18 to 70 ms and ends more segments
  inside an utterance.
- Whisper is mostly not detected (Silero scores it below 0.5). Laughter,
  coughs and sneezes often are.
- A DC offset in the input no longer hides the silence from the quiet clock
  (up to 0.0.5 it did: no early `pause`, no early end). Levels are now measured behind a 20 Hz
  high-pass; the audio you get back and what Silero sees keep the offset. On
  the 3,401 turns with a constant offset of −40 dBFS added, the speech-off
  lead was 0 ms (earlier than Silero in 6 % of turns) and is now 102 ms
  (99 %), against 100 ms (98 %) without the offset. What is left is Silero
  itself: it is a little less sure with an offset under the speech, most of
  all under quiet speech (at −52 dBFS it finds no speech in 254 turns instead
  of 139). Remove a large offset before the VAD if you have one.
- 16 kHz mono only.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The edge-case tests in `tests/audio/` run on about 100 MB of freely licensed
audio that is not in git (speech in 8 languages, noise, music, coughs and
laughter, whisper, real backchannels and hesitations). Without it they are
skipped. To get it (about 1.3 GB of downloads, checked by sha256):

```bash
python tests/audio/fetch_audio.py
python tests/audio/fetch_backchannels.py
pytest tests/audio
```

They take about 10 s with 8 worker processes and print what each test guards, with the
measured value and its limit. Sources and licenses:
[LICENSES-audio.md](https://github.com/KoljaB/RealtimeVAD/blob/main/tests/audio/LICENSES-audio.md),
[LICENSES-backchannels.md](https://github.com/KoljaB/RealtimeVAD/blob/main/tests/audio/LICENSES-backchannels.md).

## License and credits

MIT. The bundled speech model is [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
