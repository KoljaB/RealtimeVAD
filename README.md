# RealtimeVAD

*Know that the speaker stopped, about 100 ms before Silero does.*

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
- **Lead:** speech-off a median 100 ms before Silero's probability drops, and 206 ms before Silero's `VADIterator` reports the end, on 3,279 real turns. The `end` itself comes before Silero's in 82 % of turns, by a median 70 ms.
- **Price:** it also fires in short gaps between words, about ten times as often as Silero.
- **Quiet input:** a quiet speaker's soft word endings are not silence; on a noisy quiet line the lead is gone instead ([details](#quiet-input-and-phone-audio)).
- **Runs on:** one CPU thread, far faster than realtime. Needs only numpy and onnxruntime; the Silero model is bundled, no torch.
- **Status:** alpha (0.0.5). The API may still change.

## Live demo

https://github.com/user-attachments/assets/d01900dd-9601-4f3d-8c88-db55e8bc83a1

The video uses a synthetic voice and shows a median lead of 92 ms. The 100 ms above is the median on real voice-assistant turns.

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
| `pause` | The speaker may have stopped (speech-off) | Within about 10 ms of the audio going quiet |
| `end` | The segment is over | 0.24 s after speech-off, unless speech resumes. Never later than Silero alone |
| `onset` | Speech may have begun | At the first confident frame, before `start` confirms it. Off by default: `VAD(onset_events=True)` |

`vad.audio()` returns the running segment at any time, starting about 0.3 s
before the speech. `end` carries the finished segment with its audio, start
time, end time and the time the speech stopped.

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

The constructor and the results stay the same, but `end` arrives as soon as the
audio goes quiet. Without a model argument, it uses the bundled Silero model and
doesn't need torch.

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
sooner than with Silero at 0.25 s.

## Results

Measured on 3,279 turns from a live voice assistant, each with its trailing
silence, with default settings. Silero's decisions are computed from the very
same frame probabilities, so both see identical input. "Speech-off" is
RealtimeVAD's last `pause` before the turn ended, against the first frame of
Silero's final run below its threshold.

| RealtimeVAD speech-off earlier than … | median | p25 – p75 | p90 | earlier in |
|---|---|---|---|---|
| Silero, probability below 0.5 (its default threshold) | 100 ms | 80 – 150 ms | 214 ms | 98 % |
| Silero, probability below 0.35 (`VADIterator`'s end threshold) | 110 ms | 90 – 176 ms | 230 ms | 98 % |
| Silero `VADIterator` end, default 100 ms minimum silence | 206 ms | 186 – 272 ms | 336 ms | 99 % |

| Speech-off signals followed by more speech | per second of speech | turns without any |
|---|---|---|
| RealtimeVAD `pause` | 1.5 | 17 % |
| Silero, probability below 0.5 | 0.14 | 86 % |

Until 0.0.4 the first row read 110 ms. The 10 ms went into not cutting quiet
speakers, see the next section.

The `end` event, against the end Silero's own silence count gives with the
same thresholds and the same 0.24 s (2,323 turns where both closed the same
segments inside the recording):

| RealtimeVAD `end` earlier than Silero alone | earlier in | median | p25 – p75 | later in |
|---|---|---|---|---|
| 0.0.5 | 82 % | 70 ms | 40 – 92 ms | 0 |
| the same without restarting the count after a click | 51 % | 18 ms | 0 – 76 ms | 0 |

Up to 0.0.4 a single loud 10 ms block inside the quiet after a word (a click,
a lip smack, a breath) stopped the quiet count for good and left the end to
Silero. That happened in a third of 680 sampled turns. The count now starts
again after it.

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
- **A noisy quiet line has no lead.** When the noise is less than about 20 dB
  below the speaker (the room-noise row, and the quiet phone line with its
  line noise), nothing can be called quiet without calling soft speech quiet.
  RealtimeVAD then ends where Silero does: correct, and no faster.
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

What to set:

| If … | then |
|---|---|
| your input is quiet and clean | nothing; the lead stays |
| you would rather keep the old fixed floor | `VAD(floor_margin_db=None)` |
| soft starts open segments you don't want (background voices) | `VAD(soft_start_s=None)` |
| you want Silero's own count only, no early end | `VAD(quiet_rms=0)` |
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
  expect less; with noise within about 20 dB of the speaker, none.
- Earlier is not the same as correct: at the recorded level, whether a
  speech-off sometimes lands inside a soft word ending was not checked against
  word timings. The quiet-input table above checks it against the same audio
  at normal level.
- The 100 ms are for speech-off. `end` waits 0.24 s after speech-off in both
  cases; its own gain over Silero is smaller (a median 70 ms, none in 18 % of
  turns) and depends on the noise.
- Whisper is mostly not detected (Silero scores it below 0.5). Laughter,
  coughs and sneezes often are.
- A DC offset in the input hides the silence from the quiet clock: no early
  end then, nothing lost.
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
