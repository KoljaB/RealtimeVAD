# RealtimeVAD

*Fast speech-end detection for voice assistants that shouldn't wait.*

> **Status: early development.** The API may still change.

Most voice activity detectors notice speech fine, but they take hundreds of
milliseconds to decide that the speaker has stopped. In a voice assistant, that
delay comes before anything else can happen: ASR finalization, the LLM call and
TTS all wait for it. RealtimeVAD is built to make that decision as early as
possible.

**Faster than Silero at detecting speech-off.** On 3,238 real voice-assistant
turns, RealtimeVAD signalled speech-off a median **110 ms** before Silero's own
probability dropped below its threshold, and **214 ms** before Silero's
`VADIterator` reported the end with its default settings; it was earlier in 98 %
of turns. The price: it also fires in short gaps between words, about ten times
as often as Silero. If your pipeline can start speculative work at a speech-off
and cancel it quickly, using Silero leaves that time on the table; if it can't,
the extra false alarms cost more than they save. [How this was measured](#measured-speech-off-on-real-turns).

Part of the Realtime family:
[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) ·
[RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)

## Installation

```bash
pip install realtimevad
```

## Usage

Feed 16 kHz mono audio in chunks of any size, and react to the events:

```python
from RealtimeVAD import VAD

vad = VAD()
for chunk in microphone_chunks():           # float32, int16 or int16 bytes
    for event in vad.feed(chunk):
        if event.kind == "start":
            stream_to_asr(vad.audio())       # includes the audio just before speech began
        elif event.kind == "pause":
            speculate(vad.audio())           # the speaker may be done; this isn't the end yet
        elif event.kind == "end":
            finalize(event.segment.audio)
```

### About `pause`: early, and often wrong

`pause` fires extremely early, within about 10 ms of the first sign of
silence. That makes it fire constantly: on every breath, every short gap
between words and every stop consonant. Most `pause` events are false
positives, and that is intended.

`pause` is made for speculative generation: start the ASR finalization, the
LLM call or TTS right away, and cancel it the moment speech continues. If
that work is cheap to abort, a false `pause` costs almost nothing, and a
correct one saves hundreds of milliseconds. Never treat a `pause` as the end
of the turn; only `end` closes a segment. If your pipeline can't cancel
quickly, ignore `pause` and wait for `end`.

### Measured: speech-off on real turns

3,238 turns from a live voice assistant (one speaker, one microphone, browser
capture with noise suppression and echo cancellation, mostly quiet rooms),
each with its trailing silence, run through RealtimeVAD with default settings.
Silero's decisions are computed from the very same frame probabilities, so both
see identical input. "Speech-off" is RealtimeVAD's last `pause` before the turn
ended, against the first frame of Silero's final run below its threshold.

| RealtimeVAD speech-off earlier than … | median | p25 – p75 | p90 | earlier in |
|---|---|---|---|---|
| Silero, probability below 0.5 (its default threshold) | 110 ms | 84 – 190 ms | 242 ms | 98 % |
| Silero, probability below 0.35 (`VADIterator`'s end threshold) | 120 ms | 98 – 200 ms | 252 ms | 98 % |
| Silero `VADIterator` end, default 100 ms minimum silence | 214 ms | 192 – 298 ms | 356 ms | 99 % |

| Speech-off signals followed by more speech | per second of speech | turns without any |
|---|---|---|
| RealtimeVAD `pause` | 1.6 | 16 % |
| Silero, probability below 0.5 | 0.15 | 85 % |

What these numbers don't cover:

- One speaker and one capture chain. The early speech-off comes from audio that
  actually goes quiet after a word; noise suppression helps that. In the few
  noisier turns (above −70 dBFS room level, 147 turns) the median lead was
  92–132 ms; without noise suppression in a loud room expect less.
- Earlier is not the same as correct: whether a speech-off sometimes lands
  inside a soft word ending was not checked against word timings.
- This is speech-off, not the turn end. `end` waits 0.24 s after it in both
  cases; its own gain over Silero is smaller and depends on the noise.

Reproduce with your own recordings: `python benchmarks/speech_off_lead.py <wav dir> out.json`
(16 kHz mono WAVs, one turn each, with trailing silence).

Already using Silero's `VADIterator`? Change the import:

```python
from silero_vad import load_silero_vad
from RealtimeVAD import VADIterator          # was: from silero_vad import VADIterator

vad_iterator = VADIterator(load_silero_vad(), sampling_rate=16000)
speech_dict = vad_iterator(chunk, return_seconds=True)   # {'start': ...} / {'end': ...} / None
```

The constructor and the results stay the same, but `end` arrives as soon as the
audio goes quiet. Without a model argument, it uses the bundled Silero model and
doesn't need torch.

What it guarantees: [SPEC.md](https://github.com/KoljaB/RealtimeVAD/blob/main/SPEC.md).

## Live demo

[demo/](https://github.com/KoljaB/RealtimeVAD/blob/main/demo/README.md) shows, live from your microphone, when RealtimeVAD
notices that you stopped talking and when Silero alone would have, on one
scrolling timeline.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
