# RealtimeVAD

*Fast speech-end detection for voice assistants that shouldn't wait.*

> **Status: early development.** The API may still change.

Most voice activity detectors notice speech fine, but they take hundreds of
milliseconds to decide that the speaker has stopped. In a voice assistant, that
delay comes before anything else can happen: ASR finalization, the LLM call and
TTS all wait for it. RealtimeVAD is built to make that decision as early as
possible.

Part of the Realtime family:
[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) ·
[RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)

## Installation

```bash
pip install realtimevad
```

(Not yet published.)

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

What it guarantees: [SPEC.md](SPEC.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
