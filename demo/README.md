# RealtimeVAD × RealtimeSTT live demo

A voice assistant can only answer once it knows two things: that you stopped
talking, and what you said. This demo shows both arriving in real time.

- **RealtimeVAD** marks every pause within about 10 ms of silence and ends the
  turn 0.24 s after the speech stopped.
- At every pause, **RealtimeSTT** answers with the transcript so far from its
  streaming model (live-flush Preview), typically in about 50 ms.
- So the transcript is usually ready *before* the turn is even over.

The page draws Silero's speech probability, the audio with the quiet
threshold that adapts to the room, every pause with the moment its transcript
arrived, the turn end, and where Silero alone (same probabilities, same
thresholds) would have ended the turn.

## Run

The demo needs a RealtimeSTT ≥ 1.2.0 production server with live flush. The
fast path is CPU only (Linux x86-64):

```bash
pip install "RealtimeSTT[server,sherpa-onnx,nemotron-ort]"
stt-install-sherpa-models --root ./models/sherpa-onnx --model nemotron
stt-install-sherpa-models --root ./models/sherpa-onnx --model orukeet
stt-server-production --port 8010 \
  --engine sherpa_onnx_parakeet --model oruk/orukeet \
  --realtime-engine nemotron_ort \
  --realtime-model ./models/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --download-root ./models/sherpa-onnx --device cpu \
  --preview-only-transcription --preview-input-mode tail --preview-live-flush
```

Then start the demo from the repository root and open http://127.0.0.1:8200:

```bash
pip install -e . -r demo/requirements.txt
python demo/server.py --asr-url ws://127.0.0.1:8010/api/v1/ws/transcribe
```

For a remote server with TLS and a bearer token:

```bash
REALTIMESTT_TOKEN=... python demo/server.py \
  --asr-url wss://asr-host:8767/api/v1/ws/transcribe --asr-ca ca.crt
```

`--sample file.wav` (16 kHz mono) enables **Play sample**, which streams a
recording instead of the microphone, handy for repeatable recordings.

## Files

| File | What it does |
|---|---|
| `server.py` | Feeds browser audio to RealtimeVAD; opens an ASR turn on start, asks for a Preview at each pause, finalizes at end; computes the Silero-only end |
| `asr.py` | Minimal client for the RealtimeSTT production WebSocket protocol |
| `static/` | The page: capture worklet, canvas timeline, tiles and turn cards |

All times on the page are stream time (seconds of audio sent), the same clock
the server stamps on every event. Preview latencies are measured on the
server, from the pause to the transcript's arrival.
