# RealtimeVAD live demo

Speak into the microphone and stop. The page shows when RealtimeVAD notices that
you stopped, and when Silero alone would have, on one scrolling timeline.

- **Audio**: the waveform and the quiet threshold, which adapts to the room.
- **Silero**: its speech probability per 32 ms frame, with the 0.45 line below
  which a frame counts as silent.
- **RealtimeVAD**: a yellow dot at every pause (the audio went quiet for one
  10 ms block), then speech-off ●, the wait to be sure, and the turn end ▮.
- **Silero alone**: the same probabilities and thresholds without the 10 ms
  clock: speech-off at its first silent frame, then its own wait and turn end.

Between the two rows the page marks how much sooner RealtimeVAD knew, for
speech-off and for the turn end. RealtimeVAD can only be sooner or the same,
never later. The slider sets how long the audio must stay quiet before a pause
counts (fewer pauses, later speech-off).

## Run

From the repository root, then open http://127.0.0.1:8200:

```bash
pip install -e . -r demo/requirements.txt
python demo/server.py
```

`--sample file.wav` (16 kHz mono) enables **Play sample**, which streams a
recording instead of the microphone.

The microphone is captured with the browser's noise suppression on, as a voice
assistant would: it makes the gaps after words actually quiet, which is what
the 10 ms clock reacts to. A recording with a lot of room noise shows a much
smaller lead.

## Files

| File | What it does |
|---|---|
| `server.py` | Feeds browser audio to RealtimeVAD and works out the Silero-alone decisions from the same frames |
| `static/` | The page: capture worklet, canvas timeline, tiles and turn cards |

All times on the page are stream time (seconds of audio sent), the same clock
the server stamps on every event.
