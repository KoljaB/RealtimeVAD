# RealtimeVAD live demo

https://github.com/user-attachments/assets/d01900dd-9601-4f3d-8c88-db55e8bc83a1

Talk into your microphone, stop, and see how much sooner RealtimeVAD knew you
had stopped than Silero did. Both hear the same audio, and the page puts their
decisions on one timeline.

## Run

From the repository root:

```bash
pip install -e . -r demo/requirements.txt
python demo/server.py
```

Open http://127.0.0.1:8200, press **Start microphone**, say a sentence and stop.

No microphone at hand? Start the server with `--sample file.wav` (16 kHz mono)
and press **Play sample**.

## What you see

| On the page | Meaning |
|---|---|
| **The big number** | How much sooner RealtimeVAD knew you stopped in your last turn. Below it: the median over all turns so far, and in how many of them it was sooner |
| **Turn end** | How much sooner RealtimeVAD closed the turn. It can be sooner or the same, never later |
| **Early pauses** | Pauses that fired while you kept talking. This is the price of reacting so early |
| **Yellow dot** | Where RealtimeVAD knows you stopped |
| **Red dot** | Where Silero alone knows |
| **Shaded band** | The time between the two dots: what RealtimeVAD won |

The timeline has four rows: the audio, Silero's speech probability, RealtimeVAD
and Silero alone. Every finished turn is also listed below it.

"Silero alone" is Silero with the same probabilities and the same thresholds,
only without RealtimeVAD's check for quiet every 10 ms. It knows you stopped at
its first silent frame.

## Settings

Open **Settings** below the timeline:

- **Pause after:** how long the audio must stay quiet before a pause counts.
  Longer gives fewer early pauses and a later speech-off.
- **Speech-on:** how sure Silero must be, and for how long, before a start is
  confirmed. Stricter rejects clicks and knocks and confirms the start later.
- **Show every event:** adds what the default view leaves out. Onsets (a ring;
  grey if it never became speech), every pause, the wait after speech-off, both
  turn ends, the quiet threshold and Silero's 0.45 line.

Add `?window=4` to the address to zoom the timeline to 4 seconds. The default
is 8.

## What to expect

- **The video uses a synthetic voice** and shows a median lead of 92 ms. On
  real voice-assistant turns the measured median is 110 ms.
- **The lead depends on the voice** and on how each sentence ends.
- **Noise matters.** The page captures the microphone with the browser's noise
  suppression on, as a voice assistant would. That makes the gaps after words
  actually quiet, which is what the 10 ms check reacts to. In a noisy recording
  the lead is much smaller.

## Files

| File | What it does |
|---|---|
| `server.py` | Feeds the browser's audio to RealtimeVAD and works out what Silero alone would decide from the same frames |
| `static/` | The page: audio capture, the timeline, the tiles and the turn list |
| `demo.mp4` | The video above |
