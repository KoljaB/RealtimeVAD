# RealtimeVAD plugin for LiveKit Agents

[RealtimeVAD](https://github.com/KoljaB/RealtimeVAD) in LiveKit's VAD slot. A community plugin, not made by LiveKit.

```python
from livekit.agents import AgentSession
from livekit.plugins import realtimevad      # was: silero

session = AgentSession(
    vad=realtimevad.VAD.load(),              # was: silero.VAD.load()
    ...
)
```

The plugin itself is not on PyPI yet ([#5](https://github.com/KoljaB/RealtimeVAD/issues/5)). From a checkout of this repository:

```bash
pip install "realtimevad>=0.0.5"
pip install ./livekit-plugins-realtimevad
```

Tested with livekit-agents 1.8.11, Python 3.10+.

## What it changes

- **End of speech sooner.** `END_OF_SPEECH` comes 0.25 s after the audio went
  quiet. The Silero plugin's default is 0.55 s after Silero's probability fell.
- **The stop is reported where it happened.** LiveKit computes "user stopped
  speaking" as the event time minus `silence_duration`. RealtimeVAD knows the
  instant the audio went quiet; Silero's probability lags it by about 100 ms.
- **The turn detector starts sooner.** LiveKit starts its turn detector when an
  `INFERENCE_DONE` reports 0.2 s of `raw_accumulated_silence`. The plugin counts
  that from the quiet clock, not from Silero's first silent frame.
- **Very short sounds no longer interrupt.** LiveKit interrupts the agent when
  `speech_duration` reaches 0.5 s, and that counter runs until `END_OF_SPEECH`.
  With 0.25 s to the end, a sound under about 0.25 s never gets there. A
  typical 0.35 s "mm-hmm" still does, as with Silero.
- **More, shorter segments.** A 0.25 s pause inside a sentence ends a segment.
  LiveKit commits a turn only after its own endpointing delay, so this splits
  VAD segments, not turns. With a non-streaming STT behind LiveKit's
  `StreamAdapter` every segment is its own request: use
  `min_silence_duration=0.55` there.
- **Stricter start.** A segment starts on Silero probability 0.9 (the Silero
  plugin: 0.5), or after 0.13 s at 0.5 over audio clearly louder than the room.
  Fewer false starts in noise, some more missed soft sounds.

## How the events map

LiveKit has three event types and no "pause".

| RealtimeVAD | LiveKit | |
|---|---|---|
| `start` | `START_OF_SPEECH` | Same moment. `speech_duration` reaches back to where the speech began (a soft opening, or where Silero first passed 0.5); `frames` hold 0.5 s of padding plus the opening |
| `end` | `END_OF_SPEECH` | Same moment. `silence_duration` reaches back to the instant the audio went quiet; `frames` hold the whole utterance, padding and trailing silence |
| each 32 ms frame | `INFERENCE_DONE` | Silero's probability, the frame's audio at the input sample rate, `speech_duration` since the start, and the two raw counters below |
| `pause` | (none) | Runs into `raw_accumulated_silence` and stops `raw_accumulated_speech`. Also `vad.on("pause", callback)` |
| `onset` | (none) | `vad.on("onset", callback)` |

`pause` cannot start LiveKit's preemptive generation: that is triggered by
transcripts (a final or preflight transcript from the STT), not by the VAD.

`flush()` is a hard boundary, as LiveKit asks: the running speech is dropped
without an `END_OF_SPEECH`, time starts again, and what the VAD learned about
the room and the speaker is forgotten.

## Options

`VAD.load()` takes the Silero plugin's names where the meaning is the same:

| Option | Default | Silero plugin | |
|---|---|---|---|
| `min_silence_duration` | 0.25 | 0.55 | LiveKit's audio turn detector refuses less than 0.25 |
| `min_speech_duration` | 0.064 | 0.05 | Both come to two 32 ms frames |
| `prefix_padding_duration` | 0.5 | 0.5 | |
| `activation_threshold` | 0.9 | 0.5 | |
| `deactivation_threshold` | 0.45 | 0.35 | |
| `max_buffered_speech` | 60 | 60 | Longest speech kept in the `END_OF_SPEECH` frames |

Any other keyword goes to `RealtimeVAD.VAD`, for example `soft_start_s=None`
(no soft start), `quiet_rms=0` (no quiet clock: Silero's count alone),
`floor_margin_db`, or `normalize_to_dbfs=-30` for quiet phone audio (see the
main README for what it gains and costs).

`vad.update_options(...)` changes the six options of the table while streams
run, like the Silero plugin's. A running stream keeps what it has learned about
the room and the speaker; a new `min_silence_duration` counts from the next
pause. An option that `load()` got under RealtimeVAD's own name (`end_s`,
`start_s`, ...) stays as given there.

```python
vad.update_options(min_silence_duration=0.55)   # e.g. while the user dictates a number
```

Not there: 8 kHz inference (input at any rate is resampled to 16 kHz).

### When the agent speaks

LiveKit gives a VAD audio frames and nothing else; it is not told when the agent
speaks. To use RealtimeVAD's `playback` flag, tell the plugin yourself:

```python
vad = realtimevad.VAD.load()
session = AgentSession(vad=vad, ...)
session.on("agent_state_changed",
           lambda ev: vad.set_agent_speaking(ev.new_state == "speaking"))
```

While it is set, and for 0.5 s after, RealtimeVAD does not learn the room's
noise level from the audio and does not take Silero's recovery state from it,
so echo of the agent's voice that gets past the echo canceller cannot shift
either. Speech is detected as before: the user can still interrupt. Without the
call the plugin relies on echo cancellation alone, like the Silero plugin.

## Measured against the Silero plugin

Both plugins got the same 20 ms frames through `vad.stream()`. From their
events: when they fire, what they report, and what LiveKit would do with them
(turn detector at 0.2 s of raw silence; commit at `END_OF_SPEECH`, but not
before the reported stop plus `min_delay`, with an instant STT). Medians in ms.

Sentences in 8 languages, 24 speakers, speech at −30 dBFS over faint noise:

| | Silero default (0.55 s) | Silero 0.25 s | RealtimeVAD (0.25 s) | RealtimeVAD 0.55 s |
|---|---|---|---|---|
| **Opening:** `START_OF_SPEECH` after the speech began (median / p90) | 76 / 76 | 76 / 98 | 76 / 108 | 76 / 108 |
| **Ending:** `END_OF_SPEECH` after the speech stopped | 684 | 364 | 295 | 600 |
| Reported stop, later than the true stop | 108 | 108 | 27 | 44 |
| Turn detector starts after the stop | 364 | 364 | 236 | 236 |
| Turn committed after the stop, `min_delay` 0.5 / 0.3 | 684 / 684 | 608 / 408 | 527 / 327 | 600 / 600 |
| **Segmentation:** segments per sentence | 1.0 | 1.62 | 1.79 | 1.04 |
| **Interruption:** `speech_duration` reaches 0.5 s after the speech began | 524 | 524 | 492 | 492 |

Other material, same four columns:

| | Silero default | Silero 0.25 s | RealtimeVAD | RealtimeVAD 0.55 s |
|---|---|---|---|---|
| Real turn starts from meetings (60): `END_OF_SPEECH` after the stop | 838 | 518 | 449 | 745 |
| … `START_OF_SPEECH` after the turn began (median / p90) | 120 / 248 | 120 / 248 | 152 / 568 | 152 / 568 |
| … `speech_duration` reaches 0.5 s after the turn began | 568 | 568 | 568 | 568 |
| Turns with a 0.5–1.5 s pause (30): `END_OF_SPEECH` after the stop | 712 | 392 | 280 | 624 |
| … turn detector starts after the stop | 392 | 392 | 215 | 209 |
| … segments per turn | 2.4 | 3.23 | 3.23 | 2.47 |
| Phone line at −45 dBFS (24): not heard at all | 1 | 1 | 2 | 2 |
| … `END_OF_SPEECH` after the stop | 700 | 380 | 316 | 650 |
| Real backchannels, 0.15–0.8 s (200): not heard at all | 19 | 19 | 24 | 24 |
| … would interrupt the agent | 181 | 168 | 144 | 176 |
| Coughs, laughter, sighs, sneezes (50): would interrupt the agent | 36 | 30 | 29 | 33 |

Read it like this:

- Against the Silero plugin as most people run it, the end of speech comes
  about 380 ms sooner. Most of that is 0.25 s instead of 0.55 s of silence,
  and Silero can be set to 0.25 s too.
- At the same 0.25 s, RealtimeVAD's end is 65 to 110 ms earlier, its reported
  stop 65 to 110 ms closer to the real one, and the turn detector starts 95 to
  180 ms earlier. The commit moves by 70 to 105 ms at either `min_delay`.
- Either VAD at 0.25 s splits sentences at short pauses; RealtimeVAD a little
  more (1.79 against 1.62 segments per sentence).
- `START_OF_SPEECH` is as fast on clear speech and slower on soft openings (p90
  of 568 against 248 ms on the meeting turns). The audio of the opening is
  still in the segment and `speech_duration` counts it, so interruptions come
  as soon as with Silero or sooner.
- It hears slightly fewer short and quiet sounds than Silero at 0.5 (24
  against 19 of 200 backchannels missed). Fewer of them reach the 0.5 s that
  interrupts the agent (144 against 181), mostly the very short ones.

### Inside an AgentSession

The same three VADs in a real `AgentSession` (LiveKit's own end-of-turn and
interruption logic, `turn_detection="vad"`, LiveKit's test fakes for STT, LLM
and TTS, audio fed at real-time pace). Median over 6 sentences in 6 languages,
ms after the true end or start of the speech:

| | Silero default (0.55 s) | Silero 0.25 s | RealtimeVAD (0.25 s) |
|---|---|---|---|
| User state leaves "speaking" | 708 | 387 | 321 |
| Turn committed, `min_delay` 0.5 (LiveKit's default) | 709 | 632 | 561 |
| First agent audio, `min_delay` 0.5 | 812 | 734 | 663 |
| Turn committed, `min_delay` 0.3 | 708 | 433 | 361 |
| First agent audio, `min_delay` 0.3 | 811 | 535 | 463 |
| Agent paused after the user starts talking over it | 506 | 506 | 503 |
| … and the user's turn answered | 6 of 6 | 6 of 6 | 6 of 6 |
| A 0.3–0.4 s backchannel pauses the speaking agent | 4 of 6 | 4 of 6 | 5 of 6 |
| … the agent resumes after the sound ended | 2763 | 2444 | 2324 |
| Accepted by LiveKit's default turn detector (needs ≥ 0.25 s) | yes | yes | yes |

So in a session the turn is committed about 150 ms sooner than with the
Silero plugin's defaults and about 70 ms sooner than with Silero at 0.25 s;
with `min_delay` lowered to 0.3 s it is 350 ms and 70 ms. Interruptions come
at the same time. No events out of order, no errors logged.

```bash
python benchmarks/agent_session_check.py out.json   # also needs a checkout of livekit/agents for its test fakes
```

One speaker per clip, no echo, no real room, a fake STT that answers 50 ms
after the speech: these are timings of the VAD's part, not call quality.

```bash
python benchmarks/compare_silero.py out.json     # needs the repository's test audio, see the main README
pytest                                            # the event mapping, with scripted probabilities
```

## Pipecat

Tracked in [#4](https://github.com/KoljaB/RealtimeVAD/issues/4). Pipecat's slot is different: a `VADAnalyzer` only supplies a probability per
window and Pipecat runs its own start/stop timers on it. RealtimeVAD's early end
does not fit through that without overriding `analyze_audio`. Not done here.

## License

MIT. Uses LiveKit Agents (Apache-2.0) and the Silero VAD model (MIT).
