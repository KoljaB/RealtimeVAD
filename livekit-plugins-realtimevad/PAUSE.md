# What LiveKit can do with an early "speech off"

RealtimeVAD reports that the audio went quiet (`pause`) about 100 ms before Silero's
probability drops. This note goes through everything in LiveKit Agents that reads VAD events and
asks: can that signal make an agent answer sooner, through supported extension points?

Source: livekit-agents 1.8.11 (tag `livekit-agents@1.8.11`, the newest release on 2026-11-13).
Paths are relative to `livekit-agents/livekit/agents/`.

**Verdict: yes, through preemptive generation, by way of the STT slot.** LiveKit never starts
work from a VAD's pause itself. It starts a reply early from a transcript, and an STT may deliver
that transcript whenever it likes. An STT adapter that asks for the transcript at the pause is a
supported construction (it is LiveKit's own `StreamAdapter` with the request moved forward) and
is in this plugin as `realtimevad.SpeculativeSTT`. Measured (README, "Reply from the pause"): first
agent audio about 420 ms sooner with the adapter on the Silero plugin's own raw silence, and 27 to
140 ms more with RealtimeVAD's pause, which also fires more often inside a sentence.

## What a VAD plugin can say at all

`vad.py:19-70`: three event types (`START_OF_SPEECH`, `INFERENCE_DONE`, `END_OF_SPEECH`) with
`speech_duration`, `silence_duration`, `raw_accumulated_silence`, `raw_accumulated_speech`,
`speaking`, `probability`, `frames`. There is no pause event. The plugin puts the pause into
`raw_accumulated_silence` of its `INFERENCE_DONE` events: it counts from the instant the audio
went quiet.

## Every consumer of VAD events

| # | Consumer | Where | Reads | Does an early pause help? |
|---|---|---|---|---|
| 1 | Turn detector start | `voice/audio_recognition.py:1507-1512` | `raw_accumulated_silence >= 0.2 s` while speaking (`inference/eot/base.py:32`) | Starts the prediction request sooner. Does not move the commit, see below |
| 2 | Preemptive generation | `voice/audio_recognition.py:1317-1334` (final transcript), `:1343-1381` (preflight transcript); `voice/agent_activity.py:2635-2690` | transcripts only, no VAD field | **Yes, if the STT delivers the transcript at the pause** |
| 3 | Endpointing delay | `voice/audio_recognition.py:1522-1523`, `:1830-1839` | `END_OF_SPEECH` time minus `silence_duration` | Only through the reported stop time, and only when that bounds the commit |
| 4 | Interruption of the agent | `voice/agent_activity.py:2520-2530` | `speech_duration >= min_duration` | No. It is about speech starting |
| 5 | "User is silent" gate for playout | `voice/agent_activity.py:2532-2539`, `:3256-3258` | `speaking` and `raw_accumulated_silence <= min_delay / 2` | Opens the gate sooner inside a pause; a reply only waits on it when it was scheduled while the user was still marked speaking |
| 6 | "Last speaking" time, speech start | `voice/audio_recognition.py:1497-1501` | `raw_accumulated_speech > 0` | No (metrics; overwritten at `END_OF_SPEECH`) |
| 7 | User state, false-interruption timer, dynamic endpointing statistics, transcription timeout, answering-machine detection | `voice/audio_recognition.py:1465-1491`, `:1514-1539`, `:665-677`; `voice/agent_activity.py:2446-2512` | `START_OF_SPEECH`, `END_OF_SPEECH` | No. They run on the end, not the pause |
| 8 | `StreamAdapter` (non-streaming STT) | `stt/stream_adapter.py:125-164` | `END_OF_SPEECH` and its frames | No: it sends the request at `END_OF_SPEECH`. This is the part `SpeculativeSTT` replaces |

### 1. The turn detector start

`AgentSession` uses `inference.TurnDetector()` unless told otherwise (`voice/agent_session.py:586`):
a streaming detector that runs on LiveKit's inference gateway (`inference/eot/detector.py`,
`inference/eot/transports.py:156-192`). `audio_recognition.py:1507-1512` calls its `predict()` at
the first `INFERENCE_DONE` with 0.2 s of raw silence inside speech, and cancels the request when
speech goes on (`:1497-1505`, `:1482-1484`). With RealtimeVAD's counter that request leaves about
100 ms sooner.

The commit does not move because of it. The end-of-turn task is created at `END_OF_SPEECH` or at a
final transcript (`:1532-1536`, `:1336-1341`). Only there is the prediction awaited (`:1658-1686`),
and after it the task still sleeps until `reported stop + min_delay` (`:1830-1839`), or `max_delay`
when the model says the turn is not over (`:1698-1704`). So:

    turn committed = max(END_OF_SPEECH, final transcript, prediction arrived, reported stop + delay)

An earlier `predict()` shortens only the third term, and that term decides only when the
prediction takes longer than the time from 0.2 s of raw silence to `END_OF_SPEECH`: at least
50 ms (the detector refuses a VAD with less than 0.25 s `min_silence_duration`, `:932-950`),
350 ms with the Silero plugin's default. Not measured here: the detector needs a LiveKit Cloud
connection, which this project's benchmarks do not use. With `turn_detection="vad"` there is no
prediction and this route gives nothing.

Wrong pause: the request is cancelled at `:1502-1505`. Clean.

### 2. Preemptive generation

On by default (`voice/turn.py:223-228`). Triggers, all of them transcripts:

- a `FINAL_TRANSCRIPT` whose text changes the turn's transcript (`audio_recognition.py:1307`,
  `:1317-1334`), also while the VAD still says "speaking";
- a `PREFLIGHT_TRANSCRIPT` (`:1343-1381`), defined in `stt/stt.py:44-48` as "stable enough to be
  used for preemptive generation". The plugins for AssemblyAI, Cartesia, Deepgram (Flux), Soniox and
  others emit it from their provider's "eager end of turn".

`agent_activity.py:2635-2690` then starts the LLM call (`schedule_speech=False`: nothing is
spoken), and the TTS as well with `preemptive_tts=True` (`voice/turn.py:207-210`). When the turn
is committed, the reply is used if its transcript equals the final one, ignoring case and
punctuation (`agent_activity.py:2950-2970`, `:153-167`).

Nothing in this path reads the VAD, so a VAD plugin cannot trigger it. An STT can:

- **A non-streaming STT** is driven by a VAD anyway. `SpeculativeSTT` sends the segment so far to
  `recognize()` at the pause and emits the text as `PREFLIGHT_TRANSCRIPT`; at `END_OF_SPEECH` it
  emits the same text as `FINAL_TRANSCRIPT` without a second request. `audio_recognition.py:1307`
  then sees no change and the running reply is the one that gets used.
- **A streaming STT with `STTCapabilities.manual_flush`** (`stt/stt.py:144-153`) finalizes when its
  stream is flushed. LiveKit flushes it at the VAD's `END_OF_SPEECH` (`audio_recognition.py:1591-1595`,
  `:1809-1825`). `SpeculativeSTT` flushes it at the pause; the final transcript that comes back
  starts the reply (`:1317-1334`).

Both use public API only: `stt.STT`, `stt.RecognizeStream`, `vad.VAD.stream()`,
`RecognizeStream.flush()`, the `SpeechEventType` values.

Wrong pause, speech continues:

- `SpeculativeSTT` drops a transcript whose pause was followed by speech; it never becomes final.
- The reply LiveKit already began from it keeps running until the next preflight or final
  transcript replaces it (`agent_activity.py:2646-2653`), an interim transcript moves past it
  (`audio_recognition.py:1392-1396`, `agent_activity.py:2692-2705`), or the turn is committed with a
  different transcript (`agent_activity.py:2950-2980`). It is never spoken.
- A streaming STT that was flushed at a wrong pause has finalized a fragment. LiveKit joins the
  fragments of a turn (`audio_recognition.py:1304-1305`), but the recognizer starts the next
  fragment without the context of the first.
- Limits: at most `max_retries` (default 3) preemptive replies per turn and none once the user has
  been speaking for `max_speech_duration` (default 10 s) (`agent_activity.py:2655-2664`,
  `voice/turn.py:212-228`; the counter resets at `agent_activity.py:2823`). A long turn with several
  pauses uses them up; then the reply starts at the commit as without the adapter.
- Cost: one STT request and possibly one LLM call per pause that was not the end.

### 3. Endpointing delay and `min_delay`

`audio_recognition.py:1522-1523` takes "the user stopped speaking" as the `END_OF_SPEECH` time minus
its `silence_duration`, and `:1830-1839` sleeps until that plus `min_delay`. A VAD that reports the
stop where the audio went quiet (about 80 ms earlier than Silero's last voiced frame) moves that
bound by the same amount. It decides the commit only when it is later than both `END_OF_SPEECH`
and the final transcript: with `min_delay` 0.5 s and 0.25 s of VAD silence it does (the
"Inside an AgentSession" table: about 70 ms); with the Silero plugin's default 0.55 s of silence
`END_OF_SPEECH` itself comes later than `stop + 0.5 s`, and with a transcript that arrives later
still it does not. `min_delay` itself is the application's setting; a plugin cannot shorten it.

### 4. Everything else

Rows 4 to 7 of the table: none of them can start work early. Row 5 is the only other place that
reads the raw silence counter, and it only lets an already scheduled reply play.

## What was not found

- No hook that runs application code at a VAD pause. `vad.on("pause", ...)` on this plugin is the
  plugin's own event; LiveKit does not know it.
- No way for a VAD plugin to start preemptive generation, the turn commit or the LLM directly.
- No supported way to make a streaming STT without `manual_flush` finalize early from outside.
