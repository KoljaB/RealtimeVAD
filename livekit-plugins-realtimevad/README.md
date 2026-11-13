# RealtimeVAD plugin for LiveKit Agents

Makes a LiveKit agent start its reply at the user's pause, not at the VAD's
`END_OF_SPEECH`. A community plugin, not made by LiveKit. Two parts:

- **`SpeculativeSTT`** wraps your STT and asks it for the transcript at the
  pause. LiveKit then begins the reply while the VAD still waits for its end.
  Works with any LiveKit VAD, the Silero plugin included.
- **`VAD`** is [RealtimeVAD](https://github.com/KoljaB/RealtimeVAD) in
  LiveKit's VAD slot. It ends speech where the Silero plugin does and reports
  the pause about 100 ms sooner.

```python
from livekit.agents import AgentSession
from livekit.plugins import realtimevad

vad = realtimevad.VAD.load()                 # or silero.VAD.load()
session = AgentSession(
    vad=vad,
    stt=realtimevad.SpeculativeSTT(stt=your_stt, vad=vad),
    ...
)
```

```bash
pip install livekit-plugins-realtimevad
```

It installs `realtimevad` 0.1.0 or newer with it.

Plugin 0.1.0 needs RealtimeVAD 0.1.0 or newer: it relies on the end being
Silero's by default and on the repeated pause.

Tested with livekit-agents 1.8.11, Python 3.10+.

## What `SpeculativeSTT` does

LiveKit starts a reply early ("preemptive generation", on by default) as soon
as it has a final or preflight transcript. Nothing in LiveKit reads a VAD's
pause for that, so the pause is put to use on the STT side:

- A **non-streaming STT** (Whisper and the like; LiveKit would call it once at
  `END_OF_SPEECH`) gets the segment so far after `pause` seconds of raw silence.
  The text goes out as a preflight transcript, and again as the final at
  `END_OF_SPEECH` without a second request.
- A **streaming STT with `capabilities.manual_flush`** is flushed at the pause.
  One without `manual_flush` cannot be finalized from outside: `ValueError`.

It uses public LiveKit API only. [PAUSE.md](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/PAUSE.md) goes through LiveKit's
source with file and line: what reads VAD events, and why this is the one route
from a pause to an earlier reply. A full agent:
[examples/speculative_agent.py](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/examples/speculative_agent.py).

| Option | Default | |
|---|---|---|
| `stt` | | the STT to wrap |
| `vad` | | the VAD that finds the pauses, usually the session's |
| `pause` | 0.15 | seconds of raw silence inside speech before the transcript is requested |
| `min_new_speech` | 0.0 | pass over a pause when less speech than this came since the last request |

## Measured

A real `AgentSession` (`turn_detection="vad"`, `min_delay` 0.5 s), a real STT
running locally, fake LLM and TTS, audio at real-time pace. All four columns
wait 0.55 s of Silero silence for `END_OF_SPEECH`, so only the pause differs.
Each table says which RealtimeVAD core it was measured with: the source at
commit 60ed995 (what became 0.1.0: the end is Silero's by default, a pause can
be reported again; `pause_mode` came after it and its default is that
behaviour) or 0.0.5. Nothing was measured again for 0.1.0 itself. The Silero
columns do not depend on the core.

Core 60ed995, `pause=0.15` (the default), median ms after the true end of
speech, one repetition:

| First agent audio | Silero | Silero + `SpeculativeSTT` | RealtimeVAD | RealtimeVAD + `SpeculativeSTT` |
|---|---|---|---|---|
| faster-whisper base, 16 sentences in 8 languages, LLM first token 0.5 s | 2109 | 1725 | 2131 | 1586 |
| streaming sherpa-onnx, 10 English sentences, LLM first token 0.5 s | 1438 | 1053 | 1439 | 985 |
| the same, LLM first token 0.1 s | 1035 | 792 | 1059 | 812 |
| **Cost** over all sentences: STT requests, Whisper | 18 | 37 | 18 | 46 |
| … STT flushes / LLM calls, sherpa-onnx | 9 / 9 | 33 / 22 | 9 / 9 | 36 / 24 |

The two columns without the wrapper are from another run of the same day (they
do not depend on `pause`; one of its 10 sherpa-onnx sentences was not used).

- **The wrapper is most of the gain, and it does not need RealtimeVAD.** With
  the Silero plugin's own raw silence as the pause, the agent's audio comes
  about 385 ms sooner (240 ms with the fast LLM); 420 ms at `pause=0.1`.
- **RealtimeVAD's earlier pause adds to that:** 120 ms with Whisper (10th to
  90th percentile of the sentences: 3 to 172) and 56 ms (0 to 82) with the
  streaming recognizer, paired by sentence. With the fast LLM it adds nothing.
- **It pays with a little more speculative work:** 46 against 37 Whisper
  requests for 16 sentences, 36 against 33 flushes and 24 against 22 LLM calls
  for 10.
- With a fast LLM the turn commit becomes the limit: the reply is ready before
  `END_OF_SPEECH`, and LiveKit does not speak before it. A shorter
  `min_silence_duration` is what moves that.
- Without the wrapper the early pause does nothing in LiveKit.

### `pause` 0.15 against 0.1, core 60ed995 against 0.0.5

What RealtimeVAD's pause adds on top of the Silero plugin with the same
wrapper, one repetition per cell, the same plugin code for both cores
([benchmarks/speculative_check_by_core.txt](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/benchmarks/speculative_check_by_core.txt)).
Sooner: first agent audio, median ms (10th to 90th percentile of the
sentences). Work: RealtimeVAD against the Silero plugin.

| | core 60ed995, `pause` 0.15 | core 0.0.5, `pause` 0.15 | core 60ed995, `pause` 0.1 | core 0.0.5, `pause` 0.1 |
|---|---|---|---|---|
| Whisper, LLM 0.5 s: sooner | 120 (3 to 172) | 119 (-12 to 152) | 139 (40 to 173) | 139 (48 to 180) |
| … STT requests, 16 sentences | 46 vs 37 | 48 vs 37 | 66 vs 39 | 68 vs 39 |
| … LLM calls | 17 vs 17 | 17 vs 17 | 17 vs 17 | 17 vs 17 |
| sherpa-onnx, LLM 0.5 s: sooner | 56 (0 to 82) | 59 (33 to 82) | 88 (0 to 197) | 59 (16 to 199) |
| … flushes, 10 sentences | 36 vs 33 | 35 vs 33 | 47 vs 34 | 47 vs 34 |
| … LLM calls | 24 vs 22 | 23 vs 21 | 30 vs 22 | 30 vs 22 |
| sherpa-onnx, LLM 0.1 s: sooner | 0 | 0 | 0 | 0 |

- The new core changes neither the gain nor the cost beyond what one repetition
  can tell apart (times here come in steps of about 20 ms; 10 sentences).
- `pause=0.15` is the default because it keeps most of the gain with Whisper (120 of 139 ms) and nearly
  removes the extra work with the streaming recognizer (36 against 33 flushes).
- On core 60ed995 the plain RealtimeVAD column is within 25 ms of the Silero
  column too (first agent audio 2131 / 1439 / 1059 against 2109 / 1438 / 1035, no sentence
  split in one VAD only).

### Earlier runs: core 0.0.5, `pause` 0.1

Median ms after the true end of speech, 3 repetitions (they
agree within 30 ms):

| First agent audio | Silero | Silero + `SpeculativeSTT` | RealtimeVAD | RealtimeVAD + `SpeculativeSTT` |
|---|---|---|---|---|
| faster-whisper base, 16 sentences in 8 languages, LLM first token 0.5 s | 2110 | 1695 | 2091 | 1581 |
| streaming sherpa-onnx, 10 English sentences, LLM first token 0.5 s | 1424 | 1029 | 1397 | 964 |
| the same, LLM first token 0.1 s | 1038 | 801 | 995 | 768 |
| **Cost** over all sentences: STT requests, Whisper | 17 | 38 | 17 | 63 |
| … STT flushes / LLM calls, sherpa-onnx | 10 / 10 | 34 / 22 | 11 / 11 | 47 / 29 |

- The wrapper: about 420 ms sooner (240 ms with the fast LLM).
- RealtimeVAD's earlier pause on top: 140 ms with Whisper (10th to
  90th percentile of the sentences: 41 to 195) and 59 ms (20 to 227) with the
  streaming recognizer. With the fast LLM it adds nothing: the 27 ms in the
  table were an earlier `END_OF_SPEECH`, see below.
- More speculative work: 63 against 38 Whisper requests for 16
  sentences, 47 against 34 flushes and 29 against 22 LLM calls for 10.

In these runs RealtimeVAD's `END_OF_SPEECH` still came about 30 ms before the
Silero plugin's (the 20 to 40 ms in the "RealtimeVAD" column), and one English
sentence with a 0.58 s gap became two turns with RealtimeVAD only. Both had one
cause, fixed since: see "The VAD" below. One more repetition after the fix
([benchmarks/speculative_check_after_fixes.txt](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/benchmarks/speculative_check_after_fixes.txt)):

| First agent audio, after the fix, core 0.0.5 | Silero | Silero + `SpeculativeSTT` | RealtimeVAD | RealtimeVAD + `SpeculativeSTT` |
|---|---|---|---|---|
| faster-whisper base, LLM first token 0.5 s | 2121 | 1708 | 2115 | 1561 |
| streaming sherpa-onnx, LLM first token 0.5 s | 1437 | 1005 | 1437 | 954 |
| the same, LLM first token 0.1 s | 1055 | 795 | 1055 | 799 |
| `END_OF_SPEECH`, Whisper / sherpa-onnx | 691 / 703 | 692 / 704 | 691 / 708 | 691 / 708 |
| STT requests started, Whisper | 18 | 39 | 19 | 66 |
| STT flushes / LLM calls, sherpa-onnx | 10 / 10 | 34 / 22 | 10 / 10 | 47 / 31 |

The ends are level, no sentence splits in one VAD only, and the plain
RealtimeVAD column equals the Silero column. The wrapper gains 420 to 440 ms
(240 ms with the fast LLM); RealtimeVAD's pause adds 139 ms with Whisper (49 to
180), 42 ms with the streaming recognizer (22 to 203) and 0 ms with the fast
LLM.

One Hindi sentence had no reply in any column of the first runs because the
benchmark closed the session before Whisper was done; it now waits (Whisper
base needs 6 to 13 s for that sentence, so its times stay out of the medians).
Preemptive generation ran with `max_retries` 10 and `max_speech_duration` 30 s
in all columns. Full output of the three repetitions with the machine's load
during each run: [benchmarks/speculative_check.txt](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/benchmarks/speculative_check.txt).

```bash
python benchmarks/speculative_check.py r1.json 1 sherpa     # models: see benchmarks/local_stt.py
python benchmarks/speculative_check.py report r*.json
```

## What a wrong pause costs

A pause is often not the end. When speech goes on:

- The request is cancelled if it still runs, and its transcript is never sent.
  The next pause asks again with all the audio so far: the newest pause always
  replaces an older one. RealtimeVAD (since 0.1.0) reports a
  pause again when sound follows the first; the wrapper asks again only while
  Silero has not counted `pause` of silence itself. A noise after the last word
  does not throw the transcript away (it did in a first version: 4 of 16
  sentences came 120 to 480 ms later).
- A reply LiveKit already began from a transcript is replaced by the next
  transcript or discarded when the turn is committed. It is never spoken.
- A streaming STT that was flushed has finalized a fragment. LiveKit joins the
  fragments of a turn, but the recognizer starts the next one without the
  context of the first.
- LiveKit starts at most `max_retries` (default 3) preemptive replies per turn
  and none once the user has spoken for `max_speech_duration` (default 10 s).
  A long turn with several pauses uses them up; then the reply starts at the
  commit, as without the wrapper. Raise them in
  `turn_handling={"preemptive_generation": {...}}`.

So the cost is one STT request, and possibly one LLM call, per pause that was
not the end.

### Fewer requests

`pause` is the knob. Replayed over 24 sentences (165 s of speech, no STT; the
requests the wrapper would make, and when the one that becomes the reply is
made, in ms of audio after the true end of speech). RealtimeVAD core 60ed995,
core 0.0.5 in brackets where it differs:

| `pause` | Silero plugin: requests per sentence | … last request | RealtimeVAD: requests per sentence | … last request |
|---|---|---|---|---|
| 0.05 | 2.33 | 204 | 6.79 (6.62) | 92 |
| 0.1 | 2.29 | 268 | 3.79 (3.83) | 140 |
| 0.15 (default) | 2.21 | 300 | 2.67 | 204 (220) |
| 0.2 | 2.00 | 364 | 2.33 (2.29) | 252 (284) |
| 0.3 | 1.58 | 460 | 2.00 (1.92) | 348 (380) |

- RealtimeVAD with `pause=0.15` or 0.2 makes 2.3 to 2.7 requests per sentence,
  about as many as the Silero plugin at 0.05 or 0.1 (2.3), and asks at 204 to
  252 ms where the Silero plugin asks at 204 to 268 ms: at equal cost the
  earlier pause is worth little. Its gain is in the range the Silero plugin
  cannot reach.
- The Silero plugin's raw silence cannot ask before about 200 ms after the
  speech. RealtimeVAD's can (140 ms, 92 ms), at 1.7 to 3 times the requests.
- `min_new_speech` saves less and loses more: with RealtimeVAD at `pause=0.1`,
  0.3 s takes the requests from 3.79 to 3.29 and passes over the last pause of
  2 of the 24 sentences, which are then transcribed at `END_OF_SPEECH`; 1.0 s
  gives 2.25 and loses 7.

Full tables: [benchmarks/pause_requests.txt](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/benchmarks/pause_requests.txt), core 0.0.5: [benchmarks/pause_requests_core_0.0.5.txt](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/benchmarks/pause_requests_core_0.0.5.txt)
(`python benchmarks/pause_requests.py out.txt`).

## The VAD

`realtimevad.VAD.load()` replaces `silero.VAD.load()` without changing where
speech ends:

- **`END_OF_SPEECH` where the Silero plugin puts it.** Same default, 0.55 s of
  Silero silence, counted the way the Silero plugin counts (it smooths the
  probability, which makes its end one or two 32 ms frames later than a count
  on the raw probability; a gap of just that length would otherwise end a
  segment here and not there). Over the 24 sentences: median 684 ms after the
  speech for both.
- **The pause is early.** `raw_accumulated_silence` of the `INFERENCE_DONE`
  events counts from the instant the audio went quiet, about 100 ms before
  Silero's probability falls. That is what `SpeculativeSTT` reads, and what
  LiveKit starts its turn detector on at 0.2 s.
- **The stop is reported where it happened.** `silence_duration` of
  `END_OF_SPEECH` reaches back to that instant.
- **Stricter start.** A segment starts on Silero probability 0.9 (the Silero
  plugin: 0.5), or after 0.13 s at 0.5 over audio clearly louder than the room.
  Fewer false starts in noise; `START_OF_SPEECH` is later on soft openings (p90
  of 568 against 248 ms on 60 real meeting turns) and a few more short, quiet
  sounds are missed (24 against 19 of 200 backchannels). The opening's audio is
  still in the segment and `speech_duration` counts it, so interruptions come
  as soon as with Silero.

Why 0.55 s: it is the Silero plugin's default, so swapping the VAD changes no
segment and no turn. The end is the safe signal and may be slow, because the
reply no longer waits for it. A shorter end splits sentences into segments
(1.6 to 1.8 per sentence at 0.25 s), and each is a transcript LiveKit may
commit as a turn.

| RealtimeVAD | LiveKit | |
|---|---|---|
| `start` | `START_OF_SPEECH` | `speech_duration` reaches back to where the speech began; `frames` hold 0.5 s of padding plus the opening |
| `end` | `END_OF_SPEECH` | `silence_duration` reaches back to the instant the audio went quiet; `frames` hold the whole utterance |
| each 32 ms frame | `INFERENCE_DONE` | Silero's probability, the frame's audio at the input sample rate, the raw counters |
| `pause` | (none) | Runs into `raw_accumulated_silence` and stops `raw_accumulated_speech`. Also `vad.on("pause", callback)` |
| `onset` | (none) | `vad.on("onset", callback)` |

`flush()` is a hard boundary, as LiveKit asks: the running speech is dropped
without an `END_OF_SPEECH` and what the VAD learned about the room is forgotten.

### Options

| Option | Default | Silero plugin | |
|---|---|---|---|
| `min_silence_duration` | 0.55 | 0.55 | LiveKit's audio turn detector refuses less than 0.25 |
| `early_end` | `False` | | `True`: also end after `min_silence_duration` of quiet audio, see below |
| `pause_mode` | `"accurate"` | | RealtimeVAD's `pause_mode`; `"balanced"`, `"fast"`: an earlier pause that is more often inside the last word, see below |
| `min_speech_duration` | 0.064 | 0.05 | Both come to two 32 ms frames |
| `prefix_padding_duration` | 0.5 | 0.5 | |
| `activation_threshold` | 0.9 | 0.5 | |
| `deactivation_threshold` | 0.45 | 0.35 | |
| `max_buffered_speech` | 60 | 60 | Longest speech kept in the `END_OF_SPEECH` frames |

Any other keyword goes to `RealtimeVAD.VAD`, for example `soft_start_s=None`,
`floor_margin_db`, or `normalize_to_dbfs=-30` for quiet phone audio.

`vad.update_options(...)` changes the six options the Silero plugin also has
while streams run. A running stream keeps what it has learned; a new
`min_silence_duration` counts from the next pause.

`pause_mode` is fixed when the VAD is made (not in `update_options`). It moves
the pause only, never `START_OF_SPEECH` or the default `END_OF_SPEECH`. With
`SpeculativeSTT`:

- `"balanced"` and `"fast"` count audio as quiet 3 dB and 6 dB sooner, so the
  request is made earlier and more often before the last word is over.
- In those modes the core does not report a pause again after a short sound.
  The wrapper does not depend on that event: it reads the raw silence, and
  sound that breaks the quiet count before Silero has counted `pause` of
  silence still cancels the request and asks again (tested in all three modes).
  But softer sound no longer breaks the count, so a request that was too early
  is taken back less often and later; what is left is Silero hearing the speech
  again.
- All numbers in this README are from the `"accurate"` behaviour. What the
  faster modes gain or cost in a LiveKit session is not measured.

Not there: 8 kHz inference (input at any rate is resampled to 16 kHz).

### When the agent speaks

LiveKit gives a VAD audio frames and nothing else. To use RealtimeVAD's
`playback` flag, tell the plugin yourself:

```python
session.on("agent_state_changed",
           lambda ev: vad.set_agent_speaking(ev.new_state == "speaking"))
```

While it is set, and for 0.5 s after, RealtimeVAD does not learn the room's
noise level from the audio. Speech is detected as before: the user can still
interrupt.

### `early_end=True`

The segment also ends after `min_silence_duration` of quiet audio, without
waiting for Silero. Measured on the plugin's events, sentences in 8 languages,
median ms after the speech stopped:

| | Silero default (0.55 s) | Silero 0.25 s | RealtimeVAD early end 0.25 s |
|---|---|---|---|
| `END_OF_SPEECH` | 684 | 364 | 295 |
| Segments per sentence | 1.0 | 1.62 | 1.79 |

In an `AgentSession` with a fake STT that answers at once, the turn was
committed about 70 ms sooner than with Silero at 0.25 s. With a real streaming
STT that finalizes on its own endpoint (about 940 ms after the speech) the
median gain was 0 ms: the turn waits for the transcript, not for the VAD
([benchmarks/agent_session_real_stt.txt](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/benchmarks/agent_session_real_stt.txt),
one run). The early end pays off only when the transcript is there before it,
and it ends more segments inside a sentence. That is why it is an option.

```bash
python benchmarks/compare_silero.py out.json        # needs the repository's test audio
python benchmarks/agent_session_check.py out.json   # needs a checkout of livekit/agents for its test fakes
pytest
```

## Limits

- The example agent has not been run against a LiveKit server (it needs one,
  and API keys). The measurements use LiveKit's `AgentSession` with local
  models and fake LLM and TTS.
- LiveKit's cloud turn detector was not tested (`turn_detection="vad"`
  throughout). From the source, an earlier pause starts its request sooner but
  does not move the commit ([PAUSE.md](https://github.com/KoljaB/RealtimeVAD/blob/main/livekit-plugins-realtimevad/PAUSE.md)).
- One speaker per clip, no echo, no real room.
- The three-repetition session table was measured before `END_OF_SPEECH` was
  aligned with the Silero plugin's count; after it there is one repetition.
- Cancelling a superseded request stops waiting for it. Whether the work stops
  is up to the STT: an HTTP request is dropped, a local model in a thread runs
  to its end.

## Pipecat

Not supported: Pipecat's STTs finalize only on the VAD's stop, so a pause
cannot start speculative generation there
([#4](https://github.com/KoljaB/RealtimeVAD/issues/4), closed as not planned).

## License

MIT. Uses LiveKit Agents (Apache-2.0) and the Silero VAD model (MIT).
