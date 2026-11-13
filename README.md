# RealtimeVAD

*An early "the speaker stopped" signal for voice assistants: 92 ms before Silero by default, 130 ms in fast mode, on a close microphone. None on a distant one.*

[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) · [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) · [Live demo](demo/README.md) · [Results](docs/results.md) · [Quiet and noisy input](docs/quiet-and-noisy-input.md) · [Spec](SPEC.md) · [LiveKit plugin](livekit-plugins-realtimevad/README.md)

A voice assistant can't answer until it knows the user has stopped talking.
Speech recognition, the LLM call and the voice output all wait for that one
decision, and a voice activity detector takes a few hundred milliseconds to
make it.

RealtimeVAD gives you a second, earlier signal. It keeps
[Silero VAD](https://github.com/snakers4/silero-vad) to decide what is speech,
and every 10 ms it compares the audio level with the room's noise and with the
speaker's own level. The moment the audio goes quiet you get a `pause` event,
before Silero's probability has fallen. You start the answer on `pause` and
cancel it if the speaker goes on.

- **Earlier:** `pause` comes a median 92 ms before Silero's probability drops, and 238 ms before Silero's `VADIterator` reports the end (3,153 real voice-assistant turns, one speaker).
- **Or faster:** `pause_mode="fast"` leads by 130 ms on the same turns and more often fires inside the last word ([your choice](#earlier-or-more-accurate-pause_mode)).
- **Other speakers:** 82 to 90 ms on 6,600 close-microphone clips from 214 speakers, without noise suppression.
- **Quiet, noisy and phone input:** 84 to 122 ms on quiet speech, steady room noise and a phone line (simulated from the same turns).
- **Nothing lost at the end:** `end` comes exactly where Silero alone puts it. No turn is cut that Silero would not cut.
- **Small:** one CPU thread, about 8 ms per second of audio. Needs only numpy and onnxruntime; the Silero model is bundled, no torch.
- **Drop-in:** a `VADIterator` with Silero's interface, and a plugin for LiveKit Agents.

## Is it for you?

RealtimeVAD only pays off in one kind of pipeline, and the lead depends on
the audio. Check both before you build on it.

**You need a pipeline that can start early and cancel fast.** `pause` fires in
every short gap between words, about twelve times as often as Silero's own
speech-off. Most pauses are followed by more speech. The gain comes from
starting the transcript, the LLM call or the voice output at `pause` and
throwing that work away when the speaker continues (speculative generation).

| Your pipeline | What you get |
|---|---|
| Starts work at `pause` and cancels it within a few tens of milliseconds | The lead below, on every turn where the audio goes quiet |
| Waits for the end of the turn before doing anything | Nothing. `end` is Silero's end. Use Silero |
| Starts early but cannot cancel, or each start is expensive | More cost than gain: up to 1.7 times as many speculative requests in our LiveKit runs. Use Silero |
| Already answers faster than the end of the turn arrives (fast streaming STT and LLM) | Little or nothing: 0 ms in our LiveKit run with a 0.1 s LLM |

**The lead depends on the input.** It needs audio that goes quiet after a
word. Median lead of the last `pause` over Silero's speech-off, default
settings:

| Input | Median lead | Earlier than Silero in |
|---|---|---|
| Close microphone with noise suppression, one speaker | 92 ms | 92 % of turns |
| Read speech, raw microphone (LibriSpeech) | 82 – 84 ms | 84 – 85 % |
| Meeting headset, speaker alone (AMI) | 90 ms | 80 % |
| Meeting headset, others talking (AMI) | 86 ms | 69 % |
| Quiet speaker, clean, −38 to −52 dBFS | 100 – 122 ms | 91 – 95 % |
| Steady room noise 15 dB under the speaker (simulated) | 84 ms | 75 % |
| Phone line, −38 to −58 dBFS (simulated) | 92 – 114 ms | 79 – 87 % |
| **Distant table microphone (AMI)** | **0 ms** | 31 % |
| **Steady noise within about 11 dB of the speaker** | **none** | – |
| **Babble, music or street noise close under the speaker** | **expect none** (not measured; the noise subtraction does not apply) | – |

Where there is no lead, RealtimeVAD behaves like Silero alone: correct, and
no faster. The quiet-speaker, room-noise and phone-line rows were measured
before the last change to `pause` (sound in the trailing silence no longer
repeats it); that change can only make their lead equal or longer.

## Live demo

https://github.com/user-attachments/assets/d01900dd-9601-4f3d-8c88-db55e8bc83a1

The video was recorded with 0.1.0 and a synthetic voice and shows a median lead of 78 ms over its three turns.

[demo/](demo/README.md) runs
this page on your own microphone. It shows, turn by turn, how much sooner
RealtimeVAD knew that you stopped than Silero alone, on the same audio. It is
the quickest way to see what your microphone and room give you.

## Quick start

```bash
pip install realtimevad
```

```python
from RealtimeVAD import VAD

vad = VAD(resume_events=True)
for chunk in microphone_chunks():            # 16 kHz mono, any chunk size
    for event in vad.feed(chunk):
        if event.kind == "start":
            stream_to_asr(vad.audio())       # includes the audio just before speech began
        elif event.kind == "pause":
            speculate(vad.audio())           # the speaker may be done: start the answer
        elif event.kind == "resume":
            cancel_speculation()             # the speaker went on
        elif event.kind == "end":
            finalize(event.segment.audio)    # the turn is over
```

Input is 16 kHz mono audio as float32, int16 or int16 bytes.

## Events

| Event | Meaning | When |
|---|---|---|
| `start` | Speech has begun | After about 64 ms of confident speech, so clicks don't trigger it. Speech Silero is never confident about (a short "mm-hmm", a quiet phone voice) starts after about 0.13 s at half confidence, if it is clearly louder than the room |
| `pause` | The speaker may have stopped (speech-off) | Within about 10 ms of the audio going quiet. Again when sound came back and the audio goes quiet again |
| `end` | The segment is over | Where Silero alone ends it: after 0.24 s of Silero silence. Never later than Silero alone |
| `resume` | The speech went on after a `pause` | About 20 ms after sound is back. Off by default: `VAD(resume_events=True)` |
| `onset` | Speech may have begun | At the first confident frame, before `start` confirms it. Off by default: `VAD(onset_events=True)` |

- **Audio:** `vad.audio()` returns the running segment at any time, starting about 0.3 s before the speech.
- **Streaming speech recognition:** `start` comes about 64 ms after the speech began, so audio sent only from `start` on has lost the first sound. At `start`, send `vad.audio()` first (it holds the beginning and the 0.3 s before it), then every later chunk.
- **`end`** carries the finished segment with its audio, start time, end time and the time the speech stopped.
- **Assistant talking:** pass `vad.feed(chunk, playback=True)` for chunks recorded while your assistant speaks, so its own voice is handled as possible echo.
- **Guarantees:** [SPEC.md](SPEC.md) lists what is guaranteed and every setting.

## How to use `pause`

`pause` is early and often wrong, on purpose. It fires on every breath, every
short gap between words and every stop consonant.

- **Start work you can cancel:** finalize the transcript, call the LLM, prepare the voice output.
- **Cancel at `resume`.** It comes 20 ms after sound is back, a median 50 ms after its `pause`, and takes back 90 to 95 % of the pauses that were followed by more speech.
- **Redo at the next `pause`; the latest one counts.** A `pause` can come inside the last word, when the level dips before a final consonant. The next quiet gives a new `pause` with more audio. Drop the older result.
- **Never treat `pause` as the end of the turn.** Only `end` closes a segment.
- **Keep `end` as the fallback.** In 1 to 4 % of clips the last `pause` is taken back by a `resume` and no new one comes before the `end`. Then `end` is the signal to act on.
- **Sound after the speech takes nothing back.** Once Silero has been counting silence longer than its lag, a breath or a click gives no new `pause` and no `resume`: the standing `pause` stays the last one.

## Earlier or more accurate: `pause_mode`

One setting chooses how early `pause` comes against how often it comes too
early, inside the last word. The default is the accurate one.

```python
vad = VAD(pause_mode="fast")   # "accurate" (default), "balanced" or "fast"
```

| `pause_mode` | Median lead over Silero | Last `pause` more than 100 ms too early | … more than 200 ms | `pause` followed by more speech, per s |
|---|---|---|---|---|
| `"accurate"` (default) | 82 – 90 ms (one speaker: 92 ms) | 1 – 9 % | 0.1 – 3 % | 1.7 – 2.2 |
| `"balanced"` | 102 – 114 ms (112 ms) | 11 – 17 % | 2 – 7 % | 1.7 – 2.0 |
| `"fast"` | 118 – 126 ms (130 ms) | 18 – 24 % | 4 – 10 % | 2.0 – 2.2 |

The ranges cover four close-microphone sets from 214 speakers; the figure in
brackets is the 3,153 one-speaker turns.

- **What `"fast"` costs:** in one of five turns the last `pause` comes more than 100 ms before the last word is over. Audio cut at that `pause` misses the end of the word.
- **When to use it:** when what you start at `pause` copes with a missing word ending, or is redone at `end`.
- **What stays the same:** `start`, `end` and the segment's audio, in every mode.

Details and the full table: [results.md](docs/results.md#speech-off-pause).

## Results

| Measured on | Result |
|---|---|
| 3,153 turns, one speaker: `pause` before Silero's probability falls below 0.5 | median 92 ms (p25 – p75: 72 – 112 ms), earlier in 92 % |
| Same turns: `pause` before Silero's `VADIterator` end | median 238 ms, earlier in 99 % |
| Same turns: `pause` followed by more speech | 1.8 per second of speech (Silero: 0.16) |
| 6,600 close-microphone clips, 214 speakers: last `pause` more than 100 ms before the word ends | 1 – 2 % of read sentences, 6 – 9 % of meeting clips |
| All 7,080 clips and all quiet, noisy and phone rows: `end` | same as Silero alone, never later |
| CPU, one thread, Silero included | 8.3 ms per second of audio |

- **Everything else:** [docs/results.md](docs/results.md) has the full tables, the comparison with 0.0.5 and the `early_end` option.
- **Quiet, noisy, phone:** [docs/quiet-and-noisy-input.md](docs/quiet-and-noisy-input.md) has every level and line we measured, backchannels and DC offset.
- **Your own audio:** `python benchmarks/speech_off_lead.py <wav dir> out.json` measures the lead on 16 kHz mono WAVs, one turn each, with trailing silence.

### Why the headline number changed

This page said 110 ms at first, then 100 ms, now 92 ms. The detector did not
get slower: each release moved the default toward fewer mistakes, and the
earlier behaviour is still there as a setting.

| Version | Median lead, one speaker | What the default does |
|---|---|---|
| 0.0.4 | 110 ms | One `pause` per stretch of speech, fixed quiet floor. Cut quiet speakers: 11 % of turns ended while talking at −38 dBFS |
| 0.0.5 | 100 ms | The quiet floor follows the speaker, soft speech starts a segment. Quiet speakers are no longer cut |
| now, default | 92 ms | A `pause` inside the last word is followed by a better one, and sound in the trailing silence does not repeat it. Last `pause` too early in 2.2 % of read sentences, was 7.6 % |
| now, `repause_s=None` | 102 ms * | One `pause` per stretch of speech, as in 0.0.5 |
| now, `pause_mode="fast"` | 130 ms | Earlier than any release so far, at the price in the table above |

- **The first `pause` of a turn comes as early as before.** The number fell because the benchmark measures the last one, which is now the later, better one.
- **Not the same turns each time.** The set grew and was filtered differently (3,238, 3,279, 3,118 and now 3,153 turns), so the step from 110 to 100 ms is not attributed to one change; the step from 100 to 92 ms is, see [results.md](docs/results.md#speech-off-pause).
- \* The `repause_s=None` figure was measured on the 3,118 turns, before sound in the trailing silence stopped repeating a `pause`. The 0.0.4 and 0.0.5 rows are what those releases published.

## Limits

- **It is a signal for speculative pipelines only.** Without fast cancelling it gives you Silero's result with more events to ignore.
- **No lead on a distant microphone or in loud noise.** See the table in [Is it for you?](#is-it-for-you).
- **Earlier is not the same as correct.** The last `pause` is more than 100 ms early in 1 to 2 % of read sentences and 6 to 9 % of meeting clips, with audible sound of the last word still to come. Silero's own speech-off is that early in 0.2 to 0.3 % of read sentences: it is late, and rarely early.
- **`end` has no lead.** It is Silero's end. `VAD(early_end=True)` ends a median 16 to 70 ms sooner and ends more segments inside an utterance (2 to 3 points more clips).
- **Silero's limits remain.** Whisper is mostly not detected. Laughter, coughs and sneezes often are. Very quiet and narrowband speech makes Silero unsure, and RealtimeVAD with it.
- **16 kHz mono only.** For 8 kHz audio without resampling, use Silero.
- **Thin evidence in places.** The headline numbers come from one speaker and one capture chain (browser capture with noise suppression and echo cancellation, mostly quiet rooms). The room noise and the phone line are simulated. The LiveKit timings are one repetition on 10 to 16 sentences.
- **Alpha.** The API may still change.

## Settings

| If … | then |
|---|---|
| you want `pause` earlier and accept that it is more often inside the last word | `VAD(pause_mode="balanced")` or `"fast"` |
| you want to know when the speech went on after a `pause` | `VAD(resume_events=True)` |
| you want the `end` sooner and accept more ends inside an utterance | `VAD(early_end=True)` |
| you want one `pause` per stretch of speech, as in 0.0.5 | `VAD(repause_s=None)` |
| your input is quiet or a phone line, and clean | try `VAD(normalize_to_dbfs=-30)` |
| soft starts open segments you don't want (background voices) | `VAD(soft_start_s=None)` |
| you want Silero only, no quiet clock at all | `VAD(quiet_rms=0)` |

More settings and what each one costs: [quiet-and-noisy-input.md](docs/quiet-and-noisy-input.md#steady-noise-close-under-the-speaker-after-005).

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
not Silero's first low frame. Without a model argument it uses the bundled
Silero model and doesn't need torch.

This interface has no `pause`, so on its own it gives you no speed. Use `VAD`
for the early signal.

## Frameworks

- **LiveKit Agents:** [livekit-plugins-realtimevad/](livekit-plugins-realtimevad/README.md) is a VAD plugin plus a wrapper that starts LiveKit's preemptive generation at `pause`. LiveKit alone starts it only on a final transcript. With the wrapper the first agent audio came 120 ms sooner than with Silero in the same wrapper on Whisper, 56 ms on a streaming recognizer, and 0 ms with a 0.1 s LLM (at the wrapper's default `pause` of 0.15 s).
- **Pipecat:** not supported. Its speech recognizers finalize only when the VAD reports the stop, so nothing can start at `pause` ([issue 4](https://github.com/KoljaB/RealtimeVAD/issues/4)).

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

They take about 10 s with 8 worker processes and print what each test guards,
with the measured value and its limit. Sources and licenses:
[LICENSES-audio.md](tests/audio/LICENSES-audio.md),
[LICENSES-backchannels.md](tests/audio/LICENSES-backchannels.md).

## License and credits

MIT. The bundled speech model is [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
