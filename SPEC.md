# RealtimeVAD spec

Voice activity detection for voice assistants. It finds where speech starts
and stops in a live audio stream, and reports the stop as soon as the speaker
actually goes quiet.

## Input
- 16 kHz mono PCM, streamed in chunks of any size.
- Optionally, a flag on each chunk that says the assistant's own audio is playing (possible echo).

## Events
- **onset** (optional): speech may have begun. It comes at the first confident speech frame, so work can start early; a click or a knock can cause one too. Only **start** confirms it.
- **start**: speech has begun. It needs about 64 ms of confident speech, so short noises and single blips don't trigger it. That debounce is configurable, also while running: its length, and the speech probability every frame of it must keep (by default the continue threshold). A longer, stricter one rejects clicks and knocks, whose probability fades out within a few frames; the segment's audio still reaches back to the onset.
  - Speech the model is never confident about (a short "mm-hmm", a quiet or narrowband voice) starts a segment too, once the model has held half confidence for about 0.13 s over audio clearly louder than the room. This soft start is never weaker than the debounce above (at least as long, every frame at or above the same probability), it never fires on steady noise the model is merely unsure about, and it can be turned off.
- **pause**: the speaker may have stopped. It comes at the first sign of silence (within ~10 ms) so speculative work can begin early. It doesn't end the speech.
- **end**: the speech segment is over. Its timing:
  - It comes 0.24 s after the audio went quiet (0.28 s when room noise has
    raised what counts as quiet, since a word's fading tail then already
    counts), or 0.24 s after the speech model notices the silence, whichever
    is first. So it's never later than the speech model alone would end it.
  - Speech that resumes before then cancels it, even very soft speech.
  - A short click or noise doesn't end the segment. It starts the quiet count
    again (the end needs 0.24 s of unbroken quiet after it) and leaves the
    speech model's own count alone, so the end still comes no later than the
    model alone would give it.
  - "Quiet" is judged against the room's noise level, so **pause** and the
    early end work in noisy rooms too. The noise level is learned only from
    audio that isn't speech, and soft speech is never treated as silence.
  - "Quiet" is also judged against the speaker: it is at least 20 dB below
    their own level (the median of the running segment's speech), also when
    that is below the fixed quiet level. A quiet speaker's soft word endings
    therefore don't end the segment, at any input level. Where room noise is
    less than that far below the speaker, nothing counts as quiet: the early
    end is lost there and the speech model alone ends the segment.
  - "Quiet", the room's noise level and the speaker's level are measured
    without a constant (DC) offset in the input: an offset neither hides the
    silence nor counts as sound, and the timing is the same with and without
    it. Only these level measurements are affected; the speech model and the
    returned audio get the input as it came.
  - An optional external detector, such as semantic end-of-turn detection,
    may extend the wait. It may also shorten an extension it asked for
    earlier, but never below those base waits.
  - A segment ends anyway after 120 s.

## Audio
- The current segment's audio is available at any time while it runs:
  everything from just before speech started up to the latest sample fed. A
  client can stream from **start** onwards, or decode speculatively at a
  **pause**, without keeping its own buffer.
- About 0.3 s of audio before the start is always included.
- A soft or quiet opening, up to 1 s long, still belongs to the segment, as
  long as it's clearly louder than the room noise. In loud noise, where
  speech itself is only a few dB above it, "clearly" scales with that gap.
- The audio never reaches back into the assistant's own playback (or the 0.5 s after it), or into a previous segment.

## Segment output
Delivered with **end**:
- The segment's audio.
- The start time, the end time and the time the speech stopped.
- Total voiced time and peak speech confidence.

## Interfaces
- Native: feed chunks and get back the events, in order.
- Drop-in: a wrapper with the same constructor and call signature as
  Silero's `VADIterator`, honoring its parameters and returning its
  `{'start': …}` / `{'end': …}` results. Switching to it is a change of
  import, and the early end comes with it.

## Constraints
- Streaming. No delay beyond the 10 ms quiet block or the model's own frame.
- CPU only, one thread, far faster than realtime. No network and no threads of its own.
- Works on endless streams: if the speech model stops responding (it can lock
  up after long low-level noise or near-silent pauses), it recovers by itself:
  once per pause after about a second of hearing nothing over real audio, or
  within ~0.2 s when it ignores audio as loud as the speaker. Near-silence never
  triggers a recovery, and none happens during the assistant's own playback.
- A recovery never makes it hear speech that isn't there: it returns to how the
  model sounded after the *user's* speech, never after the assistant's own voice.
- The assistant's own voice (echo) never starts a segment by way of a recovery,
  and silence after the assistant speaks never produces a phantom segment.
- A recovery comes on the same frame with and without the early end, so the
  speech model's probabilities don't depend on the quiet clock.
- Optionally, the audio the speech model sees is amplified for quiet speakers
  (up to 20 dB, never into clipping). Everything else, including the audio
  that is returned, uses the input as it came. Off by default.
- Every threshold and duration above can be configured.
