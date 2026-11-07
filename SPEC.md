# RealtimeVAD spec

Voice activity detection for voice assistants. It finds where speech starts
and stops in a live audio stream, and reports the stop as soon as the speaker
actually goes quiet.

## Input
- 16 kHz mono PCM, streamed in chunks of any size.
- Optionally, a flag on each chunk that says the assistant's own audio is playing (possible echo).

## Events
- **start**: speech has begun. It needs about 64 ms of confident speech, so short noises and single blips don't trigger it.
- **pause**: the speaker may have stopped. It comes at the first sign of silence (within ~10 ms) so speculative work can begin early. It doesn't end the speech.
- **end**: the speech segment is over. Its timing:
  - It comes 0.24 s after the speech stopped, counted from when the audio
    went quiet, not from when a speech model notices.
  - Speech that resumes before then cancels it, even very soft speech.
  - A short click or noise doesn't end the segment, and doesn't restart the silence count either.
  - "Quiet" is judged against the room's noise level, so **pause** and the
    early end work in noisy rooms too. The noise level is learned only from
    audio that isn't speech, and soft speech is never treated as silence.
  - An optional external detector, such as semantic end-of-turn detection,
    may extend the wait. It may also shorten an extension it asked for
    earlier, but never below 0.24 s.
  - A segment ends anyway after 120 s.

## Audio
- The current segment's audio is available at any time while it runs:
  everything from just before speech started up to the latest sample fed. A
  client can stream from **start** onwards, or decode speculatively at a
  **pause**, without keeping its own buffer.
- About 0.3 s of audio before the start is always included.
- A soft or quiet opening, up to 1 s long, still belongs to the segment, as long as it's clearly louder than the room noise.
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
- Every threshold and duration above can be configured.
