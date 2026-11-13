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
  - A short first word is kept. With a longer debounce a short word ("Hey") does not last long enough to start a segment by itself. When speech that follows it is confirmed, the start reaches back to it: speech of at least 64 ms that stopped less than the end silence (0.24 s) before belongs to the segment, and the 0.3 s of audio in front are counted from there. A short sound that stands alone still starts nothing, and nothing is taken from audio marked as the assistant's own voice. With the default debounce this rule never applies.
  - Speech the model is never confident about (a short "mm-hmm", a quiet or narrowband voice) starts a segment too, once the model has held half confidence for about 0.13 s over audio clearly louder than the room. This soft start is never weaker than the debounce above (at least as long, every frame at or above the same probability), it never fires on steady noise the model is merely unsure about, and it can be turned off.
- **pause**: the speaker may have stopped. It comes at the first sign of silence (within ~10 ms) so speculative work can begin early. It doesn't end the speech.
  - It is never held back to make it more certain.
  - Several can come before one **end**. When sound comes back after a **pause** (at least 20 ms that are over the quiet level and clearly above the room's noise) and the audio goes quiet again, a new **pause** is reported. The latest one is the best estimate of where the speech stopped; a client that started speculative work on an earlier one should redo it.
  - Steady room noise alone does not cause a new **pause**. Neither does sound in the trailing silence: it counts only while the speech model still takes the audio for speech, or stopped doing so no longer ago than its lag (about 0.13 s), so that a dip inside the last word still gives a new **pause** while a breath or a click after the turn does not. A frame the model hears as speech again allows the next one.
  - One setting trades how early it comes against how often it is too early, in three steps. The default is the most accurate one. The two faster ones report one **pause** per stretch of speech (none is repeated) and count audio as quiet 3 dB or 6 dB sooner, so a word's fading end already counts and more pauses fall inside the last word. The setting changes **pause**, **resume** and the time a segment gives for the speech having stopped. It never changes **start**, and never an **end** that the speech model alone decides (the default), so not what audio a segment holds either. The optional early end follows the quiet clock and moves with it.
- **resume** (optional): the speech went on after a **pause**. It comes when sound is back that allows a new **pause**: 20 ms over the quiet level and clearly above the room's noise, or the speech model hearing speech again after a silent frame. At most one per **pause**, and never without one before it. A client cancels the work it started at the **pause**. A click (one 10 ms block) and steady room noise cause none. Without it, a **pause** stays open until the next **pause** or the **end**.
- **end**: the speech segment is over. Its timing:
  - By default it comes exactly where the speech model alone ends the
    segment: when the model's raw probability per 32 ms frame has stayed
    below the continue threshold for 0.24 s (the count `silero_vad` uses; a
    detector that counts on a smoothed probability ends one or two frames
    later at the same setting). "The speech model alone" means this count
    everywhere below. So by default there are no ends, and no lost speech,
    that the model alone with the same settings would not have.
  - The segment still says when the speech stopped: the start of the quiet
    that led to the end, from the quiet clock, when that is before the
    model's last speech frame.
  - Optionally (early end, off by default) it comes 0.24 s after the audio
    went quiet (0.28 s when room noise has raised what counts as quiet, since
    a word's fading tail then already counts), or at the model's count,
    whichever is first. The model's probability lags about 0.1 s behind the
    audio, so this ends segments in pauses of 0.24 to about 0.34 s that the
    model alone rides through: the same trade as a shorter silence setting.
    The segment says which of the two ended it.
  - Either way it is never later than the speech model alone would end it.
  - Speech that resumes before then cancels it, even very soft speech.
  - With the early end, a short click or noise doesn't end the segment. It
    starts the quiet count again (the end needs 0.24 s of unbroken quiet
    after it) and leaves the speech model's own count alone.
  - "Quiet" is judged against the room's noise level, so **pause** and the
    optional early end work in noisy rooms too. The noise level is learned only from
    audio that isn't speech, and soft speech is never treated as silence.
  - "Quiet" is also judged against the speaker: it is at least 20 dB below
    their own level (the median of the running segment's speech), also when
    that is below the fixed quiet level. A quiet speaker's soft word endings
    therefore don't end the segment, at any input level.
  - Where steady room or line noise is less than that far below the speaker,
    no level can count as quiet. There the audio also counts as quiet when
    what is left of it after subtracting the room's spectrum is below that
    same level. Speech that stands out of the noise in any part of the
    spectrum is not quiet. Noise that is not steady, or less than about 11 dB
    below the speaker, leaves too much: the early **pause** (and the optional
    early end) is lost there and the speech model alone decides. Clean input is not affected.
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
- Whether the quiet clock ended it before the speech model's count (only with the early end).
- Total voiced time and peak speech confidence.

## Interfaces
- Native: feed chunks and get back the events, in order.
- Drop-in: a wrapper with the same constructor and call signature as
  Silero's `VADIterator`, honoring its parameters and returning its
  `{'start': …}` / `{'end': …}` results. Switching to it is a change of
  import. Its end is decided where Silero's count decides it; the sample it
  reports is where the audio went quiet. The early end can be switched on.

## Constraints
- Streaming. No delay beyond the 10 ms quiet block or the model's own frame.
- CPU only, one thread, far faster than realtime. No network and no threads of its own.
- Works on endless streams: if the speech model stops responding (it can lock
  up after long low-level noise or near-silent pauses), it recovers by itself:
  once per pause after about a second of hearing nothing over real audio, or
  within ~0.2 s when it ignores audio as loud as the speaker. Near-silence never
  triggers a recovery, and none happens during the assistant's own playback.
- A recovery never makes it hear speech that isn't there: it returns to how the
  model sounded just after the *user's* speech ended (not during it, and never
  after the assistant's own voice).
- The assistant's own voice (echo) never starts a segment by way of a recovery,
  and silence after the assistant speaks never produces a phantom segment.
- A recovery comes on the same frame with and without the quiet clock, so the
  speech model's probabilities don't depend on the quiet clock.
- Optionally, the audio the speech model sees is amplified for quiet speakers
  (up to 20 dB, never into clipping). Everything else, including the audio
  that is returned, uses the input as it came. Off by default.
- Every threshold and duration above can be configured.
