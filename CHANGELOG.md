# Changelog

## 0.1.0

The product is the early `pause`. `end` is now exactly Silero's end, and
`pause` got more accurate at the cost of about 8 ms of lead. Numbers and
tables: [README](README.md), [docs/results.md](docs/results.md).

### Changed behaviour

- **`end` comes where Silero alone ends the segment.** Up to 0.0.5 the quiet
  clock could end a segment earlier. That is now the option
  `VAD(early_end=True)` (also on `VADIterator`); `Segment.early` says whether
  the quiet clock made the end. If you relied on the earlier `end`, set it.
- **`pause` is repeated.** When sound comes back after a `pause` and the
  audio goes quiet again, a new `pause` follows. The last `pause` is more
  than 100 ms too early in 1 to 9 % of clips instead of 6 to 11 %; its median
  lead over Silero is 92 ms instead of 100 ms on the one-speaker turns.
  `VAD(repause_s=None)` gives one `pause` per stretch of speech, as before.
  Sound in the trailing silence does not repeat a `pause`: it counts only
  while Silero still takes the audio for speech or stopped no longer ago than
  its lag (`repause_lag_s`, 0.128 s).

### New

- **`pause_mode`:** `"accurate"` (default), `"balanced"` or `"fast"`. `"fast"`
  leads Silero by 118 to 130 ms and puts about one in five last pauses inside
  the last word.
- **`resume` event** (`VAD(resume_events=True)`): the speech went on after a
  `pause`; cancel what you started.
- **Steady noise close under the speaker** is subtracted before the quiet
  check (`noise_subtract_db`). A noisy quiet line and a quiet phone line had
  no lead in 0.0.5 and now lead by 84 to 114 ms (simulated noise and line).
- **LiveKit plugin:** `SpeculativeSTT` starts LiveKit's preemptive generation
  at `pause`.

### Fixed

- A DC offset in the input no longer hides the silence from the quiet clock
  (levels are measured behind a 20 Hz high-pass).
- The "Silero ignores speech-loud audio" recovery could restore a state from
  inside the previous segment and report a phantom segment.
- With the early end, the recovery could run on other frames than without it,
  and one clip of 7,080 ended later than Silero alone.
- With the early end, a segment that opened right after an early end could
  end up to three frames before Silero's own count.

### Removed

- No Pipecat integration is planned: its speech recognizers finalize only
  when the VAD reports the stop, so nothing can start at `pause`
  ([issue 4](https://github.com/KoljaB/RealtimeVAD/issues/4)).
