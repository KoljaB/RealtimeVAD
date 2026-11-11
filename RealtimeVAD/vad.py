"""Speech start/end detection. Silero decides speech; the end races Silero's
silence count against a 10 ms near-silence clock (see SPEC.md)."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .silero import FRAME, SAMPLE_RATE

BLOCK = 160  # 10 ms quiet-clock block
FRAME_S = FRAME / SAMPLE_RATE
EPS = 1e-9


@dataclass(frozen=True)
class Segment:
    audio: np.ndarray  # float32, 16 kHz, from just before the start up to the end
    start: float
    end: float  # stream time the end was decided
    speech_end: float  # when the speech actually stopped
    voiced_s: float
    peak: float


@dataclass(frozen=True)
class Frame:
    """One evaluated 32 ms model frame (only with ``frame_events=True``)."""

    probability: float
    voiced: bool
    in_speech: bool
    silent_frames: int  # consecutive unvoiced frames inside speech
    streak: int  # voiced frames toward a start
    voiced_s: float  # of the running segment
    peak: float  # of the running segment
    infer_s: float  # model inference time


@dataclass(frozen=True)
class Event:
    kind: str  # "onset" | "start" | "pause" | "end" | "frame"
    time: float
    segment: Optional[Segment] = None  # set on "end"
    source: str = ""  # "pause": "quiet" (10 ms clock) or "silero" (first unvoiced frame)
    frame: Optional[Frame] = None  # set on "frame"


def _frames(seconds: float) -> int:
    return math.ceil(seconds / FRAME_S - EPS)


def _dbfs(x: np.ndarray) -> float:
    power = float(np.mean(x * x))
    return max(-100.0, 10.0 * math.log10(power)) if power > 0.0 else -100.0


def _pcm(audio) -> np.ndarray:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    a = np.asarray(audio)
    if a.dtype == np.int16:
        return a.astype(np.float32).reshape(-1) / 32768.0
    return a.astype(np.float32, copy=False).reshape(-1)


class VAD:
    """Feed 16 kHz mono PCM; ``feed`` returns start/pause/end events in order."""

    def __init__(
        self,
        *,
        start_threshold: float = 0.90,
        continue_threshold: float = 0.45,
        start_s: float = 0.064,
        end_s: float = 0.24,  # Silero's silence count
        quiet_end_s: Optional[float] = None,  # quiet clock; default end_s
        noisy_extra_s: float = 0.04,  # added when room noise has raised the quiet threshold
        quiet_rms: float = 0.002,
        noise_margin_db: float = 15.0,
        speech_margin_db: float = 15.0,
        preroll_s: float = 0.30,
        lookback_s: float = 1.0,
        restore_s: float = 1.0,  # Silero recovery after this much p < 0.1 over real (non-zero) audio; 0 = off
        deaf_s: float = 0.19,  # ...or after Silero ignored speech-loud audio this long
        max_speech_s: float = 120.0,
        extend: Optional[Callable[[float], Optional[float]]] = None,
        frame_events: bool = False,
        onset_events: bool = False,  # "onset" at the first confident frame, before "start" confirms it
        confirm_threshold: Optional[float] = None,  # each frame of the start debounce; default continue_threshold
        rearm_s: Optional[float] = None,  # sound that re-arms the quiet clock; default start_s
        probability: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        if probability is None:
            from .silero import Silero

            probability = Silero(restore_s=0)  # restores are decided here, from audio evidence
        self._p = probability
        self._extend = extend
        self._frame_events = frame_events
        self._onset_events = onset_events
        self.start_threshold, self.continue_threshold = start_threshold, continue_threshold
        # After the first confident frame, every frame until start_s is reached must stay at
        # or above this (a click fades out above continue_threshold for a few frames).
        self.confirm_threshold = continue_threshold if confirm_threshold is None else confirm_threshold
        # The quiet clock can only make an end earlier than Silero's count, never later. Over
        # near-silence it waits end_s like Silero; when room noise has raised the threshold, a
        # word's fading tail already counts as quiet, so it waits noisy_extra_s longer.
        self.end_s = end_s
        self.quiet_end_s = end_s if quiet_end_s is None else quiet_end_s
        self.noisy_extra_s = noisy_extra_s
        self.quiet_rms = quiet_rms
        self.noise_margin_db, self.speech_margin_db = noise_margin_db, speech_margin_db
        self.start_s = start_s
        self._end_frames = max(1, _frames(end_s))
        self._max_frames = max(1, _frames(max_speech_s))
        self._preroll_n = _frames(preroll_s)
        self._lookback_n = max(0, _frames(lookback_s))
        self._rearm_samples = max(1, math.ceil((start_s if rearm_s is None else rearm_s) * SAMPLE_RATE))
        self._restore_n = _frames(restore_s) if restore_s > 0 else 0
        self._deaf_n = _frames(deaf_s) if deaf_s > 0 and restore_s > 0 else 0
        self.reset()

    def reset(self) -> None:
        reset = getattr(self._p, "reset", None)
        if reset is not None:
            reset()
        self._n = 0  # samples processed
        self._block_buf = np.zeros(0, dtype=np.float32)
        self._frame_buf = np.zeros(0, dtype=np.float32)
        self._echo_until = -math.inf
        self._history: deque = deque(maxlen=self._preroll_n + self._lookback_n)  # (frame, dBFS, echo)
        self._floor: Optional[float] = None  # idle floor for the onset look-back
        self._noise_db: Optional[float] = None  # room noise for the quiet threshold
        self._speech_db: Optional[float] = None
        self._quiet = self.quiet_rms if self.quiet_rms > 0 else -1.0  # current quiet threshold (RMS)
        self._tail = False  # waiting for Silero to drop after an early end
        self._low = self._deaf = 0  # frames of p < 0.1 over real audio / over speech-loud audio
        self._quiet_restored = False  # the quiet rule already restored in this pause
        self._in_speech = False
        self._idle = 0
        self._streak = 0
        self._lookback_s = 0.0
        self._clear_segment()

    @property
    def start_s(self) -> float:
        """Speech needed to confirm a start (the debounce after "onset"); settable while running."""
        return self._start_s

    @start_s.setter
    def start_s(self, seconds: float) -> None:
        self._start_s = seconds
        self._start_frames = max(1, _frames(seconds))

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def onset_lookback_s(self) -> float:
        """How far the latest start was moved back over a soft opening."""
        return self._lookback_s

    @property
    def quiet_clock(self) -> tuple[bool, bool, int]:
        """(armed, unbroken, quiet blocks since it armed) of the 10 ms quiet clock."""
        return self._t_quiet is not None, self._unbroken, self._quiet_blocks

    def audio(self) -> Optional[np.ndarray]:
        """The running segment's audio up to the latest sample fed, or None outside speech."""
        if not self._in_speech:
            return None
        return np.concatenate(self._audio + [self._frame_buf])

    def segment(self) -> Optional[Segment]:
        """The running segment so far (end = now), or None outside speech."""
        if not self._in_speech:
            return None
        return Segment(self.audio(), self._start, self._n / SAMPLE_RATE, self._last_voiced,
                       self._voiced * FRAME_S, self._peak)

    def feed(self, audio, playback: bool = False) -> list[Event]:
        """``playback=True`` marks the chunk as possibly containing your own output (echo)."""
        events: list[Event] = []
        x = _pcm(audio)
        i = 0
        while i < len(x):
            # advance to the next block or frame boundary, whichever is first
            step = min(BLOCK - len(self._block_buf), FRAME - len(self._frame_buf), len(x) - i)
            part = x[i : i + step]
            i += step
            self._n += step
            self._block_buf = np.concatenate((self._block_buf, part))
            self._frame_buf = np.concatenate((self._frame_buf, part))
            if playback:
                self._echo_until = self._n / SAMPLE_RATE + 0.5
            block = self._block_buf if len(self._block_buf) == BLOCK else None
            quiet = block is not None and self._block_start(block)
            if len(self._frame_buf) == FRAME:
                frame, self._frame_buf = self._frame_buf, np.zeros(0, dtype=np.float32)
                self._frame(frame, self._n / SAMPLE_RATE, events)
            if block is not None:
                self._block_buf = np.zeros(0, dtype=np.float32)
                self._block_end(quiet, self._n / SAMPLE_RATE, events)
        return events

    # -- per 10 ms block (quiet clock) -------------------------------------
    def _block_start(self, block: np.ndarray) -> bool:
        quiet = math.sqrt(float(np.mean(block * block))) <= self._quiet
        if self._t_quiet is not None:
            if quiet:
                self._loud = 0
                self._quiet_blocks += self._unbroken
            else:
                self._unbroken = False
                self._loud += BLOCK
                if self._loud >= self._rearm_samples:
                    self._t_quiet, self._loud, self._paused = None, 0, False
        return quiet

    def _block_end(self, quiet: bool, t: float, events: list[Event]) -> None:
        if not self._in_speech:
            return
        if self._t_quiet is not None and self._unbroken and t - self._t_quiet + EPS >= self._target(t):
            self._finish(t, events, speech_end=self._t_quiet)
        elif quiet and self._t_quiet is None:
            self._t_quiet, self._unbroken, self._loud, self._quiet_blocks = t, True, 0, 1
            raised = self._quiet > self.quiet_rms * (1 + 1e-6)
            self._quiet_wait = self.quiet_end_s + (self.noisy_extra_s if raised else 0.0)
            self._pause(t, events, "quiet")

    def _target(self, t: float) -> float:
        """Quiet-clock wait (set when it armed), or longer if ``extend`` asks once that is reached."""
        wait = self._quiet_wait
        if t - self._t_quiet + EPS < wait or self._extend is None:
            return wait
        extra = self._extend(t)
        return wait if extra is None else max(wait, extra)

    # -- per 32 ms Silero frame --------------------------------------------
    def _frame(self, frame: np.ndarray, t: float, events: list[Event]) -> None:
        t0 = time.perf_counter()
        p = self._p(frame)
        infer_s = time.perf_counter() - t0
        if self._in_speech:
            voiced = p >= self.continue_threshold
        else:  # idle: the first frame needs start_threshold, the rest of the debounce confirm_threshold
            voiced = p >= (self.confirm_threshold if self._streak else self.start_threshold)
        level = _dbfs(frame)
        echo = t <= self._echo_until
        self._history.append((frame, level, echo))
        self._levels(p, level, echo)
        if self._tail:  # after an early end, Silero's fading tail must not open a new segment...
            self._tail = p >= self.continue_threshold
            if level <= 20.0 * math.log10(max(self._quiet, 1e-9)):
                voiced = False  # ...over quiet audio; speech that resumes at once is loud and may start
        if not self._in_speech:
            self._idle_frame(p, voiced, level, t, events)
            self._report(t, p, voiced, infer_s, events)
            return
        self._audio.append(frame)
        self._peak = max(self._peak, p)
        if voiced:
            if self._silent:
                self._paused = False
                if self._t_quiet is not None and t > self._t_quiet:
                    self._unbroken = False
            self._voiced += 1
            self._silent = 0
            self._last_voiced = t
            self._end_target = self._end_frames
        else:
            self._silent += 1
            if self._silent == 1:
                self._pause(t, events, "silero")
        self._report(t, p, voiced, infer_s, events)
        if len(self._audio) >= self._max_frames:
            self._finish(t, events)
            return
        if self._silent >= self._end_frames and self._extend is not None:
            extra = self._extend(t)
            self._end_target = self._end_frames if extra is None else max(self._end_frames, _frames(extra))
        if self._silent >= self._end_target:
            self._finish(t, events)

    def _levels(self, p: float, level: float, echo: bool) -> None:
        """Learn speech and room-noise levels and set the quiet threshold."""
        if p >= self.start_threshold:
            s = self._speech_db
            self._speech_db = level if s is None else s + max(-1.0, min(1.0, level - s))
        elif p < self.continue_threshold and level > -90.0 and not echo:  # skip gated digital silence
            n = self._noise_db
            if n is None:
                self._noise_db = level
            elif self._in_speech:  # pauses inside speech: running median, about 1 dB per frame
                self._noise_db = n + max(-1.0, min(1.0, level - n))
            else:  # idle: falls fast, rises 3 dB/s, so a soft opening barely lifts it
                self._noise_db = level if level <= n else n + min(level - n, 3.0 * FRAME_S)
        if self.quiet_rms <= 0:
            self._quiet = -1.0  # quiet clock off: no block is ever quiet
        else:
            # Noise can raise the threshold above quiet_rms, but only up to speech level - 15 dB
            # (soft speech is never quiet); it never drops below quiet_rms, so it is never stricter.
            floor_db = 20.0 * math.log10(self.quiet_rms)
            db = floor_db
            if self._noise_db is not None:
                db = max(db, self._noise_db + self.noise_margin_db)
            if self._speech_db is not None:
                db = max(floor_db, min(db, self._speech_db - self.speech_margin_db))
            self._quiet = 10.0 ** (db / 20.0)
        # Silero's recurrent state drifts in long noisy or near-silent stretches until it is
        # "deaf" (p ~ 0 through loud speech). Restore the last confident-speech state after a
        # second of p < 0.1 over real audio (gated digital silence doesn't drift it, and a
        # restore there only disturbs the next onset), never on a frame that sounds like an
        # onset; or at once when Silero has ignored speech-loud audio for deaf_s.
        #
        # The restore point is the state after the *user's* confident speech, never after
        # the assistant's own voice: a state that last heard playback echo scores following
        # room noise as speech (p ~0.96 decaying over 0.4 s), which opens phantom segments.
        restore = getattr(self._p, "restore", None)
        snapshot = getattr(self._p, "snapshot", None)
        if p >= self.start_threshold and not echo and snapshot is not None:
            snapshot()
        if self._in_speech:
            self._quiet_restored = False  # a new pause may restore once more
        if not self._restore_n or restore is None or self._in_speech or echo:
            self._low = self._deaf = 0
            return
        speech_like = self._speech_like(level)
        # a healthy Silero passes continue_threshold within 1-3 frames of loud speech
        self._deaf = 0 if p >= self.continue_threshold else self._deaf + speech_like
        self._low = self._low + 1 if p < 0.1 and level > -70.0 else 0  # near-silence (gates, noise suppression) never drifts it
        deaf = self._deaf_n and self._deaf >= self._deaf_n
        quiet = self._low >= self._restore_n and not speech_like and not self._quiet_restored
        if deaf or quiet:  # the quiet rule fires once per pause: one restore un-sticks Silero
            restore()
            self._quiet_restored = self._quiet_restored or bool(quiet)
            self._low = self._deaf = 0

    def _speech_like(self, level: float) -> bool:
        """Loud like the speaker and clearly above the idle floor."""
        if self._speech_db is None or self._floor is None:
            return False
        margin = max(3.0, min(10.0, 0.5 * (self._speech_db - self._floor)))
        return level >= max(self._floor + margin, self._speech_db - 10.0)

    def _idle_frame(self, p: float, voiced: bool, level: float, t: float, events: list[Event]) -> None:
        floor = self._floor
        if self._lookback_n:
            if floor is None or level <= floor:
                self._floor = level if floor is None else floor + (level - floor) * 0.5
            else:
                self._floor = floor + min(level - floor, 3.0 * FRAME_S)
        self._idle += 1
        self._streak = self._streak + 1 if voiced else 0
        self._peak = max(self._peak, p) if voiced else 0.0
        if self._streak == 1 and self._onset_events:
            events.append(Event("onset", t))  # maybe speech; "start" follows only if it holds
        if self._streak < self._start_frames:
            return
        lookback = self._lookback(floor)
        keep = self._preroll_n if not lookback else max(self._preroll_n, min(self._preroll_n + lookback, self._idle))
        self._in_speech = True
        self._audio = [f for f, _, _ in list(self._history)[-keep:]]
        self._start = t - (self._streak - 1 + lookback) * FRAME_S  # end of first streak frame, as in Miep
        self._lookback_s = lookback * FRAME_S
        self._voiced = self._streak
        self._last_voiced = t
        events.append(Event("start", self._start))

    def _lookback(self, floor: Optional[float]) -> int:
        """Frames before the start streak that still belong to a soft opening."""
        h, s = self._history, self._streak
        if not self._lookback_n or floor is None or len(h) < s or any(h[-i][2] for i in range(1, s + 1)):
            return 0
        margin = 12.0  # an opening must be clearly louder than the room...
        if self._speech_db is not None:  # ...but in loud noise, half the speech-to-noise gap is enough
            margin = max(3.0, min(margin, 0.5 * (self._speech_db - floor)))
        threshold = max(floor + margin, -60.0)
        count = gap = 0
        for step in range(1, min(self._lookback_n, self._idle - s, len(h) - s) + 1):
            _, level, echo = h[-(s + step)]
            if echo:
                break
            if level >= threshold:
                count, gap = step, 0
            else:
                gap += 1
                if gap > 3:  # 0.1 s
                    break
        return count if count > self._preroll_n - s else 0

    # -- segment bookkeeping -----------------------------------------------
    def _report(self, t: float, p: float, voiced: bool, infer_s: float, events: list[Event]) -> None:
        if self._frame_events:
            events.append(Event("frame", t, frame=Frame(
                p, voiced, self._in_speech, self._silent, self._streak,
                self._voiced * FRAME_S, self._peak, infer_s)))

    def _pause(self, t: float, events: list[Event], source: str) -> None:
        if not self._paused:
            self._paused = True
            events.append(Event("pause", t, source=source))

    def _finish(self, t: float, events: list[Event], speech_end: Optional[float] = None) -> None:
        segment = Segment(
            audio=self.audio(),
            start=self._start,
            end=t,
            speech_end=self._last_voiced if speech_end is None else speech_end,
            voiced_s=self._voiced * FRAME_S,
            peak=self._peak,
        )
        self._in_speech = False
        self._tail = speech_end is not None  # ended by the quiet clock, Silero may still be high
        self._idle = self._streak = 0
        self._clear_segment()
        events.append(Event("end", t, segment))

    def _clear_segment(self) -> None:
        self._audio: list[np.ndarray] = []
        self._start = self._last_voiced = self._peak = 0.0
        self._voiced = self._silent = 0
        self._end_target = self._end_frames
        self._paused = False
        self._t_quiet: Optional[float] = None
        self._unbroken = False
        self._loud = 0
        self._quiet_blocks = 0
        self._quiet_wait = self.quiet_end_s
