"""Speech start/end detection. Silero decides speech; the end races Silero's
silence count against a 10 ms near-silence clock (see SPEC.md)."""

from __future__ import annotations

import math
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
class Event:
    kind: str  # "start" | "pause" | "end"
    time: float
    segment: Optional[Segment] = None  # set on "end"


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
        end_s: float = 0.24,
        quiet_rms: float = 0.002,
        preroll_s: float = 0.30,
        lookback_s: float = 1.0,
        max_speech_s: float = 120.0,
        extend: Optional[Callable[[float], Optional[float]]] = None,
        probability: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        if probability is None:
            from .silero import Silero

            probability = Silero()
        self._p = probability
        self._extend = extend
        self.start_threshold, self.continue_threshold = start_threshold, continue_threshold
        self.end_s, self.quiet_rms = end_s, quiet_rms
        self._start_frames = max(1, _frames(start_s))
        self._end_frames = max(1, _frames(end_s))
        self._max_frames = max(1, _frames(max_speech_s))
        self._preroll_n = _frames(preroll_s)
        self._lookback_n = max(0, _frames(lookback_s))
        self._rearm_samples = max(1, math.ceil(start_s * SAMPLE_RATE))
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
        self._floor: Optional[float] = None
        self._in_speech = False
        self._idle = 0
        self._streak = 0
        self._clear_segment()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def audio(self) -> Optional[np.ndarray]:
        """The running segment's audio up to the latest sample fed, or None outside speech."""
        if not self._in_speech:
            return None
        return np.concatenate(self._audio + [self._frame_buf])

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
        quiet = math.sqrt(float(np.mean(block * block))) <= self.quiet_rms
        if self._t_quiet is not None:
            if quiet:
                self._loud = 0
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
            self._t_quiet, self._unbroken, self._loud = t, True, 0
            self._pause(t, events)

    def _target(self, t: float) -> float:
        """Quiet-clock wait: end_s, or longer if ``extend`` asks once end_s is reached."""
        if t - self._t_quiet + EPS < self.end_s or self._extend is None:
            return self.end_s
        extra = self._extend(t)
        return self.end_s if extra is None else max(self.end_s, extra)

    # -- per 32 ms Silero frame --------------------------------------------
    def _frame(self, frame: np.ndarray, t: float, events: list[Event]) -> None:
        p = self._p(frame)
        voiced = p >= (self.continue_threshold if self._in_speech or self._streak else self.start_threshold)
        level = _dbfs(frame)
        self._history.append((frame, level, t <= self._echo_until))
        if not self._in_speech:
            self._idle_frame(p, voiced, level, t, events)
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
                self._pause(t, events)
        if len(self._audio) >= self._max_frames:
            self._finish(t, events)
            return
        if self._silent >= self._end_frames and self._extend is not None:
            extra = self._extend(t)
            self._end_target = self._end_frames if extra is None else max(self._end_frames, _frames(extra))
        if self._silent >= self._end_target:
            self._finish(t, events)

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
        if self._streak < self._start_frames:
            return
        lookback = self._lookback(floor)
        keep = self._preroll_n if not lookback else max(self._preroll_n, min(self._preroll_n + lookback, self._idle))
        self._in_speech = True
        self._audio = [f for f, _, _ in list(self._history)[-keep:]]
        self._start = t - (self._streak - 1 + lookback) * FRAME_S  # end of first streak frame, as in Miep
        self._voiced = self._streak
        self._last_voiced = t
        events.append(Event("start", self._start))

    def _lookback(self, floor: Optional[float]) -> int:
        """Frames before the start streak that still belong to a soft opening."""
        h, s = self._history, self._streak
        if not self._lookback_n or floor is None or len(h) < s or any(h[-i][2] for i in range(1, s + 1)):
            return 0
        threshold = max(floor + 12.0, -60.0)
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
    def _pause(self, t: float, events: list[Event]) -> None:
        if not self._paused:
            self._paused = True
            events.append(Event("pause", t))

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
