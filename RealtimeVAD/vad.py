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
LEVEL_S = 3.0  # the speaker's level is the median of the segment's last 3 s of speech frames...
LEVEL_MIN_S = 0.25  # ...once it has this much; until then the previous segment's level counts
FIRST_WORD_FRAMES = 2  # speech this long (64 ms, the default start_s) before a start belongs to it
SOFT_OVER_DB = 10.0  # a soft start must be this far above the idle floor (steady noise Silero is unsure about is not)
# Levels (the quiet clock's blocks, the speaker's and the room's level) are measured behind a
# one-pole 20 Hz high-pass, so a DC offset in the input is not mistaken for sound.
_DC_R = math.exp(-2.0 * math.pi * 20.0 / SAMPLE_RATE)
_DC_GAIN = 0.5 * (1.0 + _DC_R)  # unity gain at the top of the band
_DC_POW = _DC_R ** np.arange(FRAME, dtype=np.float64)
_DC_INV = 1.0 / _DC_POW
# Steady noise close under the speaker (a noisy phone line, a raw microphone in a room) never
# falls below the quiet threshold. There a 10 ms block also counts as quiet when what is left
# of its spectrum after subtracting the room's is below the quiet floor: 20 ms Hann windows,
# 16 mel-spaced bands from 100 Hz to 8 kHz.
_NS_WIN = 2 * BLOCK
_NS_HZ = 700.0 * (10.0 ** (np.linspace(np.log10(1 + 100 / 700), np.log10(1 + 8000 / 700), 17)) - 1.0)


def _ns_bands(n: int) -> tuple:
    """(Hann window, first bin of each band, scale that makes band powers add up to the mean square)."""
    hann = np.hanning(n + 1)[:-1]
    return hann, np.searchsorted(np.fft.rfftfreq(n, 1 / SAMPLE_RATE), _NS_HZ[:-1]), 2.0 / (n * float(np.sum(hann * hann)))


_NS_BLOCK, _NS_FRAME = _ns_bands(_NS_WIN), _ns_bands(FRAME)  # a block is judged on 20 ms, the room is learned per frame
NS_RISE_DB = 3.0 * FRAME_S  # the room's spectrum falls at once and rises 3 dB/s, like the idle floor: soft speech
# Silero is unsure about barely lifts it, and a word's tail is forgotten in the next frame
NS_NEAR_DB = 3.0  # used only where the room is at most this far below the quiet floor (the level alone is blind there)
NS_MIN_SNR_DB = 6.0  # ...and at least this far below the speaker


@dataclass(frozen=True)
class Segment:
    audio: np.ndarray  # float32, 16 kHz, from just before the start up to the end
    start: float
    end: float  # stream time the end was decided
    speech_end: float  # when the speech actually stopped (from the quiet clock where it saw it, else Silero's last voiced frame)
    voiced_s: float
    peak: float
    early: bool = False  # ended by the quiet clock before Silero's count (only with early_end=True)


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
    kind: str  # "onset" | "start" | "pause" | "resume" | "end" | "frame"
    time: float
    segment: Optional[Segment] = None  # set on "end"
    source: str = ""  # "pause": "quiet" (10 ms clock) or "silero" (first unvoiced frame)
    frame: Optional[Frame] = None  # set on "frame"


class _ByMode:
    def __repr__(self) -> str:
        return "<set by pause_mode>"


_MODE = _ByMode()
# How early "pause" comes against how often it is too early (inside the last word). One choice
# sets the parameters below; each of them still wins when given. Measured in README.md.
#   accurate  a "pause" is repeated when sound came back, so the last one is rarely early
#   balanced  one "pause" per stretch of speech, and audio counts as quiet 3 dB sooner
#   fast      the same, 6 dB sooner
PAUSE_MODES = {
    "accurate": dict(repause_s=0.02, quiet_rms=0.002, floor_margin_db=20.0, speech_margin_db=15.0, noise_margin_db=15.0),
    "balanced": dict(repause_s=None, quiet_rms=0.0028, floor_margin_db=17.0, speech_margin_db=12.0, noise_margin_db=18.0),
    "fast": dict(repause_s=None, quiet_rms=0.004, floor_margin_db=14.0, speech_margin_db=9.0, noise_margin_db=21.0),
}


def _frames(seconds: float) -> int:
    return math.ceil(seconds / FRAME_S - EPS)


def _dbfs(power: float) -> float:
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
        early_end: bool = False,  # the quiet clock may end a segment before Silero's count does
        quiet_end_s: Optional[float] = None,  # ...after this much quiet; default end_s
        noisy_extra_s: float = 0.04,  # ...plus this when room noise has raised the quiet threshold
        pause_mode: str = "accurate",  # or "balanced", "fast": an earlier "pause" that is more often too early (PAUSE_MODES)
        quiet_rms: float = _MODE,  # accurate: 0.002
        noise_margin_db: float = _MODE,  # accurate: 15
        speech_margin_db: float = _MODE,  # accurate: 15
        floor_margin_db: Optional[float] = _MODE,  # accurate: 20. Quiet input: quiet_rms drops to this far below the speaker; None = fixed
        noise_subtract_db: Optional[float] = 4.0,  # steady noise near the speaker: subtract the room's spectrum, this far over-estimated; None = off
        preroll_s: float = 0.30,
        lookback_s: float = 1.0,
        restore_s: float = 1.0,  # Silero recovery after this much p < 0.1 over real (non-zero) audio; 0 = off
        deaf_s: float = 0.19,  # ...or after Silero ignored speech-loud audio this long
        max_speech_s: float = 120.0,
        extend: Optional[Callable[[float], Optional[float]]] = None,
        frame_events: bool = False,
        onset_events: bool = False,  # "onset" at the first confident frame, before "start" confirms it
        resume_events: bool = False,  # "resume" when the speech went on after a "pause"
        confirm_threshold: Optional[float] = None,  # each frame of the start debounce; default continue_threshold
        rearm_s: Optional[float] = None,  # sound that re-arms the quiet clock; default start_s
        repause_s: Optional[float] = _MODE,  # accurate: 0.02. Sound after a "pause" that lets the next quiet block report a new one; None = off
        repause_room_db: float = 6.0,  # ...counted only this far above the learned room noise
        repause_lag_s: Optional[float] = 0.128,  # ...and only while Silero's own silence is no longer than this; None = always
        soft_start_s: Optional[float] = 0.128,  # a start also after this long at soft_start_threshold; None = off
        soft_start_threshold: float = 0.5,
        normalize_to_dbfs: Optional[float] = None,  # quiet input: amplify what Silero sees up to this speech level (try -30)
        normalize_max_db: float = 20.0,
        probability: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        if probability is None:
            from .silero import Silero

            probability = Silero(restore_s=0)  # restores are decided here, from audio evidence
        # pause_mode chooses how early "pause" comes: it only sets the five parameters in
        # PAUSE_MODES, and one given here wins. It changes "pause" (and "resume", and the
        # speech_end a segment reports), never "start" or the default "end".
        if pause_mode not in PAUSE_MODES:
            raise ValueError(f"pause_mode must be one of {', '.join(PAUSE_MODES)}, not {pause_mode!r}")
        mode = PAUSE_MODES[pause_mode]
        self.pause_mode = pause_mode
        quiet_rms = mode["quiet_rms"] if quiet_rms is _MODE else quiet_rms
        noise_margin_db = mode["noise_margin_db"] if noise_margin_db is _MODE else noise_margin_db
        speech_margin_db = mode["speech_margin_db"] if speech_margin_db is _MODE else speech_margin_db
        floor_margin_db = mode["floor_margin_db"] if floor_margin_db is _MODE else floor_margin_db
        repause_s = mode["repause_s"] if repause_s is _MODE else repause_s
        self._p = probability
        self._extend = extend
        self._frame_events = frame_events
        self._onset_events = onset_events
        # A "pause" is a guess. "resume" takes it back: sound came back after it (what allows a
        # new "pause": repause_s of sound clearly above the room, sound that re-arms the quiet
        # clock, or Silero hearing speech again after a silent frame). At most one per "pause";
        # a consumer cancels the work it started at the "pause" and starts again at the next.
        self._resume_events = resume_events
        self.start_threshold, self.continue_threshold = start_threshold, continue_threshold
        # After the first confident frame, every frame until start_s is reached must stay at
        # or above this (a click fades out above continue_threshold for a few frames).
        self.confirm_threshold = continue_threshold if confirm_threshold is None else confirm_threshold
        # Short hums ("mm-hmm"), quiet and narrowband speech often stay below start_threshold for
        # their whole length. They also start a segment once the probability has held
        # soft_start_threshold for soft_start_s over audio clearly louder than the room: slower,
        # and never weaker than the debounce above (at least start_s long, every frame at or
        # above confirm_threshold).
        self.soft_start_s, self.soft_start_threshold = soft_start_s, soft_start_threshold
        # Silero is less sure about quiet speech, most of all on a phone line: it finds fewer
        # turns and drops out inside them. With normalize_to_dbfs the frames it sees (only those;
        # levels and the quiet clock use the audio as it came) are amplified until speech that
        # it scores 0.5 or more sits at that level, by at most normalize_max_db. Off by default:
        # it also changes what Silero says about speakers a little below that level.
        self.normalize_to_dbfs, self.normalize_max_db = normalize_to_dbfs, normalize_max_db
        # A segment ends when Silero has heard end_s of silence. The quiet clock reports the
        # speech-off ("pause") about 100 ms before Silero notices it, and tells the segment when
        # the speech stopped; by default it does not end the segment. Silero's probability lags
        # that long behind the audio, so an end after end_s of quiet audio comes in pauses of
        # end_s to end_s + 0.1 s that Silero alone rides through: the same as a shorter end_s,
        # and it costs the same (more ends inside an utterance). early_end=True allows it: over
        # near-silence the quiet clock then waits quiet_end_s; when room noise has raised the
        # threshold, a word's fading tail already counts as quiet, so it waits noisy_extra_s
        # longer. Either way an end is never later than Silero's count alone gives it.
        self.end_s = end_s
        self.early_end = early_end
        self.quiet_end_s = end_s if quiet_end_s is None else quiet_end_s
        self.noisy_extra_s = noisy_extra_s
        self.quiet_rms = quiet_rms
        self.noise_margin_db, self.speech_margin_db = noise_margin_db, speech_margin_db
        # quiet_rms suits a speaker at a normal level, whose soft word endings stay above it.
        # Below a quieter speaker (a phone line, a distant microphone) the floor keeps this
        # distance to their level instead, or those endings would count as silence and end the
        # segment. Measured on real turns: 19 dB still cuts some, 20.5 and 22 dB are equally safe.
        self.floor_margin_db = floor_margin_db
        # Where steady noise is less than about 20 dB below the speaker, the level never drops
        # under the quiet floor and the quiet clock would never run. A block is then also quiet
        # when its spectrum, minus the room's spectrum over-estimated by noise_subtract_db, has
        # less left than the quiet floor. Speech 20 dB below the speaker that stands out of the
        # noise in any band keeps its energy; what the noise fully covers Silero does not hear
        # either. The room's spectrum is tracked from below, so 4 dB is about 2 dB over its mean.
        self.noise_subtract_db = noise_subtract_db
        self.start_s = start_s
        self._end_frames = max(1, _frames(end_s))
        self._max_frames = max(1, _frames(max_speech_s))
        self._preroll_n = _frames(preroll_s)
        self._lookback_n = max(0, _frames(lookback_s))
        self._rearm_samples = max(1, math.ceil((start_s if rearm_s is None else rearm_s) * SAMPLE_RATE))
        # A "pause" often comes inside the last word: the level dips under the threshold before a
        # release, a fricative or a weak last syllable, comes back for 20 to 60 ms and only then
        # stops. That is too short to re-arm the quiet clock, so the count restarted silently and
        # the only "pause" was up to several hundred ms early. Now sound of repause_s that is over
        # the quiet threshold and repause_room_db above the room (not the room's own flutter)
        # lets the restarted count report a new "pause": the latest one is the best estimate.
        self.repause_s, self.repause_room_db = repause_s, repause_room_db
        # Sound counts for this only while Silero still takes the audio for speech, or has just
        # stopped: its probability lags the audio by about 0.1 s, which is what lets a dip
        # inside the last word be reported at all. Once its silence has run longer than that,
        # sound it does not score as speech (a breath, a click, a door in the trailing silence)
        # takes nothing back: the standing "pause" remains the last one and no "resume" comes.
        # A voiced frame starts Silero's count over, so real speech reopens all of it.
        self.repause_lag_s = repause_lag_s
        self._repause_lag_n = None if repause_lag_s is None else _frames(repause_lag_s)
        self._repause_n = 0 if repause_s is None else max(1, math.ceil(repause_s * SAMPLE_RATE / BLOCK - 1e-9))
        self._restore_n = _frames(restore_s) if restore_s > 0 else 0
        self._deaf_n = _frames(deaf_s) if deaf_s > 0 and restore_s > 0 else 0
        self.reset()

    def reset(self) -> None:
        reset = getattr(self._p, "reset", None)
        if reset is not None:
            reset()
        self._n = 0  # samples processed
        self._block_n = 0  # samples of the running 10 ms block...
        self._block_energy = self._frame_energy = 0.0  # ...and its and the frame's energy behind the high-pass
        self._dc_x: Optional[float] = None  # high-pass state: last input sample...
        self._dc_y = 0.0  # ...and last output sample
        self._frame_buf = np.zeros(0, dtype=np.float32)
        self._ns_parts: deque = deque()  # the latest 20 ms or a little more, as fed
        self._ns_len = 0
        self._ns_noise_db: Optional[np.ndarray] = None  # the room's band powers, dB
        self._ns_noise = np.zeros(0)  # ...as over-estimated powers...
        self._ns_total_db = -100.0  # ...and their plain sum
        self._ns_quiet = False  # the latest block was quiet after subtraction
        self._floor_db = -100.0  # quiet floor, dBFS
        self._echo_until = -math.inf
        self._history: deque = deque(maxlen=self._preroll_n + self._lookback_n)  # (frame, dBFS, echo)
        self._floor: Optional[float] = None  # idle floor for the onset look-back
        self._noise_db: Optional[float] = None  # room noise for the quiet threshold
        self._speech_db: Optional[float] = None  # level of the latest confident frames (follows a fade)
        self._levels_seen: deque = deque(maxlen=_frames(LEVEL_S))  # speech frame levels of this segment
        self._level_db: Optional[float] = None  # their median: the speaker's level
        self._soft = 0  # frames at soft_start_threshold toward a soft start
        self._gain_db = 0.0  # normalize_to_dbfs: gain on Silero's input, moving 2 dB per frame...
        self._gain_seen: deque = deque(maxlen=16)  # ...toward the median level of the last half second of speech
        self._shadow: Optional[int] = None  # Silero's silent frames after an early end, until it would end too
        self._shadow_soft = False  # ...and whether that segment began softly
        self._shadowed = False
        self._soft_segment = False  # the running segment began that way
        self._quiet = self.quiet_rms if self.quiet_rms > 0 else -1.0  # current quiet threshold (RMS)
        self._tail = False  # waiting for Silero to drop after an early end
        self._low = self._deaf = 0  # frames of p < 0.1 over real audio / over speech-loud audio
        self._quiet_restored = False  # the quiet rule already restored in this pause
        self._settling = False  # Silero heard the user's confident speech and has not let go of it yet
        self._in_speech = False
        self._idle = 0
        self._streak = 0
        self._tried: Optional[tuple] = None  # (first, last idle frame) of speech that did not last start_s
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
    def _ns_on(self) -> bool:
        return self.noise_subtract_db is not None and self.quiet_rms > 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def onset_lookback_s(self) -> float:
        """How far the latest start was moved back, over a soft opening or a short first word."""
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
            step = min(BLOCK - self._block_n, FRAME - len(self._frame_buf), len(x) - i)
            part = x[i : i + step]
            i += step
            self._n += step
            self._block_n += step
            self._frame_buf = np.concatenate((self._frame_buf, part))
            energy = self._energy(part)
            if self._ns_on:
                parts = self._ns_parts
                parts.append(part)
                self._ns_len += step
                while self._ns_len - len(parts[0]) >= _NS_WIN:
                    self._ns_len -= len(parts.popleft())
            self._block_energy += energy
            self._frame_energy += energy
            if playback:
                self._echo_until = self._n / SAMPLE_RATE + 0.5
            block = self._block_n == BLOCK
            if block:
                self._ns_quiet = self._in_speech and self._ns_on and self._subtracted_quiet()
            quiet = block and self._block_start(self._block_energy / BLOCK)
            if self._resume_due:
                self._resume(self._n / SAMPLE_RATE, events)
            if len(self._frame_buf) == FRAME:
                frame, self._frame_buf = self._frame_buf, np.zeros(0, dtype=np.float32)
                level, self._frame_energy = _dbfs(self._frame_energy / FRAME), 0.0
                self._frame(frame, level, self._n / SAMPLE_RATE, events)
            if block:
                self._block_n, self._block_energy = 0, 0.0
                self._block_end(quiet, self._n / SAMPLE_RATE, events)
        return events

    def _energy(self, part: np.ndarray) -> float:
        """Energy of up to one frame of input without its DC offset (only levels use this;
        Silero and the segment's audio get the input as it came)."""
        x = part.astype(np.float64)
        if self._dc_x is None:
            self._dc_x = float(x[0])  # an offset present from the first sample makes no transient
        d = np.empty_like(x)
        d[0] = x[0] - self._dc_x
        d[1:] = x[1:] - x[:-1]
        n = len(x)
        y = _DC_POW[:n] * (_DC_R * self._dc_y + _DC_GAIN * np.cumsum(d * _DC_INV[:n]))
        self._dc_x = float(x[-1])
        self._dc_y = float(y[-1]) if abs(y[-1]) > 1e-30 else 0.0
        return float(np.dot(y, y))

    @staticmethod
    def _bands(x: np.ndarray, layout: tuple) -> np.ndarray:
        hann, edges, scale = layout
        spectrum = np.fft.rfft(x * hann)
        return np.add.reduceat(spectrum.real ** 2 + spectrum.imag ** 2, edges) * scale

    def _subtracted_quiet(self) -> bool:
        """Is what the latest 20 ms leave above the room's spectrum below the quiet floor?"""
        # only where the room is close under the floor (else the level decides, as it always
        # did) and clearly under the speaker (else it cannot be told from them)
        if (self._ns_noise_db is None or self._level_db is None
                or not self._floor_db - NS_NEAR_DB <= self._ns_total_db <= self._level_db - NS_MIN_SNR_DB):
            return False
        x = np.concatenate(self._ns_parts)[-_NS_WIN:]
        if len(x) < _NS_WIN:
            return False
        left = float(np.sum(np.maximum(0.0, self._bands(x, _NS_BLOCK) - self._ns_noise)))
        return left <= 10.0 ** (self._floor_db / 10.0)

    def _learn_spectrum(self, frame: np.ndarray) -> None:
        """The room's spectrum follows the frames that are not speech."""
        seen = 10.0 * np.log10(self._bands(frame, _NS_FRAME) + 1e-12)
        n = self._ns_noise_db
        self._ns_noise_db = n = seen if n is None else np.minimum(seen, n + NS_RISE_DB)
        self._ns_noise = 10.0 ** ((n + self.noise_subtract_db) / 10.0)
        self._ns_total_db = 10.0 * math.log10(float(np.sum(10.0 ** (n / 10.0))) + 1e-12)

    # -- per 10 ms block (quiet clock) -------------------------------------
    def _block_start(self, power: float) -> bool:
        quiet = math.sqrt(power) <= self._quiet
        self._by_subtraction = not quiet and self._ns_quiet
        quiet = quiet or self._ns_quiet
        if self._t_quiet is not None:
            if quiet:
                self._loud = self._clear = 0
                self._quiet_blocks += self._unbroken
            else:
                self._unbroken = False
                self._loud += BLOCK
                heard = self._repause_lag_n is None or self._silent <= self._repause_lag_n
                if self._repause_n and not self._again and heard:
                    n = self._noise_db
                    if n is None or power >= 10.0 ** ((n + self.repause_room_db) / 10.0):
                        self._clear += 1
                        self._again = self._clear >= self._repause_n
                        self._resume_due = self._again
                if self._loud >= self._rearm_samples:
                    self._t_quiet, self._loud, self._resumed = None, 0, False
                    self._again = False
                    if heard:  # sound in the trailing silence takes nothing back
                        self._paused = False
                        self._resume_due = True
        return quiet

    def _block_end(self, quiet: bool, t: float, events: list[Event]) -> None:
        if not self._in_speech:
            return
        if (self.early_end and self._t_quiet is not None and self._unbroken
                and t - self._t_quiet + EPS >= self._target(t)):
            self._finish(t, events, early=True)
        elif quiet and (self._t_quiet is None or not (self._unbroken or self._resumed)):
            # armed, or counting anew after a click or a breath broke the count (speech that
            # Silero heard resume keeps it off until real sound re-arms it)
            self._t_quiet, self._unbroken, self._loud, self._quiet_blocks = t, True, 0, 1
            if self._again:  # real sound since the last "pause": this one is a better estimate
                self._paused = self._again = False
            self._clear = 0
            raised = self._quiet > self.quiet_rms * (1 + 1e-6) or self._by_subtraction
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
    def _frame(self, frame: np.ndarray, level: float, t: float, events: list[Event]) -> None:
        t0 = time.perf_counter()
        gain = self._gain_db
        if gain >= 0.01:  # never into clipping: a loud frame right after a soft one gets less
            peak = float(np.max(np.abs(frame)))
            gain = max(0.0, min(gain, -6.0 - 20.0 * math.log10(peak))) if peak > 0.0 else 0.0
        p = self._p(frame if gain < 0.01 else frame * np.float32(10.0 ** (gain / 20.0)))
        infer_s = time.perf_counter() - t0
        if self._in_speech:
            voiced = p >= self.continue_threshold
        else:  # idle: the first frame needs start_threshold, the rest of the debounce confirm_threshold
            voiced = p >= (self.confirm_threshold if self._streak else self.start_threshold)
        echo = t <= self._echo_until
        self._history.append((frame, level, echo))
        if (self.normalize_to_dbfs is not None and p >= 0.5 and not echo
                and (self._floor is None or level >= self._floor + SOFT_OVER_DB)):  # not Silero's tail over the room
            self._gain_seen.append(level)
            want = self.normalize_to_dbfs - sorted(self._gain_seen)[len(self._gain_seen) // 2]
            want = max(0.0, min(self.normalize_max_db, want))
            self._gain_db += max(-2.0, min(2.0, want - self._gain_db))
        self._levels(frame, p, level, echo)
        soft = (self.soft_start_s is not None and p >= self._soft_level()
                and (self._floor is None or level >= self._floor + SOFT_OVER_DB))
        if self._in_speech:
            # Only while idle. Speech that resumed loud enough has opened a new segment: inside
            # it a frame is voiced by Silero's probability alone, or its silence count would
            # begin over the quiet audio before Silero's does and end it before Silero alone.
            self._tail = False
        if self._tail:  # after an early end, Silero's fading tail must not open a new segment...
            self._tail = p >= self.continue_threshold
            soft = False  # ...softly at all (a breath after the word would), and not confidently...
            if level <= 20.0 * math.log10(max(self._quiet, 1e-9)):
                voiced = False  # ...over quiet audio; speech that resumes at once is loud and may start
        if not self._in_speech:
            self._idle_frame(p, voiced, soft, level, t, events)
            self._report(t, p, voiced, infer_s, events)
            return
        self._audio.append(frame)
        self._peak = max(self._peak, p)
        if voiced:
            if self._silent:
                self._paused = False
                self._resume(t, events)
                if self._t_quiet is not None and t > self._t_quiet:
                    self._unbroken, self._resumed = False, True
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

    def _levels(self, frame: np.ndarray, p: float, level: float, echo: bool) -> None:
        """Learn speech and room-noise levels and set the quiet threshold."""
        # A segment that began softly may never reach start_threshold: its levels are learned
        # from the frames that carry it.
        # (After an early end, until Silero's own count would have ended the segment, everything
        # the recovery below reads is kept as it would be without the early end.)
        shadow = self._shadow is not None
        soft_segment = self._shadow_soft if shadow else self._in_speech and self._soft_segment
        carries = p >= (self._soft_level() if soft_segment else self.start_threshold)
        if carries:
            s = self._speech_db
            self._speech_db = level if s is None else s + max(-1.0, min(1.0, level - s))
            if self._in_speech:
                self._learn_level(level)
        noise = not carries and p < self.continue_threshold and not echo
        if noise and self._ns_on:  # over gated digital silence the room's spectrum falls: nothing to subtract there
            self._learn_spectrum(frame)
        if noise and level > -90.0:  # the noise level skips gated digital silence
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
            # The floor is quiet_rms, or floor_margin_db below the speaker when that is lower
            # (quiet input). Noise can raise the threshold above the floor, but only up to the
            # latest speech level - 15 dB; it never drops below the floor.
            floor_db = 20.0 * math.log10(self.quiet_rms)
            if self._level_db is not None and self.floor_margin_db is not None:
                floor_db = min(floor_db, self._level_db - self.floor_margin_db)
            self._floor_db = db = floor_db
            if self._noise_db is not None:
                db = max(db, self._noise_db + self.noise_margin_db)
            if self._speech_db is not None:
                db = max(floor_db, min(db, self._speech_db - self.speech_margin_db))
            self._quiet = 10.0 ** (db / 20.0)
        # Silero's recurrent state drifts in long noisy or near-silent stretches until it is
        # "deaf" (p ~ 0 through loud speech). Restore a state from just after the user's speech
        # after a second of p < 0.1 over real audio (gated digital silence doesn't drift it, and
        # a restore there only disturbs the next onset), never on a frame that sounds like an
        # onset; or at once when Silero has ignored speech-loud audio for deaf_s.
        #
        # The restore point is the state on the first frame where Silero lets go of the
        # user's confident speech (p below continue_threshold): it has just heard speech, so
        # it hears the next speech at once, and it is no longer in speech. A state taken
        # during confident speech replays a speech tail onto whatever follows the restore
        # (steady room noise 25 dB below the speaker: p ~0.92 decaying over 0.3 s), and that
        # opened phantom segments. Never after the assistant's own voice either: a state that
        # last heard playback echo scores following room noise as speech (p ~0.96 decaying
        # over 0.4 s).
        restore = getattr(self._p, "restore", None)
        snapshot = getattr(self._p, "snapshot", None)
        if echo:
            self._settling = False
        elif p >= self.start_threshold:
            self._settling = True
        elif self._settling and p < self.continue_threshold:
            self._settling = False
            if snapshot is not None:
                snapshot()
        # After an early end Silero's own count would still be running. Until it would have
        # ended the segment too, this counts as speech here, so a restore comes exactly when it
        # would without the quiet clock and the probabilities that follow are the same.
        in_speech = self._in_speech or self._shadow is not None
        self._shadowed = in_speech and not self._in_speech  # this frame: idle here, in speech for Silero alone
        if self._shadow is not None and not self._in_speech:
            self._shadow = 0 if p >= self.continue_threshold else self._shadow + 1
            if self._shadow >= self._end_frames:
                self._shadow = None  # this frame is the one Silero's count would end on
        if in_speech:
            self._quiet_restored = False  # a new pause may restore once more
        if not self._restore_n or restore is None or in_speech or echo:
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

    def _soft_level(self) -> float:
        return max(self.soft_start_threshold, self.confirm_threshold)

    def _learn_level(self, level: float) -> None:
        seen = self._levels_seen
        seen.append(level)
        if self._level_db is None or len(seen) * FRAME_S >= LEVEL_MIN_S:
            self._level_db = sorted(seen)[len(seen) // 2]

    def _speech_like(self, level: float) -> bool:
        """Loud like the speaker and clearly above the idle floor."""
        if self._speech_db is None or self._floor is None:
            return False
        margin = max(3.0, min(10.0, 0.5 * (self._speech_db - self._floor)))
        return level >= max(self._floor + margin, self._speech_db - 10.0)

    def _idle_frame(self, p: float, voiced: bool, soft: bool, level: float, t: float, events: list[Event]) -> None:
        floor = self._floor
        if self._lookback_n and not self._shadowed:  # the idle floor is learned where Silero alone is idle too
            if floor is None or level <= floor:
                self._floor = level if floor is None else floor + (level - floor) * 0.5
            else:
                self._floor = floor + min(level - floor, 3.0 * FRAME_S)
        self._idle += 1
        if self._streak >= FIRST_WORD_FRAMES and not voiced:  # speech that did not last start_s
            first, last = self._idle - self._streak, self._idle - 1
            if self._tried is not None and first - self._tried[1] <= self._end_frames:
                first = self._tried[0]  # one stretch with the one before
            self._tried = (first, last)
        self._streak = self._streak + 1 if voiced else 0
        self._peak = max(self._peak, p) if voiced else 0.0
        self._soft = self._soft + 1 if soft else 0
        if self._streak == 1 and self._onset_events:
            events.append(Event("onset", t))  # maybe speech; "start" follows only if it holds
        self._soft_segment = False
        if self._streak < self._start_frames:
            if not soft or self._soft < max(self._start_frames, _frames(self.soft_start_s)):
                return
            if self._onset_events and not self._streak:
                events.append(Event("onset", t))  # no confident frame came: the onset is the start
            self._streak, self._soft_segment = self._soft, True
            self._peak = max(self._peak, p)
        lookback = max(self._lookback(floor), self._first_word())
        # the audio begins as long before the speech as with the default start_s, however long
        # a longer start_s takes to confirm (preroll_s is counted back from the confirmation)
        preroll = self._preroll_n + max(0, self._start_frames - FIRST_WORD_FRAMES)
        keep = preroll if not lookback else max(preroll, min(preroll + lookback, self._idle))
        self._in_speech = True
        self._audio = [f for f, _, _ in list(self._history)[-keep:]]
        self._start = t - (self._streak - 1 + lookback) * FRAME_S  # end of first streak frame, as in Miep
        self._lookback_s = lookback * FRAME_S
        self._voiced = self._streak
        self._last_voiced = t
        self._levels_seen.clear()  # the speaker's level is learned anew in every segment
        for _, past, _ in list(self._history)[-self._streak:]:
            self._learn_level(past)
        events.append(Event("start", self._start))

    def _first_word(self) -> int:
        """Frames before the start streak back to a short first word ("Hey. What's up").

        With a long start_s a short word does not last long enough to start a segment, and the
        start would come at the second word. Speech of at least 64 ms (what the default
        start_s asks for) that stopped less than end_s before the start streak began belongs
        to the segment: with the default start_s it would have started it, and the pause
        after it would not have ended it. A sound that stands alone still starts nothing.
        """
        h, s = self._history, self._streak
        if self._tried is None:
            return 0
        first, last = self._tried
        begin = self._idle - s + 1  # first frame of the start streak
        back = begin - first
        if begin - last > self._end_frames or back > len(h) - s:
            return 0
        if any(h[-(s + step)][2] for step in range(1, back + 1)):  # never over the assistant's own voice
            return 0
        return back

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
            self._paused = self._pause_open = True
            events.append(Event("pause", t, source=source))

    def _resume(self, t: float, events: list[Event]) -> None:
        self._resume_due = False
        if self._pause_open:
            self._pause_open = False
            if self._resume_events:
                events.append(Event("resume", t))

    def _finish(self, t: float, events: list[Event], early: bool = False) -> None:
        # The speech stopped where the unbroken quiet began, if that is before Silero's last
        # voiced frame (its probability lags behind the audio).
        speech_end = self._last_voiced
        if self._t_quiet is not None and self._unbroken and (early or self._t_quiet < speech_end):
            speech_end = self._t_quiet
        segment = Segment(
            audio=self.audio(),
            start=self._start,
            end=t,
            speech_end=speech_end,
            voiced_s=self._voiced * FRAME_S,
            peak=self._peak,
            early=early,
        )
        self._in_speech = False
        self._tail = early  # ended by the quiet clock, Silero may still be high
        if early and self._shadow is None:
            self._shadow_soft = self._soft_segment  # the segment Silero alone is still in
        self._shadow = self._silent if early else None  # Silero's count, carried on
        self._idle = self._streak = self._soft = 0
        self._tried = None
        self._clear_segment()
        events.append(Event("end", t, segment))

    def _clear_segment(self) -> None:
        self._audio: list[np.ndarray] = []
        self._start = self._last_voiced = self._peak = 0.0
        self._voiced = self._silent = 0
        self._end_target = self._end_frames
        self._paused = False
        self._pause_open = False  # a "pause" was reported and no "resume" since
        self._resume_due = False
        self._t_quiet: Optional[float] = None
        self._unbroken = False
        self._resumed = False  # Silero heard speech resume inside the running quiet count
        self._loud = 0
        self._clear = 0  # blocks in a row clearly above the room since the count last ran
        self._again = False  # ...enough of them: the restarted count reports a new "pause"
        self._quiet_blocks = 0
        self._quiet_wait = self.quiet_end_s
        self._by_subtraction = False
