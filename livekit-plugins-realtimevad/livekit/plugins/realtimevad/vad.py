"""RealtimeVAD behind LiveKit's VAD interface.

How RealtimeVAD's signals map to LiveKit's three event types:

  start  -> START_OF_SPEECH   same moment; ``speech_duration`` reaches back to where the speech
                              began (the confirming frames plus a soft opening before them, or
                              where Silero first passed 0.5, whichever is earlier)
  end    -> END_OF_SPEECH     same moment; ``silence_duration`` reaches back to the instant the
                              audio went quiet (or Silero's last voiced frame), so LiveKit's
                              "user stopped speaking" time is the real one
  pause  -> no event of its own (LiveKit has none). The quiet clock runs into
            ``raw_accumulated_silence`` of the INFERENCE_DONE events, which is what LiveKit
            starts its turn detector on at 0.2 s, and stops ``raw_accumulated_speech``, which is
            what LiveKit refreshes "last speaking" from. Also emitted as ``vad.on("pause")``.
  onset  -> ``vad.on("onset")`` only.

INFERENCE_DONE comes once per 32 ms model frame, as with the Silero plugin, and carries the
audio of that frame. Interruptions need nothing special: LiveKit interrupts when
``speech_duration`` of an INFERENCE_DONE reaches its ``min_duration``, counted from the start.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad

from RealtimeVAD import VAD as _Core
from RealtimeVAD.silero import FRAME, SAMPLE_RATE

from .log import logger

FRAME_S = FRAME / SAMPLE_RATE
SLICE = 160  # the core is fed in 10 ms slices so the quiet clock is read at its own resolution


@dataclass
class _Options:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float
    core: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PauseEvent:
    """``vad.on("pause")``: the speaker may have stopped. Early and often wrong; never an end."""

    timestamp: float  # audio time of the stream, like VADEvent.timestamp
    source: str  # "quiet" (10 ms quiet clock) or "silero" (first unvoiced frame)


class VAD(agents_vad.VAD):
    """``VAD.load()`` returns a VAD for ``AgentSession(vad=...)``."""

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.064,
        min_silence_duration: float = 0.25,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.9,
        deactivation_threshold: float = 0.45,
        **core_options: Any,
    ) -> "VAD":
        """
        Args:
            min_speech_duration: confident speech needed before START_OF_SPEECH.
            min_silence_duration: quiet (or Silero silence) before END_OF_SPEECH. The Silero
                plugin's default is 0.55; LiveKit's audio turn detector refuses less than 0.25.
            prefix_padding_duration: audio before the speech kept in the event frames.
            max_buffered_speech: longest speech kept in END_OF_SPEECH frames (the newest is kept).
            activation_threshold: Silero probability that starts speech (RealtimeVAD's 0.9, not
                the Silero plugin's 0.5; speech that stays lower starts through the soft start).
            deactivation_threshold: probability below which a frame counts as silence.
            core_options: any other ``RealtimeVAD.VAD`` argument, e.g. ``soft_start_s=None``,
                ``quiet_rms=0`` (quiet clock off: Silero's count alone), ``floor_margin_db``.
        """
        return cls(_Options(min_speech_duration, min_silence_duration, prefix_padding_duration,
                            max_buffered_speech, activation_threshold, deactivation_threshold, dict(core_options)))

    def __init__(self, opts: _Options) -> None:
        super().__init__(capabilities=agents_vad.VADCapabilities(update_interval=FRAME_S))
        self._opts = opts

    @property
    def model(self) -> str:
        return "realtimevad"

    @property
    def provider(self) -> str:
        return "ONNX"

    @property
    def min_silence_duration(self) -> float:
        return self._opts.min_silence_duration

    def stream(self) -> "VADStream":
        return VADStream(self, self._opts)


class VADStream(agents_vad.VADStream):
    def __init__(self, vad: VAD, opts: _Options) -> None:
        super().__init__(vad)
        self._opts = opts
        self._core: Optional[_Core] = None

    def _new_core(self) -> _Core:
        o = self._opts
        return _Core(**{
            "start_s": o.min_speech_duration, "end_s": o.min_silence_duration,
            "preroll_s": o.prefix_padding_duration + o.min_speech_duration,  # the core counts the confirming frames in
            "start_threshold": o.activation_threshold,
            "continue_threshold": o.deactivation_threshold, **o.core,
            "frame_events": True, "onset_events": True,
        })

    def _feed(self, pcm: np.ndarray) -> list:
        """Runs in the executor: (event, quiet-clock seconds, start of the segment's audio)."""
        core, out = self._core, []
        for i in range(0, len(pcm), SLICE):
            part = pcm[i : i + SLICE]
            self._fed += len(part)
            found = core.feed(part)
            armed, unbroken, blocks = core.quiet_clock
            quiet = blocks * SLICE / SAMPLE_RATE if armed and unbroken else 0.0
            ended = next((e for e in found if e.kind == "end"), None)
            for e in found:
                seg, silent = None, quiet
                if e.kind == "start":  # (where the segment's audio begins, soft opening before the confirming frames)
                    seg = ((self._fed - len(core.audio())) / SAMPLE_RATE, core.onset_lookback_s)
                elif e.kind == "frame" and ended is not None and e.time <= ended.time:
                    silent = max(0.0, e.time - ended.segment.speech_end)  # the end in this slice already cleared the clock
                out.append((e, silent, seg))
        return out

    def _reset(self) -> None:
        if self._core is not None:
            self._core.reset()
        self._fed = 0  # 16 kHz samples fed since the stream began (or the last flush)
        self._buf = np.zeros(0, dtype=np.int16)  # input-rate audio for the event frames
        self._buf_t0 = 0.0  # stream time of its first sample
        self._pushed = 0  # input-rate samples received
        self._speaking = False
        self._begin = 0.0  # when the confirming debounce began
        self._seg_from = 0.0  # where the segment's audio begins (prefix padding)
        self._silence_from = 0.0  # when the last speech stopped
        self._speech_run = self._silence_run = 0.0
        self._half_run = 0  # frames in a row with Silero at 0.5 or more
        self._window_from = 0.0

    def _frames(self, t0: float, t1: float) -> list[rtc.AudioFrame]:
        a = max(0, int(round((t0 - self._buf_t0) * self._rate)))
        b = min(len(self._buf), int(round((t1 - self._buf_t0) * self._rate)))
        data = self._buf[a:b] if b > a else np.zeros(0, dtype=np.int16)
        return [rtc.AudioFrame(data=data.tobytes(), sample_rate=self._rate, num_channels=1, samples_per_channel=len(data))]

    def _trim(self) -> None:
        """Keep the running segment (up to max_buffered_speech), or the padding while idle."""
        if self._speaking:
            keep = min(self._pushed / self._rate - self._seg_from,
                       self._opts.max_buffered_speech + self._opts.prefix_padding_duration)
        else:
            keep = self._opts.prefix_padding_duration + 1.2  # plus the core's soft-opening look-back
        drop = len(self._buf) - int(keep * self._rate)
        if drop > self._rate:  # about once a second
            self._buf = self._buf[drop:]
            self._buf_t0 += drop / self._rate

    async def _main_task(self) -> None:
        loop = asyncio.get_running_loop()
        self._rate = 0
        resampler: Optional[rtc.AudioResampler] = None
        self._reset()
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._reset()  # a hard segment boundary: no END for speech that was running
                continue
            if not isinstance(item, rtc.AudioFrame):
                continue
            if not self._rate:
                self._rate = item.sample_rate
                if self._rate != SAMPLE_RATE:
                    resampler = rtc.AudioResampler(input_rate=self._rate, output_rate=SAMPLE_RATE,
                                                   quality=rtc.AudioResamplerQuality.QUICK)
                self._core = await loop.run_in_executor(None, self._new_core)
            elif item.sample_rate != self._rate:
                logger.error("a frame with another sample rate was already pushed")
                continue
            data = np.frombuffer(item.data, dtype=np.int16)
            if item.num_channels > 1:
                data = data.reshape(-1, item.num_channels)[:, 0]
                item = rtc.AudioFrame(data=data.tobytes(), sample_rate=self._rate, num_channels=1,
                                      samples_per_channel=len(data))
            self._buf = np.concatenate((self._buf, data))
            self._pushed += len(data)
            frames16 = resampler.push(item) if resampler is not None else [item]
            for f in frames16:
                pcm = np.frombuffer(f.data, dtype=np.int16)
                self._publish(await loop.run_in_executor(None, self._feed, pcm))
            self._trim()

    def _publish(self, results: list) -> None:
        send, T = self._event_ch.send_nowait, agents_vad.VADEventType
        for i, (e, quiet_s, seg) in enumerate(results):
            if e.kind == "start":
                # the frame event that follows in the same step carries the time and the debounce length
                frame = next(r[0] for r in results[i + 1 :] if r[0].kind == "frame")
                now = frame.time
                self._speaking, self._seg_from = True, seg[0]
                # the speech began with the confirming frames and a soft opening before them, or
                # where Silero first passed 0.5 on the way up, whichever is earlier
                rising = (self._half_run + 1) * FRAME_S if frame.frame.probability >= 0.5 else 0.0
                self._begin = now - max(frame.frame.streak * FRAME_S + seg[1], rising)
                self._speech_run, self._silence_run = frame.frame.streak * FRAME_S, 0.0
                send(agents_vad.VADEvent(
                    type=T.START_OF_SPEECH, samples_index=round(now * SAMPLE_RATE), timestamp=now,
                    speech_duration=now - self._begin, silence_duration=0.0,
                    frames=self._frames(seg[0], now), speaking=True))
            elif e.kind == "end":
                s = e.segment
                silence = max(0.0, e.time - s.speech_end)
                self._speaking, self._silence_from, self._silence_run = False, s.speech_end, silence
                self._speech_run = 0.0
                send(agents_vad.VADEvent(
                    type=T.END_OF_SPEECH, samples_index=round(e.time * SAMPLE_RATE), timestamp=e.time,
                    speech_duration=max(0.0, s.speech_end - self._begin), silence_duration=silence,
                    frames=self._frames(self._seg_from, e.time), speaking=False))
            elif e.kind == "frame":
                f = e.frame
                self._half_run = self._half_run + 1 if f.probability >= 0.5 else 0
                if self._speaking:
                    silent = max(quiet_s, f.silent_frames * FRAME_S)  # the quiet clock is ahead of Silero
                    self._speech_run = 0.0 if silent else self._speech_run + FRAME_S
                    self._silence_run = silent
                elif f.streak:
                    self._speech_run, self._silence_run = f.streak * FRAME_S, 0.0
                else:
                    self._speech_run, self._silence_run = 0.0, self._silence_run + FRAME_S
                send(agents_vad.VADEvent(
                    type=T.INFERENCE_DONE, samples_index=round(e.time * SAMPLE_RATE), timestamp=e.time,
                    speech_duration=e.time - self._begin if self._speaking else 0.0,
                    silence_duration=0.0 if self._speaking else e.time - self._silence_from,
                    frames=self._frames(self._window_from, e.time), probability=f.probability,
                    inference_duration=f.infer_s, speaking=self._speaking,
                    raw_accumulated_silence=self._silence_run, raw_accumulated_speech=self._speech_run))
                self._window_from = e.time
            elif e.kind in ("pause", "onset"):
                self._vad.emit(e.kind, PauseEvent(e.time, e.source) if e.kind == "pause" else e.time)
