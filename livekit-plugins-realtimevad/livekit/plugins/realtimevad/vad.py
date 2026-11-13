"""RealtimeVAD behind LiveKit's VAD interface.

How RealtimeVAD's signals map to LiveKit's three event types:

  start  -> START_OF_SPEECH   same moment; ``speech_duration`` reaches back to where the speech
                              began (the confirming frames plus a soft opening before them, or
                              where Silero first passed 0.5, whichever is earlier)
  end    -> END_OF_SPEECH     same moment; ``silence_duration`` reaches back to the instant the
                              audio went quiet (or Silero's last voiced frame), so LiveKit's
                              "user stopped speaking" time is the real one. By default the end
                              comes where the Silero plugin puts it (see ``early_end``).
  pause  -> no event of its own (LiveKit has none). The quiet clock runs into
            ``raw_accumulated_silence`` of the INFERENCE_DONE events, which is what LiveKit
            starts its turn detector on at 0.2 s, and stops ``raw_accumulated_speech``, which is
            what LiveKit refreshes "last speaking" from. Also emitted as ``vad.on("pause")``.
  onset  -> ``vad.on("onset")`` only.

INFERENCE_DONE comes once per 32 ms model frame, as with the Silero plugin, and carries the
audio of that frame. Interruptions need nothing special: LiveKit interrupts when
``speech_duration`` of an INFERENCE_DONE reaches its ``min_duration``, counted from the start.

LiveKit's VAD interface (``livekit.agents.vad``, 1.8.11) passes a stream audio frames, ``flush()``
and ``end_input()`` and nothing else: a VAD is not told when the agent speaks. RealtimeVAD's
``feed(chunk, playback=True)`` is therefore driven by ``VAD.set_agent_speaking()``, which the
application calls (from the session's ``agent_state_changed`` event).
"""

from __future__ import annotations

import asyncio
import inspect
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from RealtimeVAD import VAD as _Core
from RealtimeVAD.silero import FRAME, SAMPLE_RATE

from .log import logger

FRAME_S = FRAME / SAMPLE_RATE
SLICE = 160  # the core is fed in 10 ms slices so the quiet clock is read at its own resolution
NEVER = 1e6  # a quiet-clock wait that is never reached
# The Silero plugin (livekit-plugins-silero 1.8.11, vad.py) counts silence on a smoothed probability:
# ExpFilter(alpha=0.35), silent while it is at or below its deactivation threshold 0.35.
SILERO_PLUGIN_ALPHA, SILERO_PLUGIN_DEACTIVATION = 0.35, 0.35
_CORE_HAS_EARLY_END = "early_end" in inspect.signature(_Core.__init__).parameters


@dataclass
class _Options:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float
    early_end: bool = False
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
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.9,
        deactivation_threshold: float = 0.45,
        early_end: bool = False,
        **core_options: Any,
    ) -> "VAD":
        """
        Args:
            min_speech_duration: confident speech needed before START_OF_SPEECH.
            min_silence_duration: Silero silence before END_OF_SPEECH; the Silero plugin's
                default. LiveKit's audio turn detector refuses less than 0.25.
            early_end: False (default): END_OF_SPEECH is not sent before the Silero plugin would
                send it at the same ``min_silence_duration``; the earlier stop is in
                ``raw_accumulated_silence`` (the pause). True: the segment also ends after
                ``min_silence_duration`` of quiet audio, about 100 ms sooner and more often
                inside a sentence.
            prefix_padding_duration: audio before the speech kept in the event frames.
            max_buffered_speech: longest speech kept in END_OF_SPEECH frames (the newest is kept).
            activation_threshold: Silero probability that starts speech (RealtimeVAD's 0.9, not
                the Silero plugin's 0.5; speech that stays lower starts through the soft start).
            deactivation_threshold: probability below which a frame counts as silence.
            core_options: any other ``RealtimeVAD.VAD`` argument, e.g. ``soft_start_s=None``,
                ``floor_margin_db``, ``normalize_to_dbfs``.
        """
        return cls(_Options(min_speech_duration, min_silence_duration, prefix_padding_duration,
                            max_buffered_speech, activation_threshold, deactivation_threshold, early_end,
                            dict(core_options)))

    def __init__(self, opts: _Options) -> None:
        super().__init__(capabilities=agents_vad.VADCapabilities(update_interval=FRAME_S))
        self._opts = opts  # shared with the streams
        self._streams: "weakref.WeakSet[VADStream]" = weakref.WeakSet()
        self._agent_speaking = False

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
        stream = VADStream(self, self._opts)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        deactivation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Change options of this VAD, its running streams included (the Silero plugin's signature).

        A running stream keeps everything it has learned about the room and the speaker. A new
        ``min_silence_duration`` counts from the next pause; a pause that is already being counted
        ends on the old one. ``prefix_padding_duration`` and ``max_buffered_speech`` apply from the
        next START_OF_SPEECH. An option that was also given to ``load()`` under RealtimeVAD's own
        name (``end_s``, ``quiet_end_s``, ``start_s``, ...) stays as given there.
        """
        o = self._opts
        for name, value in (("min_speech_duration", min_speech_duration),
                            ("min_silence_duration", min_silence_duration),
                            ("prefix_padding_duration", prefix_padding_duration),
                            ("max_buffered_speech", max_buffered_speech),
                            ("activation_threshold", activation_threshold),
                            ("deactivation_threshold", deactivation_threshold)):
            if is_given(value):
                setattr(o, name, value)
        for stream in self._streams:
            stream._retune()

    def set_agent_speaking(self, speaking: bool) -> None:
        """Tell the VAD that the agent's own voice may be in the microphone signal.

        While set, and for 0.5 s after it is cleared, audio is fed to RealtimeVAD with
        ``playback=True``: room noise is not learned from it and Silero's recovery state is not
        taken from it. Speech is still detected, so the user can interrupt. LiveKit does not tell a
        VAD when the agent speaks; wire it to the session::

            session.on("agent_state_changed", lambda ev: vad.set_agent_speaking(ev.new_state == "speaking"))
        """
        self._agent_speaking = bool(speaking)

    @property
    def agent_speaking(self) -> bool:
        return self._agent_speaking


class VADStream(agents_vad.VADStream):
    def __init__(self, vad: VAD, opts: _Options) -> None:
        super().__init__(vad)
        self._opts = opts
        self._core: Optional[_Core] = None

    def _new_core(self) -> _Core:
        o = self._opts
        args = {"start_s": o.min_speech_duration, "end_s": o.min_silence_duration,
                "start_threshold": o.activation_threshold, "continue_threshold": o.deactivation_threshold}
        if _CORE_HAS_EARLY_END:
            args["early_end"] = o.early_end
        elif not o.early_end:
            args["quiet_end_s"] = NEVER  # RealtimeVAD 0.0.5: the quiet clock always may end a segment
        args.update(o.core)
        self._own_extend = args.get("extend")
        if not args.get("early_end", o.early_end):
            args["extend"] = self._hold_for_silero_plugin
        # the event frames are cut from this stream's own buffer: the core's preroll is not used
        return _Core(**args, frame_events=True, onset_events=True)

    def _hold_for_silero_plugin(self, t: float) -> Optional[float]:
        """The core's ``extend``: asked on every frame once its own Silero count is full.

        The core counts frames whose probability is below ``deactivation_threshold``. The Silero
        plugin counts on a smoothed probability, which falls a frame or two later, so the same
        number ends a segment there a frame or two later, and a pause of just that length ends it
        here and not there. Hold the end until the Silero plugin's count would be full as well.
        Runs inside ``core.feed`` before this frame's event: the counters are one frame behind.
        """
        missing = self._opts.min_silence_duration - (self._plugin_silent + 1) * FRAME_S
        hold = (self._core_silent + 1) * FRAME_S + missing if missing > 1e-9 else None
        own = self._own_extend(t) if self._own_extend is not None else None
        return own if hold is None else hold if own is None else max(hold, own)

    def _retune(self) -> None:
        """Options changed (``VAD.update_options``): bring the running core in line, keeping its state."""
        core, o = self._core, self._opts
        if core is None:
            return  # not built yet: _new_core reads the options
        own = o.core  # RealtimeVAD's own names given to load() win, as they do in _new_core
        if "start_s" not in own:
            core.start_s = o.min_speech_duration
        if "start_threshold" not in own:
            core.start_threshold = o.activation_threshold
        if "continue_threshold" not in own:
            core.continue_threshold = o.deactivation_threshold
            if "confirm_threshold" not in own:
                core.confirm_threshold = o.deactivation_threshold
        if "end_s" not in own and core.end_s != o.min_silence_duration:
            core.end_s = o.min_silence_duration
            if "quiet_end_s" not in own and (o.early_end or _CORE_HAS_EARLY_END):
                core.quiet_end_s = o.min_silence_duration  # read when the quiet clock next arms
            # Silero's silence count is a number of frames that RealtimeVAD (0.0.5) derives from
            # end_s only when it is built; it has no public setter for it yet.
            if hasattr(core, "_end_frames") and hasattr(core, "_end_target"):
                frames = max(1, math.ceil(o.min_silence_duration / FRAME_S - 1e-9))
                extended = core._end_target != core._end_frames  # an ``extend`` callback holds this pause open
                core._end_frames = frames
                if not extended:
                    core._end_target = frames
            else:
                logger.warning("this RealtimeVAD version cannot change min_silence_duration on a running "
                               "stream; it applies to streams opened from now on")

    def _feed(self, pcm: np.ndarray, playback: bool = False) -> list:
        """Runs in the executor: (event, quiet-clock seconds, soft opening before the confirming frames)."""
        core, out = self._core, []
        for i in range(0, len(pcm), SLICE):
            part = pcm[i : i + SLICE]
            self._fed += len(part)
            found = core.feed(part, playback=playback)
            armed, unbroken, blocks = core.quiet_clock
            quiet = blocks * SLICE / SAMPLE_RATE if armed and unbroken else 0.0
            ended = next((e for e in found if e.kind == "end"), None)
            for e in found:
                seg, silent = None, quiet
                if e.kind == "frame":  # what the Silero plugin would count, for _hold_for_silero_plugin
                    p = e.frame.probability
                    self._smoothed = p if self._smoothed is None else (
                        SILERO_PLUGIN_ALPHA * self._smoothed + (1 - SILERO_PLUGIN_ALPHA) * p)
                    self._plugin_silent = 0 if self._smoothed > SILERO_PLUGIN_DEACTIVATION else self._plugin_silent + 1
                    self._core_silent = e.frame.silent_frames
                if e.kind == "start":
                    seg = core.onset_lookback_s
                elif e.kind == "frame" and ended is not None and e.time <= ended.time:
                    silent = max(0.0, e.time - ended.segment.speech_end)  # the end in this slice already cleared the clock
                out.append((e, silent, seg))
        return out

    def _reset(self) -> None:
        if self._core is not None:
            self._core.reset()
        self._fed = 0  # 16 kHz samples fed since the stream began (or the last flush)
        self._smoothed: Optional[float] = None  # the Silero plugin's smoothed probability
        self._plugin_silent = self._core_silent = 0  # frames of silence: its count, the core's count
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
                self._publish(await loop.run_in_executor(None, self._feed, pcm, self._vad._agent_speaking))
            self._trim()

    def _publish(self, results: list) -> None:
        send, T = self._event_ch.send_nowait, agents_vad.VADEventType
        for i, (e, quiet_s, seg) in enumerate(results):
            if e.kind == "start":
                # the frame event that follows in the same step carries the time and the debounce length
                frame = next(r[0] for r in results[i + 1 :] if r[0].kind == "frame")
                now = frame.time
                # the segment's audio: the padding, a soft opening, the confirming frames
                confirmed = frame.frame.streak * FRAME_S + seg
                self._speaking = True
                self._seg_from = max(self._buf_t0, now - confirmed - self._opts.prefix_padding_duration)
                # the speech began with the confirming frames and a soft opening before them, or
                # where Silero first passed 0.5 on the way up, whichever is earlier
                rising = (self._half_run + 1) * FRAME_S if frame.frame.probability >= 0.5 else 0.0
                self._begin = now - max(confirmed, rising)
                self._speech_run, self._silence_run = frame.frame.streak * FRAME_S, 0.0
                send(agents_vad.VADEvent(
                    type=T.START_OF_SPEECH, samples_index=round(now * SAMPLE_RATE), timestamp=now,
                    speech_duration=now - self._begin, silence_duration=0.0,
                    frames=self._frames(self._seg_from, now), speaking=True))
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
                    # The quiet clock is ahead of Silero. Sound after a pause breaks the clock, and
                    # the silence falls back to Silero's own count: under it the pause starts anew,
                    # but sound that Silero does not call speech never makes the silence shorter
                    # than Silero's (a noise after the last word must not undo the pause).
                    silent = max(quiet_s, f.silent_frames * FRAME_S)
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
