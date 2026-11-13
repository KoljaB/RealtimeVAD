"""Speculative transcription: the STT starts on the user's pause, not on the VAD's END_OF_SPEECH.

LiveKit starts a reply early ("preemptive generation") from transcripts only: a final transcript,
or a PREFLIGHT_TRANSCRIPT ("stable enough to be used for preemptive generation",
``livekit.agents.stt.SpeechEventType``). Nothing in LiveKit reads a VAD's pause for that. But an
STT is free to produce those transcripts whenever it likes, and LiveKit's own ``StreamAdapter``
shows the shape: an ``stt.STT`` built from another STT and a VAD. ``SpeculativeSTT`` is that
adapter with the recognition moved forward to the pause:

  wrapped STT is not streaming   at the pause it sends the segment so far to ``recognize()`` and
      (Whisper and the like)     emits the result as PREFLIGHT_TRANSCRIPT. LiveKit starts the reply
                                 then. At END_OF_SPEECH the same text goes out as FINAL_TRANSCRIPT
                                 without a second request, so the reply that is already being
                                 generated is the one LiveKit uses. If the user spoke on after the
                                 pause, the result is dropped and the next pause, or the END,
                                 transcribes again.
  wrapped STT is streaming and   at the pause it calls ``flush()`` on the recognizer's stream, which
  has ``manual_flush``           finalizes what was said so far. LiveKit starts the reply from that
                                 final transcript. (LiveKit itself flushes such an STT only at the
                                 VAD's END_OF_SPEECH.)

"Pause" is read from standard VAD events: an INFERENCE_DONE inside speech whose
``raw_accumulated_silence`` reaches ``pause`` seconds. With RealtimeVAD that counter runs from the
instant the audio went quiet; with the Silero plugin it starts about 100 ms later, so the adapter
works with any LiveKit VAD and is that much earlier with this one.

A pause is often not the end. Then the transcript was requested for nothing and the reply LiveKit
began from it is cancelled when the next transcript arrives; nothing of it is spoken, because
LiveKit schedules a preemptive reply only once the turn is committed. The costs are extra STT
requests and extra LLM calls, and LiveKit starts at most ``max_retries`` (default 3) preemptive
replies per turn and none once the user has spoken for ``max_speech_duration`` (default 10 s);
see ``turn_handling={"preemptive_generation": {...}}``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents import vad as agents_vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

from .log import logger

T, V = stt.SpeechEventType, agents_vad.VADEventType


class SpeculativeSTT(stt.STT):
    def __init__(self, *, stt: stt.STT, vad: agents_vad.VAD, pause: float = 0.1) -> None:
        """
        Args:
            stt: a non-streaming STT, or a streaming one with ``capabilities.manual_flush``.
            vad: the VAD that finds the pauses (its own stream, as in LiveKit's StreamAdapter);
                usually the session's VAD object.
            pause: seconds of raw silence inside speech after which the transcript is requested.
        """
        caps = stt.capabilities
        if caps.streaming and not caps.manual_flush:
            raise ValueError("a streaming STT must support manual_flush: there is no other way to "
                             "make it finalize at the pause")
        super().__init__(capabilities=type(caps)(
            streaming=True, interim_results=caps.streaming and caps.interim_results, diarization=False,
            offline_recognize=caps.offline_recognize, keyterms=caps.keyterms, chat_context=caps.chat_context,
            manual_flush=caps.streaming))  # LiveKit's flush at END_OF_SPEECH is passed on
        self._wrapped, self._vad, self._pause = stt, vad, pause
        self._wrapped.on("metrics_collected", self._on_metrics)

    @property
    def wrapped_stt(self) -> stt.STT:
        return self._wrapped

    @property
    def model(self) -> str:
        return self._wrapped.model

    @property
    def provider(self) -> str:
        return self._wrapped.provider

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._wrapped._update_session_keyterms(keyterms)

    def _push_conversation_item(self, item: Any) -> None:
        self._wrapped._push_conversation_item(item)

    async def _recognize_impl(self, buffer: utils.AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> stt.SpeechEvent:
        return await self._wrapped.recognize(buffer=buffer, language=language, conn_options=conn_options)

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN,
               conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "SpeculativeStream":
        return SpeculativeStream(self, language=language, conn_options=conn_options)

    def prewarm(self) -> None:
        self._wrapped.prewarm()

    def _on_metrics(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def aclose(self) -> None:
        self._wrapped.off("metrics_collected", self._on_metrics)


class _Guess:
    """One transcript requested at a pause."""

    def __init__(self) -> None:
        self.task: Optional["asyncio.Task[str]"] = None
        self.stale = False  # the user spoke on after the pause


class SpeculativeStream(stt.RecognizeStream):
    def __init__(self, parent: SpeculativeSTT, *, language: NotGivenOr[str], conn_options: APIConnectOptions) -> None:
        # the wrapped STT retries on its own (as in LiveKit's StreamAdapter)
        super().__init__(stt=parent, conn_options=APIConnectOptions(max_retry=0, timeout=conn_options.timeout))
        self._parent, self._language, self._wrapped_conn = parent, language, conn_options

    async def _metrics_monitor_task(self, event_aiter: Any) -> None:
        async for _ in event_aiter:  # the wrapped STT reports its own metrics
            pass

    async def _run(self) -> None:
        wrapped = self._parent._wrapped
        vad_stream = self._parent._vad.stream()
        inner = (wrapped.stream(language=self._language, conn_options=self._wrapped_conn)
                 if wrapped.capabilities.streaming else None)

        async def forward_input() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    if inner is not None:
                        inner.flush()  # LiveKit's own flush, at END_OF_SPEECH
                    else:
                        vad_stream.flush()
                    continue
                vad_stream.push_frame(item)
                if inner is not None:
                    inner.push_frame(item)
            vad_stream.end_input()
            if inner is not None:
                inner.end_input()

        async def forward_events() -> None:
            async for ev in inner:
                self._event_ch.send_nowait(ev)

        tasks = [asyncio.create_task(forward_input(), name="forward_input"),
                 asyncio.create_task(self._flush_at_pauses(vad_stream, inner) if inner is not None
                                     else self._recognize_at_pauses(vad_stream), name="pauses")]
        if inner is not None:
            tasks.append(asyncio.create_task(forward_events(), name="forward_events"))
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()
            if inner is not None:
                await inner.aclose()

    def _pauses(self, ev: agents_vad.VADEvent, state: dict) -> Optional[str]:
        """"pause" when the raw silence of a pause first reaches the limit, "resume" when speech goes on."""
        if ev.type != V.INFERENCE_DONE or not ev.speaking:
            return None
        if ev.raw_accumulated_silence >= self._parent._pause:
            if not state.get("told"):
                state["told"] = True
                return "pause"
        elif ev.raw_accumulated_silence == 0 and state.get("told"):
            state["told"] = False
            return "resume"
        return None

    async def _flush_at_pauses(self, vad_stream: agents_vad.VADStream, inner: stt.RecognizeStream) -> None:
        state: dict = {}
        async for ev in vad_stream:
            if ev.type == V.START_OF_SPEECH:
                state.clear()
            elif self._pauses(ev, state) == "pause":
                inner.flush()

    async def _recognize(self, frames: list[rtc.AudioFrame]) -> str:
        event = await self._parent._wrapped.recognize(buffer=utils.merge_frames(frames), language=self._language,
                                                      conn_options=self._wrapped_conn)
        self._mark_provider_progress()
        return event.alternatives[0].text if event.alternatives else ""

    def _send(self, kind: stt.SpeechEventType, text: str, speech_end_time: Optional[float] = None) -> None:
        language = self._language if isinstance(self._language, str) else ""
        self._event_ch.send_nowait(stt.SpeechEvent(
            type=kind, alternatives=[stt.SpeechData(language=language, text=text)], speech_end_time=speech_end_time))

    async def _recognize_at_pauses(self, vad_stream: agents_vad.VADStream) -> None:
        state: dict = {}
        audio: list[rtc.AudioFrame] = []  # the running segment so far
        started_at = 0.0  # stream time of its START_OF_SPEECH
        guess: Optional[_Guess] = None
        finals: Optional[asyncio.Task] = None  # final transcripts go out in the order of their segments

        async def guessed(g: _Guess, frames: list[rtc.AudioFrame]) -> str:
            text = await self._recognize(frames)
            if not g.stale and text:
                self._send(T.PREFLIGHT_TRANSCRIPT, text)
            return text

        async def final(previous: Optional[asyncio.Task], g: Optional[_Guess], frames: list[rtc.AudioFrame],
                        speech_end_time: float) -> None:
            if previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
            text = None
            if g is not None and not g.stale:
                try:
                    text = await g.task  # what was heard at the pause is all there was
                except Exception:
                    logger.warning("speculative transcription failed, transcribing the segment again", exc_info=True)
            if text is None:
                text = await self._recognize(frames)
            if text:
                self._send(T.FINAL_TRANSCRIPT, text, speech_end_time)

        try:
            async for ev in vad_stream:
                if ev.type == V.START_OF_SPEECH:
                    state.clear()
                    audio, started_at, guess = list(ev.frames), ev.timestamp, None
                    self._event_ch.send_nowait(stt.SpeechEvent(T.START_OF_SPEECH))
                elif ev.type == V.END_OF_SPEECH:
                    end_time = time.time() - ev.silence_duration - ev.inference_duration
                    self._event_ch.send_nowait(stt.SpeechEvent(type=T.END_OF_SPEECH, speech_end_time=end_time))
                    finals = asyncio.create_task(final(finals, guess, list(ev.frames), end_time))
                    audio, guess = [], None
                elif ev.speaking:
                    if ev.timestamp > started_at + 1e-6:  # START_OF_SPEECH already held this frame's audio
                        audio.extend(ev.frames)
                    what = self._pauses(ev, state)
                    if what == "pause":
                        if guess is not None:
                            guess.stale = True
                        guess = _Guess()
                        guess.task = asyncio.create_task(guessed(guess, list(audio)))
                        guess.task.add_done_callback(lambda t: t.cancelled() or t.exception())  # may go unused
                    elif what == "resume" and guess is not None:
                        guess.stale = True
            if finals is not None:
                await finals
        finally:
            if finals is not None:
                await utils.aio.cancel_and_wait(finals)
