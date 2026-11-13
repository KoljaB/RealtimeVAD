"""SpeculativeSTT: the transcript is requested at the pause, with scripted probabilities as in test_plugin."""

import asyncio
import inspect

import pytest
from livekit import rtc
from livekit.agents import stt
from livekit.agents.stt import SpeechEventType as S
from livekit.plugins import realtimevad

from test_plugin import EARLY, LAGGING, LEAD, SPEECH, T_QUIET, F, scripted, tone

ON = 5 * F


class Whisperish(stt.STT):
    """Not streaming: answers "text 1", "text 2", ... and notes how much audio each request held."""

    def __init__(self, fail_first=False, takes=0.01):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.requests, self._fail_first, self._takes, self.cancelled = [], fail_first, takes, 0

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        frame = rtc.combine_audio_frames(buffer) if isinstance(buffer, list) else buffer
        self.requests.append(frame.samples_per_channel / frame.sample_rate)
        n = len(self.requests)
        try:
            await asyncio.sleep(self._takes)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self._fail_first and n == 1:
            raise RuntimeError("no luck")
        return stt.SpeechEvent(type=S.FINAL_TRANSCRIPT, alternatives=[stt.SpeechData(language="en", text=f"text {n}")])


class Streaming(stt.STT):
    """Streaming with manual_flush: every flush finalizes "flushed N"."""

    def __init__(self, manual_flush=True):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True,
                                                          offline_recognize=False, manual_flush=manual_flush))
        self.flushes = []  # audio received when each flush came

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        raise NotImplementedError

    def stream(self, *, language=None, conn_options=None):
        return StreamingStream(stt=self, conn_options=conn_options)


class StreamingStream(stt.RecognizeStream):
    async def _run(self):
        pushed = 0.0
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._stt.flushes.append(pushed)
                self._event_ch.send_nowait(stt.SpeechEvent(type=S.FINAL_TRANSCRIPT, alternatives=[
                    stt.SpeechData(language="en", text=f"flushed {len(self._stt.flushes)}")]))
            else:
                pushed += item.samples_per_channel / item.sample_rate


async def run(segments, wrapped, flush_at_end=False, on_pause=None, vad_options=EARLY, **options):  # the VAD ends early: short test audio
    probs = [p for n, p, _ in segments for _ in range(n)]
    vad = realtimevad.VAD.load(probability=scripted(probs), **vad_options)
    if on_pause is not None:
        vad.on("pause", on_pause)
    speculative = realtimevad.SpeculativeSTT(stt=wrapped, vad=vad, **options)
    pcm = tone(segments)
    stream = speculative.stream()
    for i in range(0, len(pcm) - 319, 320):
        stream.push_frame(rtc.AudioFrame(data=pcm[i : i + 320].tobytes(), sample_rate=16000, num_channels=1,
                                         samples_per_channel=320))
        await asyncio.sleep(0.004)  # frame by frame, so the VAD keeps up as it does with live audio
    await asyncio.sleep(0.3)
    if flush_at_end:
        stream.flush()
    stream.end_input()
    events = [(ev.type, ev.alternatives[0].text if ev.alternatives else None) async for ev in stream]
    await stream.aclose()
    return events


async def test_a_pause_that_is_the_end_costs_one_request_and_the_final_repeats_the_preflight():
    wrapped = Whisperish()
    events = await run(LEAD + SPEECH + LAGGING, wrapped)
    assert events == [(S.START_OF_SPEECH, None), (S.PREFLIGHT_TRANSCRIPT, "text 1"), (S.END_OF_SPEECH, None),
                      (S.FINAL_TRANSCRIPT, "text 1")]
    # the request held the padding and the speech up to 0.1 s after the audio went quiet
    assert wrapped.requests == [pytest.approx(T_QUIET + 0.1, abs=0.04)]


async def test_speech_that_goes_on_after_the_pause_drops_that_transcript():
    gap = [(5, 0.6, 0.0)]  # 0.16 s quiet inside the sentence: a pause, not an end
    wrapped = Whisperish()
    events = await run(LEAD + SPEECH + gap + SPEECH + LAGGING, wrapped)
    assert len(wrapped.requests) == 2 and wrapped.requests[1] > wrapped.requests[0] + 0.4  # the second holds it all
    assert [e for e in events if e[0] != S.PREFLIGHT_TRANSCRIPT] == [
        (S.START_OF_SPEECH, None), (S.END_OF_SPEECH, None), (S.FINAL_TRANSCRIPT, "text 2")]
    assert [e for e in events if e[0] == S.PREFLIGHT_TRANSCRIPT][-1] == (S.PREFLIGHT_TRANSCRIPT, "text 2")
    assert events.index((S.PREFLIGHT_TRANSCRIPT, "text 2")) < events.index((S.FINAL_TRANSCRIPT, "text 2"))


async def test_without_a_pause_before_the_end_it_is_a_plain_stream_adapter():
    wrapped = Whisperish()
    events = await run(LEAD + SPEECH + LAGGING, wrapped, pause=1.0)
    assert events == [(S.START_OF_SPEECH, None), (S.END_OF_SPEECH, None), (S.FINAL_TRANSCRIPT, "text 1")]
    assert wrapped.requests == [pytest.approx(T_QUIET + 0.25, abs=0.04)]  # the whole segment, as StreamAdapter sends


async def test_a_failed_speculative_request_is_made_again_at_the_end():
    wrapped = Whisperish(fail_first=True)
    events = await run(LEAD + SPEECH + LAGGING, wrapped)
    assert events == [(S.START_OF_SPEECH, None), (S.END_OF_SPEECH, None), (S.FINAL_TRANSCRIPT, "text 2")]


async def test_a_streaming_stt_is_flushed_at_the_pause():
    segments = LEAD + SPEECH + LAGGING
    plain = Streaming()
    await run(segments, plain, pause=1.0)  # no pause that long: only closing the streams flushes
    wrapped = Streaming()
    events = await run(segments, wrapped)
    assert len(wrapped.flushes) == len(plain.flushes) + 1
    assert events[0] == (S.FINAL_TRANSCRIPT, "flushed 1") and len(events) == len(wrapped.flushes)  # passed on
    # LiveKit's own flush (at END_OF_SPEECH) is passed on as well
    both = Streaming()
    await run(segments, both, flush_at_end=True)
    assert len(both.flushes) == len(wrapped.flushes) + 1
    with pytest.raises(ValueError):
        realtimevad.SpeculativeSTT(stt=Streaming(manual_flush=False), vad=realtimevad.VAD.load())


GAP = [(5, 0.6, 0.0)]  # 0.16 s quiet inside the sentence


async def test_a_request_that_still_runs_when_speech_goes_on_is_cancelled():
    wrapped = Whisperish(takes=0.2)  # longer than the gap (the test feeds 5 times faster than real time)
    events = await run(LEAD + SPEECH + GAP + SPEECH + LAGGING, wrapped)
    assert wrapped.cancelled == 1 and len(wrapped.requests) == 2
    assert [e for e in events if e[0] in (S.PREFLIGHT_TRANSCRIPT, S.FINAL_TRANSCRIPT)] == [
        (S.PREFLIGHT_TRANSCRIPT, "text 2"), (S.FINAL_TRANSCRIPT, "text 2")]


async def test_min_new_speech_passes_over_pauses_after_too_little_speech():
    segments = LEAD + SPEECH + GAP + SPEECH + LAGGING  # 0.32 s of speech before each pause
    wrapped = Whisperish()
    events = await run(segments, wrapped, min_new_speech=0.5)  # the first pause is passed over
    assert len(wrapped.requests) == 1 and wrapped.requests[0] > 0.8  # asked at the second, with both parts
    assert [e for e in events if e[0] in (S.PREFLIGHT_TRANSCRIPT, S.FINAL_TRANSCRIPT)] == [
        (S.PREFLIGHT_TRANSCRIPT, "text 1"), (S.FINAL_TRANSCRIPT, "text 1")]
    wrapped = Whisperish()
    events = await run(segments, wrapped, min_new_speech=2.0)  # both passed over: a plain StreamAdapter
    assert [e for e in events if e[0] != S.START_OF_SPEECH] == [(S.END_OF_SPEECH, None), (S.FINAL_TRANSCRIPT, "text 1")]
    streaming = Streaming()
    await run(segments, streaming)
    limited = Streaming()
    await run(segments, limited, min_new_speech=0.5)
    assert len(limited.flushes) == len(streaming.flushes) - 1


async def test_a_repeated_pause_replaces_the_first_one():
    # the audio goes quiet, 64 ms of sound follow (Silero says "voiced" throughout), then it is quiet for
    # good: RealtimeVAD reports a pause, then a second one, and the second is the stop
    again = [(5, 0.6, 0.0), (2, 0.6, 0.1)]
    segments = LEAD + SPEECH + again + LAGGING
    pauses = []
    wrapped = Whisperish(takes=0.2)
    events = await run(segments, wrapped, on_pause=pauses.append)
    assert len(pauses) == 2 and pauses[1].timestamp == pytest.approx(pauses[0].timestamp + 7 * F, abs=0.011)
    assert wrapped.cancelled == 1 and len(wrapped.requests) == 2  # the first request was dropped
    assert wrapped.requests[1] == pytest.approx(wrapped.requests[0] + 7 * F, abs=0.04)  # asked again 0.1 s after the second
    assert [e for e in events if e[0] in (S.PREFLIGHT_TRANSCRIPT, S.FINAL_TRANSCRIPT)] == [
        (S.PREFLIGHT_TRANSCRIPT, "text 2"), (S.FINAL_TRANSCRIPT, "text 2")]


REPAUSE = "repause_s" in inspect.signature(realtimevad.vad._Core.__init__).parameters


@pytest.mark.skipif(not REPAUSE, reason="this RealtimeVAD reports one pause per stretch of Silero silence")
async def test_a_repeated_pause_while_silero_says_silence_replaces_the_first_one_too():
    # Silero's probability falls with the audio; 64 ms of sound that it does not hear as voice follow.
    # Its silence count runs through, but the raw silence counts from the second pause.
    segments = LEAD + SPEECH + [(5, 0.1, 0.0), (2, 0.1, 0.1), (30, 0.1, 0.0)]
    pauses = []
    wrapped = Whisperish(takes=0.2)
    events = await run(segments, wrapped, on_pause=pauses.append, vad_options={})
    assert len(pauses) == 2
    assert wrapped.cancelled == 1 and len(wrapped.requests) == 2
    assert wrapped.requests[1] == pytest.approx(wrapped.requests[0] + 7 * F, abs=0.04)
    assert [e for e in events if e[0] in (S.PREFLIGHT_TRANSCRIPT, S.FINAL_TRANSCRIPT)] == [
        (S.PREFLIGHT_TRANSCRIPT, "text 2"), (S.FINAL_TRANSCRIPT, "text 2")]
