"""The mapping of RealtimeVAD's events to LiveKit's VADEvents, with a scripted probability per
32 ms frame and constant-level audio (as in RealtimeVAD's own tests/test_vad.py)."""

import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.vad import VADEventType as T
from livekit.plugins import realtimevad

F = 512 / 16000
SPEECH = [(2, 0.95, 0.1), (8, 0.9, 0.1)]  # frames 0-9: confident speech at -20 dBFS
LAGGING = [(24, 0.6, 0.0)]  # the audio is silent, Silero still says "voiced"
LEAD = [(5, 0.1, 0.0)]  # idle frames first; 15 frames are 48 quiet-clock blocks exactly
ON = 5 * F  # the speech begins
T_QUIET = 15 * F + 0.01  # end of the first quiet 10 ms block after it
# most tests here describe the early end: 0.25 s of quiet audio ends the speech
EARLY = {"min_silence_duration": 0.25, "early_end": True}
DEFAULTS = {"min_silence_duration": 0.55, "early_end": False}  # what VAD.load() has


def scripted(probs):
    it = iter(probs)
    return lambda frame: next(it, 0.0)


def tone(segments):
    """A 1 kHz tone with RMS ``amp`` per segment: every 10 ms block and every 32 ms frame has exactly
    that level, and there is no DC part (a constant value is an offset, which a VAD may ignore)."""
    amps = np.concatenate([np.full(n * 512, float(amp)) for n, _, amp in segments])
    wave = np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(len(amps)) / 16000)
    return np.round(amps * wave * 32767).astype(np.int16)


async def run(segments, rate=16000, frame_ms=20, flush_at=None, at=(), vad=None, **load):
    """``at``: (seconds, function(vad)) pairs, each called once the audio before that time is through.
    ``vad``: open the stream on this VAD (with a fresh probability script) instead of loading one."""
    probs = [p for n, p, _ in segments for _ in range(n)]
    pcm = tone(segments)
    pcm = np.concatenate((pcm, np.zeros(-len(pcm) % (16 * frame_ms), dtype=np.int16)))  # whole push frames
    if rate != 16000:
        pcm = np.repeat(pcm, rate // 16000)[: len(pcm) * rate // 16000] if rate % 16000 == 0 else np.interp(
            np.arange(int(len(pcm) * rate / 16000)) * 16000 / rate, np.arange(len(pcm)), pcm).astype(np.int16)
    if vad is None:
        vad = realtimevad.VAD.load(probability=scripted(probs), **{**EARLY, **load})
    else:
        vad._opts.core["probability"] = scripted(probs)
    seen, todo = [], sorted(at, key=lambda x: x[0])
    vad.on("pause", lambda ev: seen.append(ev))
    stream, step = vad.stream(), rate * frame_ms // 1000
    for i in range(0, len(pcm) - step + 1, step):
        if flush_at is not None and i <= flush_at * rate < i + step:
            stream.flush()
        while todo and todo[0][0] * rate < i + step:
            while i and (getattr(stream, "_pushed", 0) < i or stream._fed < i * 16000 // rate):
                await asyncio.sleep(0.002)
            await asyncio.sleep(0.02)  # the last 10 ms slice is through the core
            todo.pop(0)[1](vad)
        stream.push_frame(rtc.AudioFrame(data=pcm[i : i + step].tobytes(), sample_rate=rate, num_channels=1,
                                         samples_per_channel=step))
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    return events, seen, vad


def only(events, kind):
    return [e for e in events if e.type == kind]


def seconds(event):
    return sum(f.samples_per_channel / f.sample_rate for f in event.frames)


async def test_start_and_end_report_when_the_speech_really_began_and_stopped():
    events, _, vad = await run(LEAD + SPEECH + LAGGING)
    assert [e.type for e in events if e.type != T.INFERENCE_DONE] == [T.START_OF_SPEECH, T.END_OF_SPEECH]
    start, end = only(events, T.START_OF_SPEECH)[0], only(events, T.END_OF_SPEECH)[0]
    assert start.timestamp == pytest.approx(ON + 2 * F)  # two confident frames
    assert start.timestamp - start.speech_duration == pytest.approx(ON)  # back-dates to the onset
    assert start.speaking and seconds(start) == pytest.approx(ON + 2 * F)  # padding: all there is before it
    # the audio went quiet 10 ms after the last speech frame; the end comes min_silence_duration later
    assert end.timestamp == pytest.approx(T_QUIET + 0.25, abs=1e-6)
    assert end.timestamp - end.silence_duration == pytest.approx(T_QUIET, abs=1e-6)
    assert end.speech_duration == pytest.approx(T_QUIET - ON, abs=1e-6)
    assert not end.speaking and seconds(end) == pytest.approx(end.timestamp, abs=0.002)  # padding + speech + silence
    assert vad.min_silence_duration == 0.25 and vad.model == "realtimevad"


async def test_inference_done_once_per_frame_with_that_frames_audio():
    segments = LEAD + SPEECH + LAGGING
    events, _, _ = await run(segments)
    done = only(events, T.INFERENCE_DONE)
    assert len(done) == sum(n for n, _, _ in segments)
    assert all(sum(f.samples_per_channel for f in e.frames) == 512 for e in done)  # no overlap, no gap
    assert [e.timestamp for e in done] == pytest.approx([(i + 1) * F for i in range(len(done))])
    assert [e.probability for e in done[:7]] == pytest.approx([0.1] * 5 + [0.95] * 2)


async def test_quiet_clock_runs_into_raw_accumulated_silence_before_silero_notices():
    events, pauses, _ = await run(LEAD + SPEECH + LAGGING)
    done = only(events, T.INFERENCE_DONE)
    speaking = [e for e in done if e.speaking]
    assert speaking[3].raw_accumulated_speech > 0 and speaking[3].raw_accumulated_silence == 0
    # Silero's p stays 0.6 ("voiced") to the end: without the quiet clock LiveKit would see no silence
    after = [e for e in speaking if e.timestamp > T_QUIET]
    assert all(e.raw_accumulated_speech == 0 for e in after)
    assert after[0].raw_accumulated_silence == pytest.approx(after[0].timestamp - T_QUIET + 0.01, abs=0.011)
    # LiveKit starts its turn detector at 0.2 s of raw silence: reached before END, 0.2 s after the stop
    trigger = next(e.timestamp for e in speaking if e.raw_accumulated_silence >= 0.2)
    end = only(events, T.END_OF_SPEECH)[0]
    assert trigger < end.timestamp and trigger - T_QUIET == pytest.approx(0.2, abs=0.033)
    assert [p.source for p in pauses] == ["quiet"] and pauses[0].timestamp == pytest.approx(T_QUIET)
    off = await run(LEAD + SPEECH + LAGGING, quiet_rms=0)
    assert not only(off[0], T.END_OF_SPEECH)  # quiet clock off: Silero alone never ends this


async def test_speech_duration_drives_interruptions_and_a_short_sound_stays_short():
    # LiveKit interrupts the agent when an INFERENCE_DONE has speech_duration >= 0.5 s
    long, _, _ = await run(LEAD + [(2, 0.95, 0.1), (30, 0.9, 0.1)])
    reached = next(e.timestamp for e in only(long, T.INFERENCE_DONE) if e.speech_duration >= 0.5)
    assert reached - ON == pytest.approx(0.5, abs=0.033)  # 0.5 s after the onset
    # 0.13 s of sound, then silence: the segment is over 0.25 s later, before 0.5 s is reached
    short, _, _ = await run(LEAD + [(2, 0.95, 0.1), (2, 0.9, 0.1), (30, 0.1, 0.0)])
    assert only(short, T.END_OF_SPEECH)
    assert max(e.speech_duration for e in only(short, T.INFERENCE_DONE)) < 0.5
    assert all(e.speech_duration == 0 for e in only(short, T.INFERENCE_DONE) if not e.speaking)


async def test_start_counts_from_where_silero_first_passed_half():
    # Silero climbs 0.6, 0.7 before its first confident frame: the speech began there, so
    # speech_duration (which LiveKit interrupts on at 0.5 s) counts from there, as with Silero at 0.5.
    events, _, _ = await run(LEAD + [(1, 0.6, 0.1), (1, 0.7, 0.1), (2, 0.95, 0.1), (20, 0.9, 0.1)])
    start = only(events, T.START_OF_SPEECH)[0]
    assert start.timestamp == pytest.approx(ON + 4 * F)
    assert start.timestamp - start.speech_duration == pytest.approx(ON)
    reached = next(e.timestamp for e in only(events, T.INFERENCE_DONE) if e.speech_duration >= 0.5)
    assert reached - ON == pytest.approx(0.5, abs=0.033)


async def test_flush_is_a_hard_boundary():
    events, _, _ = await run(LEAD + [(2, 0.95, 0.1), (60, 0.9, 0.1)], flush_at=1.0)
    assert not only(events, T.END_OF_SPEECH)  # the running speech is dropped, not ended
    first, second = only(events, T.START_OF_SPEECH)  # the speech that goes on starts a fresh segment
    assert first.timestamp == pytest.approx(ON + 2 * F)
    assert second.timestamp == pytest.approx(2 * F, abs=0.033)  # time begins again at the flush
    assert seconds(second) <= second.timestamp + 0.021  # and no audio from before it


async def test_other_sample_rates_are_resampled_and_frames_stay_at_the_input_rate():
    segments = LEAD + SPEECH + LAGGING
    native, _, _ = await run(segments)
    other, _, _ = await run(segments, rate=48000, frame_ms=50)  # the resampler has a few ms of delay

    def marks(events):
        return [(e.type, e.timestamp) for e in events if e.type != T.INFERENCE_DONE]

    assert [k for k, _ in marks(other)] == [k for k, _ in marks(native)]
    assert [t for _, t in marks(other)] == pytest.approx([t for _, t in marks(native)], abs=0.04)
    end = only(other, T.END_OF_SPEECH)[0]
    assert {f.sample_rate for f in end.frames} == {48000}
    assert seconds(end) == pytest.approx(end.timestamp, abs=0.01)


async def test_min_silence_duration_and_core_options_are_passed_on():
    events, _, vad = await run(LEAD + SPEECH + [(40, 0.6, 0.0)], min_silence_duration=0.55)
    end = only(events, T.END_OF_SPEECH)[0]
    assert vad.min_silence_duration == 0.55 and end.silence_duration == pytest.approx(0.55, abs=1e-6)
    soft, _, _ = await run(LEAD + [(20, 0.6, 0.1)])  # never confident: the soft start takes it
    assert len(only(soft, T.START_OF_SPEECH)) == 1
    assert not only((await run(LEAD + [(20, 0.6, 0.1)], soft_start_s=None))[0], T.START_OF_SPEECH)


SILENT = 0.1, 0.0  # Silero says "silence" and the audio is silent
QUIET_OFF = {"quiet_rms": 0}  # the quiet clock off: Silero's frame count alone ends the speech


def marks(events):
    return [(e.type, e.timestamp) for e in events if e.type != T.INFERENCE_DONE]


async def test_update_options_changes_the_silence_needed_on_a_running_stream():
    again = LEAD + SPEECH + LAGGING + SPEECH + [(40, 0.6, 0.0)]  # two utterances, 34 frames apart
    second_quiet = 1.58  # the speech stops at 49 frames = 1.568 s; the first whole quiet 10 ms block ends here
    events, _, vad = await run(again, at=[(1.3, lambda v: v.update_options(min_silence_duration=0.55))])
    first, second = only(events, T.END_OF_SPEECH)
    assert first.silence_duration == pytest.approx(0.25, abs=1e-6)  # ended before the change
    assert second.silence_duration == pytest.approx(0.55, abs=1e-6)
    assert second.timestamp - second.silence_duration == pytest.approx(second_quiet, abs=1e-6)
    assert vad.min_silence_duration == 0.55  # what LiveKit reads
    # Silero's own count follows too: 8 frames for 0.25 s, 18 frames for 0.55 s
    again = LEAD + SPEECH + [(24, *SILENT)] + SPEECH + [(40, *SILENT)]
    events, _, _ = await run(again, at=[(1.3, lambda v: v.update_options(min_silence_duration=0.55))], **QUIET_OFF)
    first, second = only(events, T.END_OF_SPEECH)
    assert first.silence_duration == pytest.approx(8 * F) and second.silence_duration == pytest.approx(18 * F)
    # and back down while the stream runs
    events, _, _ = await run(again, min_silence_duration=0.55,
                             at=[(1.3, lambda v: v.update_options(min_silence_duration=0.25))], **QUIET_OFF)
    first, second = only(events, T.END_OF_SPEECH)
    assert first.silence_duration == pytest.approx(18 * F) and second.silence_duration == pytest.approx(8 * F)


async def test_update_options_without_a_change_changes_nothing():
    segments = LEAD + SPEECH + LAGGING

    def fields(events):
        return [(e.type, e.timestamp, e.speech_duration, e.silence_duration, e.raw_accumulated_silence)
                for e in events]

    plain, _, _ = await run(segments)
    touched, _, _ = await run(segments, at=[(0.3, lambda v: v.update_options(max_buffered_speech=30.0)),
                                            (0.6, lambda v: v.update_options(min_silence_duration=0.25))])
    assert fields(touched) == fields(plain)


async def test_update_options_thresholds_and_min_speech_duration():
    unsure = LEAD + [(10, 0.8, 0.1), (10, *SILENT)] + [(10, 0.8, 0.1), (10, *SILENT)]  # never reaches 0.9
    events, _, _ = await run(unsure, soft_start_s=None,
                             at=[(0.6, lambda v: v.update_options(activation_threshold=0.7))])
    start, = only(events, T.START_OF_SPEECH)  # the first one was not heard, the second is
    assert start.timestamp == pytest.approx(25 * F + 2 * F)
    # deactivation_threshold: frames at 0.4 are silence at 0.45 and speech at 0.35
    fading = LEAD + SPEECH + [(12, 0.4, 0.1)] + [(12, *SILENT)]
    assert only((await run(fading, **QUIET_OFF))[0], T.END_OF_SPEECH)[0].timestamp == pytest.approx(23 * F)
    events, _, _ = await run(fading, at=[(0.3, lambda v: v.update_options(deactivation_threshold=0.35))], **QUIET_OFF)
    assert only(events, T.END_OF_SPEECH)[0].timestamp == pytest.approx(35 * F)
    # min_speech_duration: five confirming frames instead of two
    events, _, _ = await run(LEAD + SPEECH + LAGGING, at=[(0.1, lambda v: v.update_options(min_speech_duration=0.16))])
    start = only(events, T.START_OF_SPEECH)[0]
    assert start.timestamp == pytest.approx(ON + 5 * F) and start.timestamp - start.speech_duration == pytest.approx(ON)


async def test_update_options_prefix_padding_and_streams_opened_later():
    long_lead = [(40, 0.1, 0.0)] + SPEECH + LAGGING
    assert seconds(only((await run(long_lead))[0], T.START_OF_SPEECH)[0]) == pytest.approx(0.5 + 2 * F, abs=0.001)
    events, _, vad = await run(long_lead, at=[(0.5, lambda v: v.update_options(prefix_padding_duration=0.1))])
    start, end = only(events, T.START_OF_SPEECH)[0], only(events, T.END_OF_SPEECH)[0]
    assert seconds(start) == pytest.approx(0.1 + 2 * F, abs=0.001)
    assert seconds(end) == pytest.approx(0.1 + end.timestamp - 40 * F, abs=0.001)
    # a stream opened afterwards starts with the changed options
    vad.update_options(min_silence_duration=0.55, prefix_padding_duration=0.2)
    events, _, _ = await run(long_lead + [(20, 0.6, 0.0)], vad=vad)
    start, end = only(events, T.START_OF_SPEECH)[0], only(events, T.END_OF_SPEECH)[0]
    assert seconds(start) == pytest.approx(0.2 + 2 * F, abs=0.001)
    assert end.silence_duration == pytest.approx(0.55, abs=1e-6)


async def test_update_options_leaves_what_was_set_under_realtimevads_own_name():
    events, _, _ = await run(LEAD + SPEECH + [(40, 0.6, 0.0)], end_s=0.4,
                             at=[(0.1, lambda v: v.update_options(min_silence_duration=0.55))])
    assert only(events, T.END_OF_SPEECH)[0].silence_duration == pytest.approx(0.4, abs=1e-6)


async def test_agent_speaking_reaches_the_core_as_playback(monkeypatch):
    from livekit.plugins.realtimevad import vad as module

    fed, original = [], module._Core.feed

    def feed(self, audio, playback=False):
        fed.append((len(audio), playback))
        return original(self, audio, playback=playback)

    monkeypatch.setattr(module._Core, "feed", feed)
    segments = LEAD + SPEECH + LAGGING + [(20, *SILENT)]
    plain, _, vad = await run(segments)
    assert not vad.agent_speaking and not any(playback for _, playback in fed)
    del fed[:]
    # the agent speaks from 0.1 s to 1.0 s; the user starts at 0.16 s and is heard all the same
    events, _, vad = await run(segments, at=[(0.1, lambda v: v.set_agent_speaking(True)),
                                             (1.0, lambda v: v.set_agent_speaking(False))])
    flags = [playback for n, playback in fed for _ in range(n)]  # per 16 kHz sample
    assert not any(flags[: int(0.08 * 16000)]) and all(flags[int(0.12 * 16000) : int(0.98 * 16000)])
    assert not any(flags[int(1.02 * 16000) :]) and not vad.agent_speaking
    assert marks(events) == marks(plain)
    # every stream of the VAD gets it (LiveKit's StreamAdapter opens a second one)
    del fed[:]
    vad.set_agent_speaking(True)
    await run([(10, *SILENT)], vad=vad)
    assert fed and all(playback for _, playback in fed)


# -- the default: END_OF_SPEECH where the Silero plugin puts it ------------------------
GONE = 0.1, 0.0  # Silero's probability falls with the audio


async def test_by_default_the_end_waits_for_the_silero_plugins_count_and_the_pause_is_early():
    events, _, _ = await run(LEAD + SPEECH + [(30, *GONE)], **DEFAULTS)
    end = only(events, T.END_OF_SPEECH)[0]
    # the Silero plugin smooths the probability (0.9 -> 0.38 -> 0.2): its count of 18 frames (0.55 s)
    # begins one frame after the raw probability fell, so it is full with the 19th silent frame
    assert end.timestamp == pytest.approx((15 + 19) * F)
    quiet = [e for e in only(events, T.INFERENCE_DONE) if e.speaking and e.raw_accumulated_silence >= 0.1]
    assert quiet[0].timestamp < end.timestamp - 0.4  # the pause is there long before the end


async def test_a_gap_the_silero_plugin_bridges_does_not_end_the_segment():
    # 18 silent frames are 0.576 s: the raw count is full, the Silero plugin's (one frame behind) is not
    events, _, _ = await run(LEAD + SPEECH + [(18, *GONE)] + SPEECH + [(30, *GONE)], **DEFAULTS)
    assert [e.type for e in events if e.type != T.INFERENCE_DONE] == [T.START_OF_SPEECH, T.END_OF_SPEECH]
    with_early_end, _, _ = await run(LEAD + SPEECH + [(18, *GONE)] + SPEECH + [(30, *GONE)],
                                     min_silence_duration=0.55)
    assert len(only(with_early_end, T.END_OF_SPEECH)) == 2


async def test_an_extend_callback_of_the_application_still_holds_the_end():
    events, _, _ = await run(LEAD + SPEECH + [(40, *GONE)], extend=lambda t: 0.8, **DEFAULTS)
    assert only(events, T.END_OF_SPEECH)[0].timestamp == pytest.approx((15 + 25) * F)  # 0.8 s are 25 frames
