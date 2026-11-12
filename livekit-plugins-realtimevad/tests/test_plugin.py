"""The mapping of RealtimeVAD's events to LiveKit's VADEvents, with a scripted probability per
32 ms frame and constant-level audio (as in RealtimeVAD's own tests/test_vad.py)."""

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


def scripted(probs):
    it = iter(probs)
    return lambda frame: next(it, 0.0)


async def run(segments, rate=16000, frame_ms=20, flush_at=None, **load):
    probs = [p for n, p, _ in segments for _ in range(n)]
    pcm = np.concatenate([np.full(n * 512, int(amp * 32767), dtype=np.int16) for n, _, amp in segments])
    pcm = np.concatenate((pcm, np.zeros(-len(pcm) % (16 * frame_ms), dtype=np.int16)))  # whole push frames
    if rate != 16000:
        pcm = np.repeat(pcm, rate // 16000)[: len(pcm) * rate // 16000] if rate % 16000 == 0 else np.interp(
            np.arange(int(len(pcm) * rate / 16000)) * 16000 / rate, np.arange(len(pcm)), pcm).astype(np.int16)
    vad = realtimevad.VAD.load(probability=scripted(probs), **load)
    seen = []
    vad.on("pause", lambda ev: seen.append(ev))
    stream, step = vad.stream(), rate * frame_ms // 1000
    for i in range(0, len(pcm) - step + 1, step):
        if flush_at is not None and i <= flush_at * rate < i + step:
            stream.flush()
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
