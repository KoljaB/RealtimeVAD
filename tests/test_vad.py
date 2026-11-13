"""Spec checks with a scripted probability per 32 ms frame (see SPEC.md)."""

import math

import numpy as np
import pytest

from RealtimeVAD import VAD

FRAME = 512
F = FRAME / 16000


def level(n, amp):
    """n samples at RMS amp without a DC part (a constant value would be an offset, not sound)."""
    x = np.full(n, amp, dtype=np.float32)
    x[1::2] *= -1
    return x


def scripted(probs):
    it = iter(probs)
    return lambda frame: next(it, 0.0)


def run(segments, **kw):
    """segments: (frames, p, amplitude) -> constant-level audio, scripted p."""
    probs, audio = [], []
    for n, p, amp in segments:
        probs += [p] * n
        audio.append(level(n * FRAME, amp))
    kw.setdefault("early_end", True)  # most tests below are about the quiet clock's early end (off by default)
    kw.setdefault("end_s", 0.24)  # timings below assume 0.24 s on both clocks
    kw.setdefault("quiet_end_s", 0.24)
    vad = VAD(probability=scripted(probs), **kw)
    return vad.feed(np.concatenate(audio))


def kinds(events):
    return [e.kind for e in events]


SPEECH = [(2, 0.95, 0.1), (8, 0.9, 0.1)]  # frames 0-9


def test_start_needs_strong_then_continue_frame():
    ev = run([(1, 0.95, 0.1), (1, 0.5, 0.1), (5, 0.9, 0.1)])
    assert kinds(ev) == ["start"]
    assert ev[0].time == pytest.approx(F)


def test_single_strong_frame_does_not_start():
    assert run([(1, 0.95, 0.1), (1, 0.2, 0.1), (1, 0.95, 0.1), (1, 0.2, 0.1)]) == []


def test_silero_clock_ends_when_noise_keeps_quiet_clock_off():
    ev = run(SPEECH + [(12, 0.1, 0.05)])
    assert kinds(ev) == ["start", "pause", "end"]
    end = ev[-1]
    assert ev[1].time == pytest.approx(11 * F)  # first silent frame
    assert end.time == pytest.approx(18 * F)  # 8 silent frames
    assert end.segment.speech_end == pytest.approx(10 * F)


def test_quiet_clock_beats_lagging_silero():
    ev = run(SPEECH + [(20, 0.6, 0.0)])  # Silero still "voiced", audio silent
    assert kinds(ev) == ["start", "pause", "end"]
    t_quiet = (10 * FRAME + 160) / 16000
    assert ev[1].time == pytest.approx(t_quiet)
    assert ev[2].time == pytest.approx(t_quiet + 0.24)
    assert ev[2].segment.speech_end == pytest.approx(t_quiet)


def test_by_default_the_end_is_sileros_and_the_quiet_clock_gives_pause_and_speech_end():
    # Audio silent after the word, Silero lags 4 frames: the pause comes with the first quiet
    # block, the end when Silero has counted its 0.24 s, and the segment says when the speech
    # really stopped. early_end=True ends after 0.24 s of quiet audio instead.
    segs = SPEECH + [(4, 0.6, 0.0), (12, 0.1, 0.0)]
    t_quiet = (10 * FRAME + 160) / 16000
    ev = run(segs, early_end=False)
    assert [(e.kind, e.source) for e in ev] == [("start", ""), ("pause", "quiet"), ("end", "")]
    assert ev[1].time == pytest.approx(t_quiet)
    assert ev[2].time == pytest.approx(22 * F)  # 14 voiced frames + 8 silent ones
    assert ev[2].segment.speech_end == pytest.approx(t_quiet) and not ev[2].segment.early
    assert [e.time for e in VAD(probability=scripted([p for n, p, _ in segs for _ in range(n)])).feed(
        np.concatenate([level(n * FRAME, a) for n, _, a in segs]))] == [e.time for e in ev]  # the default
    early = run(segs, early_end=True)
    assert early[2].time == pytest.approx(t_quiet + 0.24) and early[2].segment.early


def test_by_default_no_end_in_a_pause_silero_rides_through():
    # 0.3 s of silence inside an utterance; Silero stays above continue_threshold through it.
    # The quiet clock alone would end the segment there (early_end=True); the default does not.
    segs = SPEECH + [(10, 0.6, 0.0), (10, 0.9, 0.1)]
    assert kinds(run(segs, early_end=False)) == ["start", "pause"]
    assert kinds(run(segs, early_end=True)) == ["start", "pause", "end", "start"]


def test_without_quiet_audio_speech_end_is_sileros_last_voiced_frame():
    ev = run(SPEECH + [(12, 0.1, 0.05)], early_end=False)  # noise after the word: never quiet
    assert kinds(ev) == ["start", "pause", "end"] and ev[1].source == "silero"
    assert ev[2].segment.speech_end == pytest.approx(10 * F)


def word_then(tail, **kw):
    """A word (frames 0-9), then silence with `tail` = [(sample offset after the word, samples, amp)]."""
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 4 + [0.1] * 12
    audio = np.zeros(26 * FRAME, dtype=np.float32)
    audio[: 10 * FRAME] = level(10 * FRAME, 0.1)
    for at, n, amp in tail:
        audio[10 * FRAME + at : 10 * FRAME + at + n] = level(n, amp)
    kw = {"early_end": True, "end_s": 0.24, "quiet_end_s": 0.24, **kw}
    return VAD(probability=scripted(probs), **kw).feed(audio)


def test_click_restarts_the_quiet_count():
    # A click (one loud 10 ms block, too short to be sound) inside the quiet after a word: the
    # count starts again at the next quiet block, without a second pause event. It used to
    # stop for good there, which left a third of all real turn ends to Silero.
    ev = word_then([(1120, 160, 0.1)])  # exactly one 10 ms block
    assert kinds(ev) == ["start", "pause", "end"]
    again = (10 * FRAME + 1120 + 160 + 160) / 16000  # the first quiet block after it ends here
    assert ev[-1].segment.speech_end == pytest.approx(again)
    assert ev[-1].time == pytest.approx(again + 0.24)  # Silero alone: 22 frames = 0.704 s
    assert ev[-1].time < 22 * F


def test_sound_after_a_pause_gives_a_new_pause():
    # The level dips under the threshold inside the last word and comes back for 30 ms (a
    # release, a weak last syllable): too short to re-arm the quiet clock, but the quiet after
    # it is reported as a new pause, so the latest pause is where the speech really stopped.
    ev = word_then([(320, 480, 0.02)])
    assert kinds(ev) == ["start", "pause", "pause", "end"]
    first, again = (10 * FRAME + 160) / 16000, (10 * FRAME + 320 + 480 + 160) / 16000
    assert ev[1].time == pytest.approx(first) and ev[2].time == pytest.approx(again)
    assert ev[2].source == "quiet"
    assert ev[-1].segment.speech_end == pytest.approx(again)
    # 20 ms are needed: one block is a click, not sound
    assert kinds(word_then([(320, 160, 0.02)])) == ["start", "pause", "end"]
    assert kinds(word_then([(320, 480, 0.02)], repause_s=None)) == ["start", "pause", "end"]
    assert kinds(word_then([(320, 480, 0.02)], repause_s=0.04)) == ["start", "pause", "end"]
    assert kinds(word_then([(320, 640, 0.02)], repause_s=0.04)) == ["start", "pause", "pause", "end"]


def test_the_rooms_own_flutter_gives_no_new_pause():
    # Room noise at 0.001 before the word; after it, 30 ms that are over the quiet threshold
    # but less than 6 dB above that room: no new pause. 10 dB above it: a new pause.
    def scene(amp, **kw):
        probs = [0.0] * 30 + [0.95] * 2 + [0.9] * 8 + [0.6] * 4 + [0.1] * 12
        audio = level(56 * FRAME, 0.001)
        audio[30 * FRAME : 40 * FRAME] = level(10 * FRAME, 0.1)
        at = 40 * FRAME + 480 - (40 * FRAME) % 160
        audio[at : at + 480] = level(480, amp)
        kw = {"quiet_rms": 0.0005, "noise_margin_db": 3.0, **kw}
        return kinds(VAD(probability=scripted(probs), **kw).feed(audio))

    assert scene(0.0018) == ["start", "pause", "end"]  # 5 dB above the room, over the threshold
    assert scene(0.0018, repause_room_db=3.0) == ["start", "pause", "pause", "end"]
    assert scene(0.0032) == ["start", "pause", "pause", "end"]  # 10 dB above it


def test_clicks_that_keep_coming_leave_the_end_to_silero():
    # A click every 0.2 s never leaves 0.24 s of unbroken quiet: Silero's count ends it.
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 4 + [0.1] * 12
    audio = np.zeros(26 * FRAME, dtype=np.float32)
    audio[: 10 * FRAME] = level(10 * FRAME, 0.1)
    for at in range(10 * FRAME + 1600, 26 * FRAME - 160, 3200):
        audio[at : at + 160] = level(160, 0.1)
    ev = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24).feed(audio)
    assert ev[-1].kind == "end" and ev[-1].time == pytest.approx(22 * F)


def test_dc_offset_does_not_hide_the_silence():
    # A constant offset far above quiet_rms under the whole stream: pause and end come from the
    # quiet clock at the same times as without it, and the segment's audio keeps the offset.
    probs = [0.1] * 5 + [0.95] * 2 + [0.9] * 8 + [0.6] * 20
    audio = np.concatenate((level(5 * FRAME, 0.0001), level(10 * FRAME, 0.1), level(20 * FRAME, 0.0001)))
    clean = VAD(probability=scripted(probs), early_end=True).feed(audio)
    assert kinds(clean) == ["start", "pause", "end"] and clean[1].source == "quiet"
    assert clean[2].time == pytest.approx((15 * FRAME + 160) / 16000 + 0.24)  # Silero alone: 35 frames + 0.24 s
    for offset in (0.02, -0.3):
        ev = VAD(probability=scripted(probs), early_end=True).feed(audio + np.float32(offset))
        assert [(e.kind, e.source) for e in ev] == [(e.kind, e.source) for e in clean]
        assert [e.time for e in ev] == pytest.approx([e.time for e in clean])
        assert float(np.mean(ev[2].segment.audio)) == pytest.approx(offset, rel=1e-3)


def test_dc_offset_that_appears_mid_stream_settles_within_100_ms():
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 20
    audio = np.concatenate((level(10 * FRAME, 0.1), np.zeros(20 * FRAME, np.float32)))
    audio[10 * FRAME:] += 0.05  # the silence after the word sits on an offset the word did not have
    ev = VAD(probability=scripted(probs), early_end=True).feed(audio)
    assert kinds(ev) == ["start", "pause", "end"] and ev[1].source == "quiet"
    assert 10 * F < ev[1].time <= 10 * F + 0.1


def test_sustained_energy_rearms_quiet_clock():
    probs = [0.95] * 2 + [0.9] * 30
    audio = level(32 * FRAME, 0.1)
    audio[10 * FRAME : 12 * FRAME] = 0.0  # short quiet, arms
    audio[24 * FRAME :] = 0.0  # real end after 12 loud frames (> 64 ms) rearm
    ev = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24).feed(audio)
    assert kinds(ev) == ["start", "pause", "pause", "end"]
    t_quiet = 12480 / 16000  # first fully quiet 10 ms block after sample 12288
    assert ev[2].time == pytest.approx(t_quiet)
    assert ev[3].time == pytest.approx(t_quiet + 0.24)


def test_silero_resume_invalidates_quiet_clock():
    # soft speech below quiet_rms that Silero still hears: no quiet endpoint
    ev = run(SPEECH + [(2, 0.1, 0.0), (10, 0.9, 0.0), (12, 0.1, 0.0)])
    assert kinds(ev) == ["start", "pause", "pause", "end"]
    assert ev[-1].segment.speech_end == pytest.approx(22 * F)
    assert ev[-1].time == pytest.approx(30 * F)


def test_extend_delays_both_clocks():
    ev = run(SPEECH + [(30, 0.6, 0.0)], extend=lambda t: 0.5)
    t_quiet = (10 * FRAME + 160) / 16000
    assert ev[-1].time == pytest.approx(t_quiet + 0.5)
    ev = run(SPEECH + [(30, 0.1, 0.05)], extend=lambda t: 0.5)
    assert ev[-1].time == pytest.approx(26 * F)  # ceil(0.5 / 0.032) = 16 silent frames


def test_max_speech_forces_end():
    ev = run([(40, 0.95, 0.1)], max_speech_s=0.64)
    assert kinds(ev)[:2] == ["start", "end"]  # continued speech then starts a new segment
    assert ev[1].time == pytest.approx(20 * F)
    assert len(ev[1].segment.audio) == 20 * FRAME


def test_onset_lookback_keeps_soft_opening():
    ev = run([(20, 0.0, 1e-4), (15, 0.3, 0.01), (2, 0.95, 0.1), (5, 0.9, 0.1)])
    start = ev[0]
    assert start.time == pytest.approx(21 * F)
    ev_end = run([(20, 0.0, 1e-4), (15, 0.3, 0.01), (2, 0.95, 0.1), (5, 0.9, 0.1), (12, 0.1, 0.05)])
    assert len(ev_end[-1].segment.audio) == (25 + 5 + 8) * FRAME


def test_lookback_off_and_short_opening_use_plain_preroll():
    ev = run([(20, 0.0, 1e-4), (15, 0.3, 0.01), (2, 0.95, 0.1)], lookback_s=0)
    assert ev[0].time == pytest.approx(36 * F)
    ev = run([(20, 0.0, 1e-4), (5, 0.3, 0.01), (2, 0.95, 0.1)])  # inside 0.3 s pre-roll
    assert ev[0].time == pytest.approx(26 * F)


def test_lookback_never_crosses_playback():
    probs = [0.0] * 20 + [0.3] * 15 + [0.95] * 2
    vad = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24)
    vad.feed(level(20 * FRAME, 1e-4))
    vad.feed(level(15 * FRAME, 0.01), playback=True)
    ev = vad.feed(level(2 * FRAME, 0.1))
    assert kinds(ev) == ["start"]  # the start itself is not gated, only the look-back
    assert ev[0].time == pytest.approx(36 * F)


def test_chunking_and_int16_are_equivalent():
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 20
    audio = np.zeros(30 * FRAME, dtype=np.float32)
    audio[: 10 * FRAME] = level(10 * FRAME, 0.1)
    whole = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24).feed(audio)
    vad = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24)
    pcm = (audio * 32767).astype(np.int16).tobytes()
    parts = []
    for i in range(0, len(pcm), 1234):
        parts += vad.feed(pcm[i : i + 1234])
    assert kinds(parts) == kinds(whole)
    assert [e.time for e in parts] == pytest.approx([e.time for e in whole])


def test_bundled_silero_on_silence():
    pytest.importorskip("onnxruntime")
    vad = VAD()
    assert vad.feed(np.zeros(16000, dtype=np.float32)) == []
    assert not vad.in_speech


def test_audio_so_far_covers_everything_fed_since_preroll():
    probs = [0.0] * 20 + [0.95] * 2 + [0.9] * 20
    signal = level(42 * FRAME, 0.1)
    signal[: 20 * FRAME] = 0.05
    signal += np.arange(len(signal), dtype=np.float32) * 1e-8  # every sample distinct
    vad = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24)
    seen = []
    for i in range(0, len(signal), 1000):
        seen += [(e.kind, i + 1000, vad.audio()) for e in vad.feed(signal[i : i + 1000])]
    kind, fed, audio = seen[0]
    assert kind == "start"
    assert np.array_equal(audio, signal[12 * FRAME : fed])  # 10 pre-roll frames incl. the 2 start frames
    assert np.array_equal(vad.audio(), signal[12 * FRAME :])


def test_no_audio_outside_speech():
    vad = VAD(probability=scripted([0.0] * 5))
    vad.feed(np.zeros(5 * FRAME, dtype=np.float32))
    assert vad.audio() is None


def test_quiet_is_judged_against_room_noise():
    # Room noise at -46 dBFS sits above a fixed -54 dBFS threshold. One pause teaches the
    # noise level; at the next pause the early end works although Silero still says "voiced".
    noise, speech = 0.005, 0.1
    segs = SPEECH + [(4, 0.3, noise), (6, 0.9, speech), (30, 0.6, noise)]
    ev = run(segs)
    assert kinds(ev) == ["start", "pause", "pause", "end"]
    t_pause = 20 * F
    assert ev[-1].segment.speech_end == pytest.approx(t_pause, abs=0.011)
    # room noise raised the threshold, so the quiet clock waits noisy_extra_s (0.04) longer
    assert ev[-1].time == pytest.approx(ev[-1].segment.speech_end + 0.28)
    fixed = run(segs, noise_margin_db=-200.0, noise_subtract_db=None)  # no adaptation: quiet stays at quiet_rms
    assert kinds(fixed) == ["start", "pause"]  # never quiet, Silero never drops: no end


def test_soft_speech_is_never_quiet():
    # Loud pause noise would put noise + margin above the speaker's level; the threshold is
    # capped at speech level - 15 dB, so soft speech keeps the segment open.
    soft = 0.0316  # -30 dBFS speech
    ev = run([(2, 0.95, soft), (8, 0.9, soft), (3, 0.3, 0.02), (30, 0.6, soft)])
    assert kinds(ev) == ["start", "pause"]


QUIET = [(2, 0.95, 0.004), (30, 0.9, 0.004)]  # a quiet speaker at -48 dBFS (a phone line), ~1 s


def noisy_scene(tail_db=None, snr_db=14.0):
    """1.3 s of white room noise, 1 s of a 1 kHz "voice" at -40 dBFS in it, then the noise alone
    (or with the voice going on tail_db softer) while Silero lags 12 frames behind."""
    frames = [40, 30, 40]
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(sum(frames) * FRAME).astype(np.float32) * 10 ** ((-40 - snr_db) / 20)
    voice = 0.01 * math.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(len(noise)) / 16000)
    gain = np.zeros(len(noise))
    gain[frames[0] * FRAME : 70 * FRAME] = 1.0
    if tail_db is not None:
        gain[70 * FRAME :] = 10 ** (tail_db / 20)
    probs = [0.1] * 40 + [0.95] * 30 + [0.6] * 12 + [0.1] * 28
    return probs, noise + (voice * gain).astype(np.float32), 70 * F


def test_steady_noise_close_under_the_speaker_still_gets_an_early_end():
    # Noise 14 dB below the speaker never falls under the quiet threshold (20 dB below them, or
    # 15 dB below their latest level). What is left after subtracting the room's spectrum does.
    probs, audio, off = noisy_scene()
    ev = VAD(probability=scripted(probs), early_end=True).feed(audio)
    assert kinds(ev) == ["start", "pause", "end"] and ev[1].source == "quiet"
    assert off < ev[1].time <= off + 0.03  # speech-off within 30 ms; Silero's first silent frame: 0.384 s
    assert ev[2].time == pytest.approx(ev[1].time + 0.28)  # the noisy wait
    silero = VAD(probability=scripted(probs), early_end=True, noise_subtract_db=None).feed(audio)
    assert [(e.kind, e.source) for e in silero] == [("start", ""), ("pause", "silero"), ("end", "")]
    assert silero[2].time == pytest.approx(off + 20 * F)  # 12 frames of lag + 8 of silence


def test_soft_speech_in_that_noise_is_not_silence():
    # The voice goes on 18 dB softer, below the noise and unheard by Silero: it stands out of
    # the room's spectrum, so the quiet count never completes (a single block may look quiet and
    # give a pause) and Silero's count ends the segment.
    for tail_db in (-12.0, -18.0):
        probs, audio, off = noisy_scene(tail_db=tail_db)
        ev = VAD(probability=scripted(probs), early_end=True).feed(audio)
        assert [e.kind for e in ev if e.kind != "pause"] == ["start", "end"]
        assert ev[-1].time == pytest.approx(off + 20 * F)


def test_noise_subtraction_leaves_clean_input_alone_and_gives_no_end_in_loud_noise():
    # Noise far below the floor: the level decides, exactly as without the subtraction.
    probs, audio, _ = noisy_scene(snr_db=40.0)
    with_it = VAD(probability=scripted(probs), early_end=True).feed(audio)
    without = VAD(probability=scripted(probs), early_end=True, noise_subtract_db=None).feed(audio)
    assert [(e.kind, e.time, e.source) for e in with_it] == [(e.kind, e.time, e.source) for e in without]
    # Noise 4 dB under the speaker leaves too much behind: Silero's count ends the segment.
    probs, audio, off = noisy_scene(snr_db=4.0)
    ev = VAD(probability=scripted(probs), early_end=True).feed(audio)
    assert [e.kind for e in ev if e.kind != "pause"] == ["start", "end"]
    assert ev[-1].time == pytest.approx(off + 20 * F)


def test_soft_ending_of_a_quiet_speaker_is_not_silence():
    # The word ending is at -60 dBFS: below quiet_rms (-54), but only 12 dB under this speaker.
    # A fixed floor would call it silence and end 0.24 s into it; the floor follows the speaker
    # (20 dB below = -68 dBFS), so the segment stays open for all 0.64 s of it.
    soft_end = [(20, 0.9, 0.001)]
    assert kinds(run(QUIET + soft_end)) == ["start"]
    fixed = run(QUIET + soft_end, floor_margin_db=None)  # the old fixed floor cuts it
    assert kinds(fixed) == ["start", "pause", "end"]
    assert fixed[-1].time == pytest.approx(32 * F + 0.01 + 0.24, abs=0.011)


def test_quiet_speaker_still_ends_early_on_silence():
    # After the same quiet speaker the line goes silent (-80 dBFS) while Silero still says
    # "voiced": the quiet clock ends the segment 0.24 s later, as at normal level.
    ev = run(QUIET + [(20, 0.6, 0.0001)])
    assert kinds(ev) == ["start", "pause", "end"]
    assert ev[1].time == pytest.approx(32 * F + 0.01, abs=0.011)  # pause within one 10 ms block
    assert ev[-1].time == pytest.approx(ev[-1].segment.speech_end + 0.24)


def test_floor_is_unchanged_for_a_normal_level_speaker():
    # -20 dBFS speech: 20 dB below is -40 dBFS, above quiet_rms, so quiet_rms stays the floor.
    # Audio at -52 dBFS (above -54) is not quiet with or without the margin; at -56 it is.
    for kw in ({}, {"floor_margin_db": None}):
        assert kinds(run(SPEECH + [(20, 0.6, 0.0025)], **kw)) == ["start"]
        ev = run(SPEECH + [(20, 0.6, 0.0016)], **kw)
        assert kinds(ev) == ["start", "pause", "end"]
        assert ev[-1].time == pytest.approx((10 * FRAME + 160) / 16000 + 0.24)


def test_floor_follows_the_speaker_within_a_second_of_speech():
    # After a loud speaker (-20 dBFS) a quiet one (-48 dBFS) talks for 3 s; the learned level
    # must have come down far enough that their -60 dBFS ending is no longer silence.
    ev = run(SPEECH + [(12, 0.1, 0.0)] + [(2, 0.95, 0.004), (92, 0.9, 0.004), (20, 0.9, 0.001)])
    assert kinds(ev) == ["start", "pause", "end", "start"]


def test_soft_start_after_four_frames_at_half_confidence():
    # A hum or a phone voice that Silero never scores 0.9: four frames (0.128 s) at 0.5 start it.
    ev = run([(3, 0.1, 0.0), (10, 0.6, 0.05)], onset_events=True)
    assert kinds(ev) == ["onset", "start"]
    assert ev[1].time == pytest.approx(4 * F)  # end of the first of the four frames
    assert run([(3, 0.1, 0.0), (3, 0.6, 0.05), (3, 0.1, 0.05)]) == []  # three frames are not enough
    assert run([(3, 0.1, 0.0), (10, 0.6, 0.05)], soft_start_s=None) == []


def test_soft_start_is_never_weaker_than_the_debounce():
    # A longer start_s or a higher confirm_threshold also binds the soft start.
    room = [(3, 0.1, 0.0)]
    assert run(room + [(9, 0.6, 0.05)], start_s=0.3) == []  # 0.3 s are 10 frames
    assert kinds(run(room + [(10, 0.6, 0.05)], start_s=0.3)) == ["start"]
    assert run(room + [(10, 0.6, 0.05)], confirm_threshold=0.8) == []
    assert kinds(run(room + [(10, 0.85, 0.05)], confirm_threshold=0.8)) == ["start"]


def test_soft_start_needs_audio_louder_than_the_room():
    # Steady noise that Silero is unsure about (a fan, babble) sits at the idle floor: no start.
    assert run([(40, 0.6, 0.05)]) == []
    assert run([(20, 0.1, 0.05), (20, 0.6, 0.06)]) == []  # 1.6 dB over the room is not "clearly"
    assert kinds(run([(20, 0.1, 0.005), (20, 0.6, 0.05)])) == ["start"]  # 20 dB over it is


def test_soft_segment_learns_its_level_and_keeps_a_soft_ending():
    # A quiet phone voice (-48 dBFS) at p 0.7 throughout: its -60 dBFS ending is not silence.
    ev = run([(3, 0.1, 0.0), (30, 0.7, 0.004), (20, 0.7, 0.001)])
    assert kinds(ev) == ["start"]


def test_normalize_amplifies_only_what_silero_sees():
    # A phone voice at -48 dBFS that Silero scores 0.6. With normalize_to_dbfs=-30 the frames
    # the model gets rise by 2 dB per frame to -30 dBFS; the segment's audio stays as it came.
    seen = []

    def model(frame):
        seen.append(20 * math.log10(float(np.sqrt(np.mean(frame * frame))) + 1e-12))
        return 0.6 if len(seen) > 3 else 0.1

    quiet = np.concatenate((np.zeros(3 * FRAME, np.float32), level(40 * FRAME, 0.004)))
    vad = VAD(probability=model, normalize_to_dbfs=-30.0)
    assert kinds(vad.feed(quiet)) == ["start"]
    assert seen[3] == pytest.approx(-48, abs=0.1) and seen[-1] == pytest.approx(-30, abs=0.1)
    assert max(b - a for a, b in zip(seen[3:], seen[4:])) <= 2.01
    assert float(np.max(np.abs(vad.audio()))) == pytest.approx(0.004)
    seen.clear()
    VAD(probability=model).feed(quiet)  # off by default
    assert seen[-1] == pytest.approx(-48, abs=0.1)


def test_normalize_never_clips_and_leaves_normal_levels_alone():
    seen = []

    def model(frame):
        seen.append(float(np.max(np.abs(frame))))
        return 0.9

    soft_then_loud = np.concatenate((np.zeros(3 * FRAME, np.float32), level(20 * FRAME, 0.004),
                                     level(5 * FRAME, 0.4)))
    VAD(probability=model, normalize_to_dbfs=-30.0).feed(soft_then_loud)
    assert max(seen) <= 0.51  # the loud frames right after the soft ones stay below -6 dBFS peak
    seen.clear()
    VAD(probability=model, normalize_to_dbfs=-30.0).feed(level(20 * FRAME, 0.1))  # -20 dBFS: no gain
    assert max(seen) == pytest.approx(0.1)


def test_silero_tail_after_early_end_does_not_start_a_segment():
    ev = run(SPEECH + [(20, 0.6, 0.0), (5, 0.95, 0.0)])
    assert kinds(ev) == ["start", "pause", "end"]  # the high tail over silence opens nothing
    ev = run(SPEECH + [(20, 0.6, 0.0), (1, 0.2, 0.0), (3, 0.95, 0.1)])
    assert kinds(ev) == ["start", "pause", "end", "start"]  # once Silero dropped, speech starts


def test_quiet_clock_default_waits_and_never_ends_later_than_silero():
    # Silero (p 0.1, loud noise keeps the quiet clock off) still ends after exactly 8 frames;
    # over near-silence the quiet clock waits end_s, like Silero's own count.
    ev = run(SPEECH + [(12, 0.1, 0.05)], quiet_end_s=None)
    assert ev[-1].time == pytest.approx(18 * F)
    ev = run(SPEECH + [(20, 0.6, 0.0)], quiet_end_s=None)
    t_quiet = (10 * FRAME + 160) / 16000
    assert ev[-1].time == pytest.approx(t_quiet + 0.24)


def test_soft_opening_kept_in_loud_noise():
    # Noise at -30 dBFS, speech only ~8 dB above it: a fixed "12 dB over the floor" never
    # sees the opening; half the speech-to-noise gap does, so the start moves back.
    segs = [(20, 0.0, 0.0316), (15, 0.3, 0.06), (2, 0.95, 0.08), (5, 0.9, 0.08)]
    assert run(segs)[0].time == pytest.approx(21 * F)
    quiet_room = [(20, 0.0, 1e-4), (15, 0.3, 0.01), (2, 0.95, 0.1)]
    assert run(quiet_room)[0].time == pytest.approx(21 * F)  # unchanged where 12 dB is reachable


class Restorable:
    """Scripted probability source with Silero's restore() hook."""

    def __init__(self, probs):
        self.it, self.restores = iter(probs), 0

    def __call__(self, frame):
        return next(self.it, 0.0)

    def restore(self):
        self.restores += 1


def feed(model, segments, **kw):
    vad = VAD(probability=model, **kw)
    for n, _, amp in segments:
        vad.feed(level(n * FRAME, amp))
    return vad


def test_no_silero_restore_over_digital_silence():
    segs = [(70, 0.0, 0.0)]  # 2.2 s of exact zeros: the state doesn't drift there
    m = Restorable([p for n, p, _ in segs for _ in range(n)])
    feed(m, segs)
    assert m.restores == 0


def test_silero_restore_after_a_second_of_low_p_over_real_audio():
    segs = [(40, 0.0, 1e-3)]  # 1.3 s of -60 dBFS room noise, Silero hears nothing
    m = Restorable([p for n, p, _ in segs for _ in range(n)])
    feed(m, segs)
    assert m.restores == 1


def test_restore_comes_when_it_would_without_the_quiet_clock():
    # Silero says "voiced" 4 frames into the silence, then hears nothing over room noise. The
    # quiet clock ends the segment early; the second of "nothing" that leads to a restore still
    # counts from where Silero's own count would have ended it, so both restore on the same frame
    # and Silero's probabilities stay the same with and without the quiet clock.
    segs = SPEECH + [(4, 0.6, 0.0), (60, 0.0, 1e-3)]

    def restored_at(**kw):
        m = Restorable([p for n, p, _ in segs for _ in range(n)])
        vad = VAD(probability=m, early_end=True, end_s=0.24, quiet_end_s=0.24, **kw)
        for i, frame in enumerate(np.concatenate([level(n * FRAME, a) for n, _, a in segs]).reshape(-1, FRAME)):
            vad.feed(frame)
            if m.restores:
                return i
    assert restored_at() == restored_at(quiet_rms=0) is not None


def test_silero_restore_when_it_ignores_speech_loud_audio():
    # A segment teaches the speech level; later Silero scores loud speech only 0.2 ("deaf").
    segs = SPEECH + [(12, 0.1, 1e-3), (6, 0.2, 0.1)]
    m = Restorable([p for n, p, _ in segs for _ in range(n)])
    feed(m, segs, end_s=0.24, quiet_end_s=0.24)
    assert m.restores == 1  # after 6 speech-loud frames (0.19 s), long before a quiet second
    segs = SPEECH + [(12, 0.1, 1e-3), (6, 0.6, 0.1)]
    m = Restorable([p for n, p, _ in segs for _ in range(n)])
    feed(m, segs, end_s=0.24, quiet_end_s=0.24)
    assert m.restores == 0  # a healthy Silero (p >= 0.45 on speech) is left alone


def test_integration_hooks_frames_pause_sources_and_running_segment():
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 3 + [0.1] * 3
    audio = np.zeros(16 * FRAME, dtype=np.float32)
    audio[: 10 * FRAME] = level(10 * FRAME, 0.1)
    vad = VAD(probability=scripted(probs), early_end=True, end_s=0.24, quiet_end_s=0.24, frame_events=True)
    ev = vad.feed(audio)
    frames = [e for e in ev if e.kind == "frame"]
    assert len(frames) == 16 and frames[1].frame.in_speech and frames[0].frame.streak == 1
    assert frames[13].frame.silent_frames == 1  # first unvoiced frame inside speech
    pauses = [e.source for e in ev if e.kind == "pause"]
    assert pauses == ["quiet"]  # the 10 ms clock came first; Silero's own pause shows in frames
    armed, unbroken, blocks = vad.quiet_clock
    assert armed and unbroken and blocks >= 2
    seg = vad.segment()
    assert seg.start == pytest.approx(F) and seg.voiced_s == pytest.approx(13 * F)
    assert seg.end == pytest.approx(16 * F) and len(seg.audio) == 16 * FRAME
    assert vad.onset_lookback_s == 0.0


def test_quiet_rms_zero_turns_the_quiet_clock_off():
    ev = run(SPEECH + [(20, 0.6, 0.0)], quiet_rms=0.0)
    assert kinds(ev) == ["start"]  # no quiet pause, no early end; Silero still says voiced


def test_restore_point_is_never_taken_from_playback_echo():
    class Model(Restorable):
        def __init__(self, probs):
            super().__init__(probs)
            self.snapshots = 0

        def snapshot(self):
            self.snapshots += 1

    probs = [0.95] * 10 + [0.95] * 10
    m = Model(probs)
    vad = VAD(probability=m, early_end=True, end_s=0.24, quiet_end_s=0.24)
    vad.feed(level(10 * FRAME, 0.1))
    user = m.snapshots
    assert user == 10  # the user's confident speech sets the restore point
    vad.feed(level(10 * FRAME, 0.1), playback=True)
    assert m.snapshots == user  # the assistant's own voice (echo) never does


def test_quiet_restore_fires_once_per_pause():
    segs = [(100, 0.0, 1e-3)]  # 3.2 s of -60 dBFS room noise, Silero hears nothing
    m = Restorable([p for n, p, _ in segs for _ in range(n)])
    feed(m, segs)
    assert m.restores == 1  # not every second: one restore un-sticks Silero


def test_speech_resuming_right_after_an_early_end_still_starts():
    # Early quiet-clock end while Silero stays high, then the user goes on talking at once:
    # Silero never drops below continue_threshold, but the loud audio may open the next segment.
    ev = run(SPEECH + [(10, 0.6, 0.0), (8, 0.95, 0.1)])
    assert kinds(ev)[:4] == ["start", "pause", "end", "start"]


def test_onset_comes_first_and_a_click_stays_an_onset():
    # a click: one confident frame and Silero's short fade above continue_threshold
    click = run([(1, 0.95, 0.1), (2, 0.6, 0.1), (6, 0.1, 0.0)], onset_events=True, start_s=0.16)
    assert kinds(click) == ["onset"]
    speech = run([(1, 0.95, 0.1), (8, 0.9, 0.1)], onset_events=True, start_s=0.16)
    assert kinds(speech) == ["onset", "start"]
    assert speech[0].time == pytest.approx(F)  # onset at the first confident frame
    assert speech[1].time == pytest.approx(F)  # the confirmed start still reaches back to it


def test_onset_events_are_off_by_default():
    assert kinds(run([(1, 0.95, 0.1), (8, 0.9, 0.1)])) == ["start"]


def test_start_s_can_change_while_running():
    vad = VAD(probability=scripted([0.95] + [0.9] * 8), start_s=0.5)
    assert vad.feed(level(3 * FRAME, 0.1)) == []
    vad.start_s = 0.064
    assert kinds(vad.feed(level(FRAME, 0.1))) == ["start"]


def test_rearm_s_stays_independent_of_start_s():
    # resumed sound shorter than rearm_s must not re-arm the quiet clock, however long start_s is
    vad = VAD(probability=scripted([0.95] * 10 + [0.6] * 30), start_s=0.3, rearm_s=0.064)
    assert vad._rearm_samples == math.ceil(0.064 * 16000)


def test_confirm_threshold_rejects_a_click_fading_out_above_continue():
    click = [(1, 0.95, 0.1), (4, 0.6, 0.1), (8, 0.1, 0.0)]
    assert kinds(run(click, start_s=0.16)) == ["start", "pause", "end"]  # default: a click passes
    assert run(click, start_s=0.16, confirm_threshold=0.8) == []
    assert kinds(run([(1, 0.95, 0.1), (9, 0.9, 0.1)], start_s=0.16, confirm_threshold=0.8)) == ["start"]


def test_confirm_threshold_does_not_change_continuing_speech():
    # once started, speech continues above continue_threshold as before
    ev = run([(1, 0.95, 0.1), (4, 0.9, 0.1), (6, 0.5, 0.1)], start_s=0.16, confirm_threshold=0.8)
    assert kinds(ev) == ["start"]
