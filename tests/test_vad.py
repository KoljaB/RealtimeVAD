"""Spec checks with a scripted probability per 32 ms frame (see SPEC.md)."""

import numpy as np
import pytest

from RealtimeVAD import VAD

FRAME = 512
F = FRAME / 16000


def scripted(probs):
    it = iter(probs)
    return lambda frame: next(it, 0.0)


def run(segments, **kw):
    """segments: (frames, p, amplitude) -> constant-level audio, scripted p."""
    probs, audio = [], []
    for n, p, amp in segments:
        probs += [p] * n
        audio.append(np.full(n * FRAME, amp, dtype=np.float32))
    kw.setdefault("end_s", 0.24)  # timings below assume 0.24 s
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


def test_click_breaks_quiet_clock_and_silero_takes_over():
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 4 + [0.1] * 12
    audio = np.full(26 * FRAME, 0.0, dtype=np.float32)
    audio[: 10 * FRAME] = 0.1
    click = 12 * FRAME
    audio[click : click + 160] = 0.1  # one 10 ms loud block, < 64 ms: no rearm
    ev = VAD(probability=scripted(probs), end_s=0.24).feed(audio)
    assert kinds(ev) == ["start", "pause", "end"]
    assert ev[-1].time == pytest.approx(22 * F)  # Silero: 8 silent frames after frame 13
    assert ev[-1].segment.speech_end == pytest.approx(14 * F)


def test_sustained_energy_rearms_quiet_clock():
    probs = [0.95] * 2 + [0.9] * 30
    audio = np.full(32 * FRAME, 0.1, dtype=np.float32)
    audio[10 * FRAME : 12 * FRAME] = 0.0  # short quiet, arms
    audio[24 * FRAME :] = 0.0  # real end after 12 loud frames (> 64 ms) rearm
    ev = VAD(probability=scripted(probs), end_s=0.24).feed(audio)
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
    vad = VAD(probability=scripted(probs), end_s=0.24)
    vad.feed(np.full(20 * FRAME, 1e-4, dtype=np.float32))
    vad.feed(np.full(15 * FRAME, 0.01, dtype=np.float32), playback=True)
    ev = vad.feed(np.full(2 * FRAME, 0.1, dtype=np.float32))
    assert kinds(ev) == ["start"]  # the start itself is not gated, only the look-back
    assert ev[0].time == pytest.approx(36 * F)


def test_chunking_and_int16_are_equivalent():
    probs = [0.95] * 2 + [0.9] * 8 + [0.6] * 20
    audio = np.zeros(30 * FRAME, dtype=np.float32)
    audio[: 10 * FRAME] = 0.1
    whole = VAD(probability=scripted(probs), end_s=0.24).feed(audio)
    vad = VAD(probability=scripted(probs), end_s=0.24)
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
    signal = np.full(42 * FRAME, 0.1, dtype=np.float32)
    signal[: 20 * FRAME] = 0.05
    signal += np.arange(len(signal), dtype=np.float32) * 1e-8  # every sample distinct
    vad = VAD(probability=scripted(probs), end_s=0.24)
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
    assert ev[-1].time == pytest.approx(ev[-1].segment.speech_end + 0.24)
    fixed = run(segs, noise_margin_db=-200.0)  # no adaptation: quiet stays at quiet_rms
    assert kinds(fixed) == ["start", "pause"]  # never quiet, Silero never drops: no end


def test_soft_speech_is_never_quiet():
    # Loud pause noise would put noise + margin above the speaker's level; the threshold is
    # capped at speech level - 15 dB, so soft speech keeps the segment open.
    soft = 0.0316  # -30 dBFS speech
    ev = run([(2, 0.95, soft), (8, 0.9, soft), (3, 0.3, 0.02), (30, 0.6, soft)])
    assert kinds(ev) == ["start", "pause"]


def test_silero_tail_after_early_end_does_not_start_a_segment():
    ev = run(SPEECH + [(20, 0.6, 0.0), (5, 0.95, 0.0)])
    assert kinds(ev) == ["start", "pause", "end"]  # the high tail over silence opens nothing
    ev = run(SPEECH + [(20, 0.6, 0.0), (1, 0.2, 0.0), (3, 0.95, 0.1)])
    assert kinds(ev) == ["start", "pause", "end", "start"]  # once Silero dropped, speech starts


def test_optional_silero_reset_after_clear_non_speech():
    class Model:
        resets = 0

        def __call__(self, frame):
            return 0.0

        def reset(self):
            Model.resets += 1

    vad = VAD(probability=Model(), reset_after_s=1.0)
    before = Model.resets
    vad.feed(np.zeros(int(2.1 * 16000), dtype=np.float32))
    assert Model.resets - before == 2  # once per second of p < 0.1
    vad = VAD(probability=Model())
    before = Model.resets
    vad.feed(np.zeros(2 * 16000, dtype=np.float32))
    assert Model.resets == before  # off by default
