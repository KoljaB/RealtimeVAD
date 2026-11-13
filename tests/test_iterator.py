"""The Silero VADIterator drop-in: same call pattern, same result shape."""

import numpy as np

from RealtimeVAD import VADIterator


class FakeSilero:
    def __init__(self, probs):
        self.probs, self.i, self.resets = probs, 0, 0

    def __call__(self, x, sr):
        assert sr == 16000 and len(x) == 512
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset_states(self):
        self.resets += 1


def stream(it, audio, **kw):
    return [(i // 512, r) for i in range(0, len(audio), 512) if (r := it(audio[i : i + 512], **kw))]


def scenario():
    probs = [0.0] * 10 + [0.9] * 10 + [0.6] * 5 + [0.2] * 15  # Silero lags 5 frames after speech
    audio = np.zeros(40 * 512, dtype=np.float32)
    audio[: 10 * 512] = 0.05  # room noise
    audio[10 * 512 : 20 * 512] = 0.1  # speech, then silence
    audio[1::2] *= -1  # no DC part: a constant value is an offset, not sound
    return FakeSilero(probs), audio


def test_same_results_shape_and_silero_start_sample():
    model, audio = scenario()
    it = VADIterator(model, sampling_rate=16000)
    assert model.resets == 1
    out = stream(it, audio)
    assert out[0] == (10, {"start": 10 * 512 - 480})  # first speech chunk minus 30 ms pad
    chunk, end = out[1]
    assert end == {"end": 10400 + 480}  # speech stopped at the first quiet 10 ms block, plus pad
    assert chunk == 24  # quiet clock: 10400 + 140 ms (min_silence + 40 ms); Silero's own count ends at chunk 29
    assert not it.triggered


def test_return_seconds_and_reset():
    model, audio = scenario()
    it = VADIterator(model)
    out = stream(it, audio, return_seconds=True)
    assert out[0][1] == {"start": 0.3}
    it.reset_states()
    assert model.resets == 2 and not it.triggered
