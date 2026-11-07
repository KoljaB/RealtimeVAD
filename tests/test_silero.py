"""Silero deaf-state recovery, with a scripted ONNX session."""

import numpy as np

from RealtimeVAD.silero import Silero


class Session:
    def __init__(self, probs):
        self.probs, self.i, self.states_in = probs, 0, []

    def run(self, _, feed):
        self.states_in.append(feed["state"][0, 0, 0])
        p = self.probs[self.i]
        self.i += 1
        return np.array([[p]], np.float32), np.full((2, 1, 128), float(self.i), np.float32)


def model(probs, restore_s=1.0):
    s = Silero.__new__(Silero)  # skip loading the real model
    s._session, s._sr = Session(probs), np.asarray(16000, np.int64)
    s._restore_frames = 32 if restore_s else 0
    s.restores = 0
    s.reset()
    return s


def test_restores_state_of_last_confident_speech_after_one_second_of_low_p():
    probs = [0.95, 0.5] + [0.0] * 32 + [0.0]
    m = model(probs)
    for _ in probs:
        m(np.zeros(512, np.float32))
    assert m.restores == 1
    assert m._session.states_in[-1] == 1.0  # state saved after frame 1 (the p >= 0.9 one)


def test_zero_state_before_any_confident_speech_and_off_switch():
    m = model([0.0] * 33)
    for _ in range(33):
        m(np.zeros(512, np.float32))
    assert m.restores == 1 and m._session.states_in[-1] == 0.0
    m = model([0.0] * 40, restore_s=0)
    for _ in range(40):
        m(np.zeros(512, np.float32))
    assert m.restores == 0
