"""Stateful Silero v6 ONNX inference on 512-sample, 16 kHz frames."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 512
MODEL = Path(__file__).parent / "assets" / "silero_vad_v6.2.1.onnx"


class Silero:
    """Silero v6 with recovery from its "deaf" state.

    On a long stream the recurrent state can drift into an attractor where the
    probability stays near 0 even through loud speech (low babble, near-silent
    pauses). After ``restore_s`` of p < 0.1 the state is put back to where the
    model last heard confident speech (p >= 0.9), or zeroed before any. A zero
    state is the worse restart point: it scores the next utterance lower and
    later. Costs no extra inference; 0 disables it.
    """

    def __init__(self, model: str | Path = MODEL, threads: int = 1, restore_s: float = 1.0) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.asarray(SAMPLE_RATE, dtype=np.int64)
        self._restore_frames = math.ceil(restore_s * SAMPLE_RATE / FRAME - 1e-9) if restore_s > 0 else 0
        self.restores = 0
        self.reset()

    def reset(self) -> None:
        """Start a new stream: zero state and context, forget the speech snapshot."""
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._snapshot = None
        self._quiet = 0

    def __call__(self, frame: np.ndarray) -> float:
        audio = np.asarray(frame, dtype=np.float32).reshape(1, FRAME)
        p, self._state = self._session.run(
            None,
            {"input": np.concatenate((self._context, audio), axis=1), "sr": self._sr, "state": self._state},
        )
        self._context = audio[:, -64:].copy()
        p = float(np.asarray(p).reshape(-1)[0])
        if p >= 0.9:
            self._snapshot = (self._state, self._context)  # fresh arrays each frame
        self._quiet = self._quiet + 1 if p < 0.1 else 0
        if self._restore_frames and self._quiet >= self._restore_frames:
            self._quiet = 0  # affects the next frame only
            self.restores += 1
            if self._snapshot is None:
                self.reset()
            else:
                self._state, self._context = (a.copy() for a in self._snapshot)
        return p
