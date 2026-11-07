"""Stateful Silero v6 ONNX inference on 512-sample, 16 kHz frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 512
MODEL = Path(__file__).parent / "assets" / "silero_vad_v6.2.1.onnx"


class Silero:
    def __init__(self, model: str | Path = MODEL, threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.asarray(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        audio = np.asarray(frame, dtype=np.float32).reshape(1, FRAME)
        p, self._state = self._session.run(
            None,
            {"input": np.concatenate((self._context, audio), axis=1), "sr": self._sr, "state": self._state},
        )
        self._context = audio[:, -64:].copy()
        return float(np.asarray(p).reshape(-1)[0])
