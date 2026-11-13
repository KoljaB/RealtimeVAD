"""Drop-in for ``silero_vad.VADIterator``: same constructor, call and results."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .silero import SAMPLE_RATE
from .vad import FRAME_S, VAD


class _SileroModel:
    """Adapts a model from ``silero_vad.load_silero_vad()`` to a frame -> probability call."""

    def __init__(self, model) -> None:
        self._model = model
        try:
            import torch

            self._tensor = torch.from_numpy
        except ImportError:
            self._tensor = lambda a: a

    def __call__(self, frame: np.ndarray) -> float:
        p = self._model(self._tensor(np.ascontiguousarray(frame)), SAMPLE_RATE)
        return float(p.item() if hasattr(p, "item") else p)

    def reset(self) -> None:
        self._model.reset_states()


class VADIterator:
    """Like Silero's ``VADIterator``: ``end`` is decided when Silero has heard the minimum
    silence, and its sample is where the audio went quiet (Silero's own lags about 100 ms).

    ``model`` may be a Silero model (used as given) or None (the bundled one).
    Extra keyword arguments go to :class:`VAD`: ``early_end=True`` decides ``end`` as soon as
    the audio has been quiet that long. ``.vad`` gives the events, among them the early
    ``pause``. Chunks may have any size.
    """

    def __init__(
        self,
        model=None,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        neg_threshold: Optional[float] = None,
        **options,
    ) -> None:
        if sampling_rate != SAMPLE_RATE:
            raise ValueError("RealtimeVAD supports 16000 Hz only")
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)
        self.sampling_rate = sampling_rate
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.vad = VAD(
            start_threshold=threshold,
            continue_threshold=neg_threshold,
            start_s=FRAME_S,  # Silero starts on one frame
            end_s=min_silence_duration_ms / 1000,
            preroll_s=speech_pad_ms / 1000,
            probability=None if model is None else _SileroModel(model),
            **options,
        )

    @property
    def triggered(self) -> bool:
        return self.vad.in_speech

    def reset_states(self) -> None:
        self.vad.reset()

    def __call__(self, x, return_seconds: bool = False, time_resolution: int = 1) -> Optional[dict]:
        result = {}
        for event in self.vad.feed(np.asarray(x)):
            if event.kind == "start":  # Silero: start of the first speech frame, minus the pad
                sample = (event.time - FRAME_S) * self.sampling_rate - self.speech_pad_samples
            elif event.kind == "end":  # Silero: where speech stopped, plus the pad
                sample = event.segment.speech_end * self.sampling_rate + self.speech_pad_samples
            else:
                continue
            sample = max(0, int(round(sample)))
            result[event.kind] = round(sample / self.sampling_rate, time_resolution) if return_seconds else sample
        return result or None
