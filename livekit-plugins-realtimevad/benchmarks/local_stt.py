"""Two real, locally running STTs behind LiveKit's ``stt.STT`` interface, for ``agent_session_real_stt.py``.

``SherpaSTT``   a streaming recognizer (sherpa-onnx streaming zipformer transducer, CPU) with interim
                and final transcripts. It finalizes like a streaming STT service does: on its own
                endpoint rule (``endpoint_silence`` seconds without a new token after the last one),
                and, with ``manual_flush=True``, also when LiveKit flushes the stream. LiveKit does
                that at the VAD's END_OF_SPEECH for an STT with ``STTCapabilities.manual_flush``
                (``AudioRecognition._run_eou_detection``, ``auto_flush``).
``WhisperSTT``  a non-streaming recognizer (faster-whisper, CPU): ``recognize()`` only. LiveKit wraps
                it in ``stt.StreamAdapter`` with the session's VAD, one request per VAD segment.

Benchmark helpers, not part of the plugin. Models are read from ``REALTIMEVAD_STT_MODELS`` (a folder
with ``sherpa-onnx-streaming-zipformer-en-2023-06-26/`` from the sherpa-onnx ``asr-models`` release and
``whisper/`` as faster-whisper's download root).
"""

from __future__ import annotations

import asyncio
import os
import time
import weakref
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN

MODELS = Path(os.environ.get("REALTIMEVAD_STT_MODELS") or Path(__file__).resolve().parents[2] / ".stt-models")
ZIPFORMER = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
SR = 16000
TAIL = np.zeros(int(0.5 * SR), dtype=np.float32)  # sherpa-onnx: pad before input_finished(), or the last word is cut


def _event(kind, text: str, language: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(type=kind, alternatives=[stt.SpeechData(language=language, text=text)])


def _floats(frame: rtc.AudioFrame) -> np.ndarray:
    x = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
    if frame.num_channels > 1:
        x = x.reshape(-1, frame.num_channels)[:, 0]
    if frame.sample_rate != SR:  # the benchmark feeds 16 kHz; anything else, linearly
        x = np.interp(np.arange(int(len(x) * SR / frame.sample_rate)) * frame.sample_rate / SR, np.arange(len(x)), x)
    return np.ascontiguousarray(x, dtype=np.float32)


class SherpaSTT(stt.STT):
    def __init__(self, *, endpoint_silence: float = 1.2, manual_flush: bool = False, log=None) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True,
                                                          offline_recognize=False, manual_flush=manual_flush))
        import sherpa_onnx

        d = MODELS / ZIPFORMER
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
            decoder=str(d / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
            joiner=str(d / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
            num_threads=1, sample_rate=SR, feature_dim=80, decoding_method="greedy_search", provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,  # nothing decoded yet (sherpa-onnx default)
            rule2_min_trailing_silence=endpoint_silence,  # after the last token (sherpa-onnx default 1.2)
            rule3_min_utterance_length=300.0)
        self._log = log or (lambda *a, **k: None)
        self._streams = weakref.WeakSet()

    @property
    def model(self) -> str:
        return ZIPFORMER

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        raise NotImplementedError("streaming only")

    def stream(self, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> "SherpaStream":
        stream = SherpaStream(stt=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def flush_streams(self) -> None:
        """Finalize now (for an application that flushes earlier than LiveKit does)."""
        for stream in list(self._streams):
            try:
                stream.flush()
            except RuntimeError:  # a closed stream of an earlier session
                self._streams.discard(stream)


class SherpaStream(stt.RecognizeStream):
    def _step(self, samples: np.ndarray | None):
        """Executor: feed audio (None = finish the utterance). Returns (text, final?)."""
        r, s = self._stt._recognizer, self._s
        if samples is None:
            s.accept_waveform(SR, TAIL)
            s.input_finished()
        else:
            s.accept_waveform(SR, samples)
        while r.is_ready(s):
            r.decode_stream(s)
        text = r.get_result(s).strip().capitalize()
        final = samples is None or r.is_endpoint(s)
        if final:
            if samples is None:
                self._s = r.create_stream()
            else:
                r.reset(s)
        return text, final

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        self._s = self._stt._recognizer.create_stream()
        last = ""
        async for item in self._input_ch:
            flush = isinstance(item, self._FlushSentinel)
            text, final = await loop.run_in_executor(None, self._step, None if flush else _floats(item))
            if final:
                self._stt._log("stt_flush" if flush else "stt_endpoint", text)
                if text:
                    self._event_ch.send_nowait(_event(stt.SpeechEventType.FINAL_TRANSCRIPT, text, "en"))
                last = ""
            elif text != last:
                last = text
                self._event_ch.send_nowait(_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, "en"))


_WHISPER: dict = {}


def whisper_model(size: str, workers: int = 1, threads: int = 4):
    if size not in _WHISPER:
        from faster_whisper import WhisperModel

        _WHISPER[size] = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads,
                                      num_workers=workers, download_root=str(MODELS / "whisper"),
                                      local_files_only=True)  # no request to the model hub
    return _WHISPER[size]


class WhisperSTT(stt.STT):
    """Non-streaming: one ``recognize()`` per buffer. ``log(kind, value, **extra)`` gets every request."""

    def __init__(self, *, size: str = "small", language: str = "en", log=None) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._size, self._language, self._model = size, language, whisper_model(size)
        self._log = log or (lambda *a, **k: None)

    @property
    def model(self) -> str:
        return "faster-whisper-" + self._size

    @property
    def provider(self) -> str:
        return "faster-whisper"

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language=self._language, beam_size=1, vad_filter=False,
                                             condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segments).strip()

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        audio = _floats(utils.merge_frames(buffer) if isinstance(buffer, list) else buffer)
        t = time.time()
        self._log("stt_request_start", None, audio_s=round(len(audio) / SR, 3))
        text = await asyncio.get_running_loop().run_in_executor(None, self._transcribe, audio)
        self._log("stt_request", text, audio_s=round(len(audio) / SR, 3), took=round(time.time() - t, 3))
        return _event(stt.SpeechEventType.FINAL_TRANSCRIPT, text, self._language)
