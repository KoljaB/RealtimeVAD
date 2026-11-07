"""RealtimeVAD: fast speech-end detection for low-latency voice assistants."""

from .iterator import VADIterator
from .vad import VAD, Event, Segment

__version__ = "0.0.1"

__all__ = ["VAD", "VADIterator", "Event", "Segment", "__version__"]
