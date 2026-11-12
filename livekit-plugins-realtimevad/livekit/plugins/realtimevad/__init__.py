"""RealtimeVAD plugin for LiveKit Agents: ``realtimevad.VAD.load()`` in place of ``silero.VAD.load()``."""

from livekit.agents import Plugin

from .log import logger
from .vad import VAD, VADStream
from .version import __version__

__all__ = ["VAD", "VADStream", "__version__"]


class RealtimeVADPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(RealtimeVADPlugin())
