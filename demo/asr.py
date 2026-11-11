"""Minimal client for the RealtimeSTT production server (protocol realtimestt.remote.v1).

One AsrTurn is one speech segment on its own WebSocket:
  start -> audio packets -> preview requests at pauses -> finalize -> completion.
Sending goes through a queue, so the caller (the VAD loop) never waits on the network.
"""

import asyncio
import json
import ssl
import struct
import time

import numpy as np
import websockets


class AsrTurn:
    def __init__(self, url, token, on_event, ca_file=None, language="auto"):
        self.on_event = on_event  # called with each server event (a dict)
        self._queue = asyncio.Queue()
        self._sequence = 0
        self._sent = 0  # samples sent so far
        self._previews = 0
        self._task = asyncio.create_task(self._run(url, token, ca_file, language))

    @property
    def sent(self):
        return self._sent

    def audio(self, samples):
        """Queue float32 or int16 samples (16 kHz mono) that follow the ones already sent."""
        if len(samples) == 0:
            return
        if samples.dtype != np.int16:
            samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        meta = json.dumps({"sampleRate": 16000, "channels": 1, "format": "pcm_s16le",
                           "frames": len(samples), "audioSequence": self._sequence}).encode()
        self._queue.put_nowait(struct.pack("<I", len(meta)) + meta + samples.tobytes())
        self._sequence += 1
        self._sent += len(samples)

    def preview(self, live_flush=True):
        """Ask for the transcript so far. live_flush: answer from the streaming model (fastest)."""
        self._previews += 1
        request_id = f"pause-{self._previews}"
        message = {"type": "preview", "previewRequestId": request_id}
        if live_flush:
            message.update(pauseTextSource="live_flush", pauseTextRefine=True)
        self._queue.put_nowait(json.dumps(message))
        return request_id

    def finalize(self):
        self._queue.put_nowait(json.dumps({"type": "finalize"}))

    def close(self):
        self._queue.put_nowait(None)

    async def _run(self, url, token, ca_file, language):
        context = None
        if url.startswith("wss://"):
            context = ssl.create_default_context(cafile=ca_file)
            context.check_hostname = False  # demo: servers are usually addressed by IP
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with websockets.connect(url, ssl=context, additional_headers=headers,
                                          max_size=None) as ws:
                await ws.send(json.dumps({"type": "start", "turnId": f"demo-{time.time_ns()}",
                                          "language": language}))
                receiver = asyncio.create_task(self._receive(ws))
                while (message := await self._queue.get()) is not None:
                    await ws.send(message)
                await asyncio.wait_for(receiver, 30)
        except Exception as exc:  # report, never crash the VAD loop
            self.on_event({"type": "error", "error": {"message": f"{type(exc).__name__}: {exc}"}})

    async def _receive(self, ws):
        async for raw in ws:
            event = json.loads(raw)
            self.on_event(event)
            if event.get("type") == "completion":
                return
