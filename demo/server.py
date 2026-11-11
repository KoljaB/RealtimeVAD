"""RealtimeVAD + RealtimeSTT live demo.

The browser streams 16 kHz int16 microphone audio over a WebSocket. For each chunk:
  1. RealtimeVAD reports start / pause / end (plus one "frame" per 32 ms for the visuals).
  2. On start, a RealtimeSTT turn opens and the segment's audio streams to it.
  3. On every pause, the demo asks for a live-flush Preview: the transcript so far.
  4. On end, the turn is finalized.
Every message to the browser carries stream time (samples received / 16000 s), so the
page can draw VAD decisions, Silero's own decision and ASR results on one timeline.

    python demo/server.py --asr-url wss://host:8767/api/v1/ws/transcribe --asr-ca ca.crt
    (token: --asr-token or the REALTIMESTT_TOKEN environment variable)
"""

import argparse
import asyncio
import math
import os
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from RealtimeVAD import VAD

from asr import AsrTurn

STATIC = Path(__file__).parent / "static"
RATE = 16000
FRAME_S = 512 / RATE

args = None  # set in main()
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/sample.wav")
def sample():
    if not args.sample:
        raise HTTPException(404, "start the server with --sample file.wav")
    return FileResponse(args.sample)


class SileroOnly:
    """What Silero alone would decide, from the very same probabilities RealtimeVAD sees:
    - its pause: the first frame below continue_threshold after RealtimeVAD's quiet pause
    - its turn end: end_s of consecutive frames below continue_threshold."""

    def __init__(self, vad):
        self.needed = math.ceil(vad.end_s / FRAME_S - 1e-9)
        self.threshold = vad.continue_threshold
        self.segment, self.silent, self.quiet_pause = None, 0, None

    def start(self, segment):
        self.segment, self.silent, self.quiet_pause = segment, 0, None

    def pause(self, t, source):
        """RealtimeVAD paused. From the quiet clock: wait for Silero's own silent frame."""
        self.quiet_pause = t if source == "quiet" else None

    def frame(self, p, t):
        """Messages for the page: Silero's pause lagging a quiet pause, Silero's turn end."""
        out = []
        if self.segment is None:
            return out
        self.silent = self.silent + 1 if p < self.threshold else 0
        if self.quiet_pause is not None and t - self.quiet_pause > 0.6:
            self.quiet_pause = None  # speech went on; this quiet pause has no Silero partner
        if self.silent == 1 and self.quiet_pause is not None:
            out.append({"type": "sileroPause", "time": t, "pauseTime": self.quiet_pause,
                        "segment": self.segment})
            self.quiet_pause = None
        if self.silent >= self.needed:
            out.append({"type": "sileroEnd", "time": t, "segment": self.segment})
            self.segment = None
        return out


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    vad = VAD(frame_events=True)
    silero = SileroOnly(vad)
    outbox: asyncio.Queue = asyncio.Queue()
    received = 0  # samples, = stream time * RATE
    segment = 0
    turn = None
    requests = {}  # preview request id -> (wall time, stream time) of the pause that asked

    def now():
        return received / RATE

    def send(message):
        message["now"] = now()
        outbox.put_nowait(message)

    def on_asr(event, segment_no):
        kind = event.get("type")
        if kind not in ("partial", "preview", "previewRefined", "previewRecord", "completion", "error"):
            return
        message = {"type": "asr", "event": kind, "segment": segment_no,
                   "text": event.get("mergedText") or event.get("text") or "",
                   "source": event.get("pauseTextSource"), "status": event.get("status")}
        asked = requests.get(event.get("previewRequestId"))
        if asked:
            message["latencyMs"] = round((time.perf_counter() - asked[0]) * 1000, 1)
            message["pauseTime"] = asked[1]
        if kind == "error":
            message["text"] = (event.get("error") or {}).get("message", "ASR error")
        send(message)

    async def sender():
        while True:
            await ws.send_json(await outbox.get())

    def sync_audio():
        """Send the running segment's audio the ASR turn hasn't had yet."""
        audio = vad.audio()
        if turn is not None and audio is not None:
            turn.audio(audio[turn.sent:])

    sending = asyncio.create_task(sender())
    try:
        while True:
            chunk = np.frombuffer(await ws.receive_bytes(), dtype=np.int16)
            events = vad.feed(chunk)
            received += len(chunk)
            for event in events:
                if event.kind == "frame":
                    f = event.frame
                    send({"type": "frame", "time": event.time, "p": round(f.probability, 3),
                          "voiced": f.voiced, "inSpeech": f.in_speech,
                          "quiet": vad._quiet,  # current (noise-adaptive) quiet threshold, RMS
                          "inferMs": round(f.infer_s * 1000, 3)})
                    for message in silero.frame(f.probability, event.time):
                        send(message)
                elif event.kind == "start":
                    segment += 1
                    silero.start(segment)
                    turn = AsrTurn(args.asr_url, args.asr_token, lambda e, n=segment: on_asr(e, n),
                                   ca_file=args.asr_ca, language=args.language)
                    send({"type": "start", "time": event.time, "segment": segment})
                elif event.kind == "pause":
                    sync_audio()
                    request = turn.preview()
                    requests[request] = (time.perf_counter(), event.time)
                    silero.pause(event.time, event.source)
                    send({"type": "pause", "time": event.time, "segment": segment,
                          "source": event.source})
                elif event.kind == "end":
                    s = event.segment
                    turn.audio(s.audio[turn.sent:])
                    turn.finalize()
                    turn.close()
                    turn = None
                    send({"type": "end", "time": event.time, "segment": segment, "start": s.start,
                          "speechEnd": s.speech_end, "voicedS": round(s.voiced_s, 3)})
            sync_audio()
    except WebSocketDisconnect:
        pass
    finally:
        if turn is not None:
            turn.close()
        sending.cancel()


def main():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asr-url", default="ws://127.0.0.1:8010/api/v1/ws/transcribe")
    parser.add_argument("--asr-token", default=os.environ.get("REALTIMESTT_TOKEN"))
    parser.add_argument("--asr-ca", help="CA bundle for a wss:// server with its own certificate")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--sample", help="a 16 kHz mono WAV the page can play instead of the mic")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    print(f"RealtimeVAD demo: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
