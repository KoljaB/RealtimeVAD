"""RealtimeVAD live demo: when RealtimeVAD knows you stopped, and when Silero alone would.

The browser streams 16 kHz int16 microphone audio over a WebSocket. RealtimeVAD reports
onset / start / pause / end, plus one "frame" per 32 ms with Silero's probability. From those same
probabilities the server also works out what Silero alone (same thresholds) would decide.
Every message carries stream time (samples received / 16000 s), so the page can draw both
on one timeline.

    python demo/server.py            then open http://127.0.0.1:8200

--record DIR keeps what a session received: the audio as a WAV and every message sent to the
page as JSON lines (with the detector's learned levels on each frame), one pair of files per
connection. Off by default; nothing is written without it. A session that looked wrong can
then be replayed: python benchmarks/long_stream.py DIR/<session>.wav
"""

import argparse
import asyncio
import json
import math
import time
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from RealtimeVAD import VAD

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


class SileroAlone:
    """What Silero alone would decide, from the very same probabilities RealtimeVAD sees.
    Speech-off: its first frame below continue_threshold. Turn end: end_s of such frames."""

    def __init__(self, vad):
        self.needed = math.ceil(vad.end_s / FRAME_S - 1e-9)
        self.threshold = vad.continue_threshold
        self.segment, self.silent, self.silent_since = None, 0, None

    def start(self, segment):
        self.segment, self.silent = segment, 0

    def frame(self, p, t):
        """Returns Silero's turn end, with its speech-off, once it is reached."""
        if self.segment is None:
            return None
        self.silent = self.silent + 1 if p < self.threshold else 0
        if self.silent == 1:
            self.silent_since = t
        if self.silent < self.needed:
            return None
        segment, self.segment = self.segment, None
        return {"type": "sileroEnd", "time": t, "silentSince": self.silent_since, "segment": segment}


class Recording:
    """One session's audio (as received) and messages, for --record."""

    def __init__(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        base = folder / time.strftime("session-%Y%m%d-%H%M%S")
        n = 1
        while base.with_suffix(".wav").exists():  # two connections in one second
            n += 1
            base = folder / f"{base.name.split('_')[0]}_{n}"
        self.wav = wave.open(str(base.with_suffix(".wav")), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(RATE)
        self.log = open(base.with_suffix(".jsonl"), "w", encoding="utf-8")
        print(f"recording to {base}.wav and .jsonl")

    def audio(self, chunk):
        self.wav.writeframes(chunk.tobytes())

    def message(self, message):
        self.log.write(json.dumps(message) + "\n")

    def close(self):
        self.wav.close()
        self.log.close()


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    record = Recording(args.record) if args.record else None
    try:
        await session(ws, record)
    finally:
        if record is not None:
            record.close()


async def session(ws: WebSocket, record):
    # start_s (speech-on debounce) follows the page slider; the quiet clock keeps re-arming
    # after 64 ms of sound, so the debounce never changes the pauses
    vad = VAD(frame_events=True, onset_events=True, start_s=0.32, confirm_threshold=0.8, rearm_s=0.064)
    silero = SileroAlone(vad)
    received = segment = 0  # samples received, speech segments so far
    pause_wait, pending = 0.0, None  # slider: extra quiet a pause needs; quiet clock start
    last_frame = 0.0  # latest reported model frame; a start comes just before its own frame event

    async def send(message):
        if record is not None:
            more = {}
            if message["type"] == "frame":  # what the detector has learned, dBFS: only in the file
                more = {"speaker": vad._level_db, "latestSpeech": vad._speech_db, "room": vad._noise_db}
            record.message({**message, **more})
        await ws.send_json(message)

    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("text"):  # sliders: pauseMs quiet for a pause; startMs and startP speech for a start
            config = json.loads(message["text"])
            if record is not None:
                record.message({"type": "config", "time": received / RATE, **config})
            if "pauseMs" in config:
                pause_wait = max(0.0, config["pauseMs"] / 1000 - 0.010)
            if "startMs" in config:
                vad.start_s = config["startMs"] / 1000
            if "startP" in config:
                vad.confirm_threshold = config["startP"]
            continue
        chunk = np.frombuffer(message["bytes"], dtype=np.int16)
        received += len(chunk)
        if record is not None:
            record.audio(chunk)
        for event in vad.feed(chunk):
            if event.kind == "frame":
                f = event.frame
                last_frame = event.time
                await send({"type": "frame", "time": event.time, "p": round(f.probability, 3),
                            "quiet": vad._quiet,  # current noise-adaptive quiet threshold (RMS)
                            "inferMs": round(f.infer_s * 1000, 3)})
                if (end := silero.frame(f.probability, event.time)) is not None:
                    await send(end)
            elif event.kind == "onset":  # maybe speech: a confident frame, maybe a click
                await send({"type": "onset", "time": event.time})
            elif event.kind == "start":  # confirmed; its time reaches back to the onset
                segment += 1
                silero.start(segment)
                await send({"type": "start", "time": event.time, "confirmed": last_frame + FRAME_S, "segment": segment})
            elif event.kind == "pause":
                if event.source == "quiet" and pause_wait > 0:
                    pending = event.time  # decided below, once the quiet has lasted
                else:
                    await send({"type": "pause", "time": event.time, "segment": segment})
            elif event.kind == "end":
                pending = None
                s = event.segment
                await send({"type": "end", "time": event.time, "segment": segment,
                            "speechEnd": s.speech_end, "voicedS": round(s.voiced_s, 3)})
        if pending is not None:  # slider > 10 ms: the quiet must last before it is a pause
            armed, unbroken, blocks = vad.quiet_clock  # 10 ms blocks since the clock armed
            if not (vad.in_speech and armed and unbroken):
                pending = None  # sound came back first: no pause
            elif (blocks - 1) * 0.010 >= pause_wait - 1e-9:
                await send({"type": "pause", "time": pending + (blocks - 1) * 0.010, "segment": segment})
                pending = None


def main():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", help="a 16 kHz mono WAV the page can play instead of the mic")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--record", metavar="DIR", help="keep each session's audio and events in this folder")
    args = parser.parse_args()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    print(f"RealtimeVAD demo: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
