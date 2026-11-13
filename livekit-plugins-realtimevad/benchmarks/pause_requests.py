"""How many transcripts would ``SpeculativeSTT`` request, and how early is the one that counts?

No STT and no session: each VAD runs once over every sentence, and ``PauseFinder`` (the class the
adapter uses) is replayed over its events for several ``pause`` and ``min_new_speech`` settings.
Not a timing: audio is fed faster than real time and only audio time is read.

  requests     pauses used per sentence (each is one recognize() or one flush)
  last         audio time of the last used pause after the true end of speech, median ms; this is
               when the transcript that becomes the reply is requested
  missed       sentences whose last pause before END_OF_SPEECH was passed over, or that had none:
               their transcript is requested at END_OF_SPEECH, as without the adapter

    python benchmarks/pause_requests.py out.txt
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_session_real_stt as real  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents.vad import VADEventType as V  # noqa: E402
from livekit.plugins import realtimevad, silero  # noqa: E402
from livekit.plugins.realtimevad.stt import PauseFinder  # noqa: E402

VADS = {"silero": lambda: silero.VAD.load(min_silence_duration=0.55), "realtimevad": lambda: realtimevad.VAD.load()}
PAUSES, MIN_NEW = (0.05, 0.1, 0.15, 0.2, 0.3), (0.0, 0.3, 0.6, 1.0)


async def events_of(vad, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    stream, out = vad.stream(), []

    async def read():
        async for ev in stream:
            ev.frames = []
            out.append(ev)

    task = asyncio.create_task(read())
    for i in range(0, len(pcm) - 319, 320):
        stream.push_frame(rtc.AudioFrame(data=pcm[i:i + 320].tobytes(), sample_rate=16000, num_channels=1,
                                         samples_per_channel=320))
        await asyncio.sleep(0.004)
    await asyncio.sleep(0.3)
    stream.end_input()
    await task
    await stream.aclose()
    return out


def replay(events, off, pause, min_new):
    """(requests, audio time of the last used pause relative to ``off`` or None, END_OF_SPEECH relative to ``off``)."""
    finder, used, state, last, end = PauseFinder(pause, min_new), 0, None, None, None
    for ev in events:
        what = finder.step(ev)
        if ev.type == V.END_OF_SPEECH and ev.timestamp >= off - 0.05:
            end, last = ev.timestamp - off, state
            break
        if ev.type in (V.START_OF_SPEECH, V.END_OF_SPEECH) or what == "resume":
            state = None  # that transcript is not the turn's
        elif what == "pause":
            used, state = used + 1, ev.timestamp - off
    return used, last, end


async def main(path):
    items = real.material(("en-US",), 10, 0.5) + real.material(tuple(l for l in real.WHISPER_LANG if l != "en-US"), 2, 0.5)
    seen = {name: [(item, await events_of(make(), item["audio"])) for item in items] for name, make in VADS.items()}
    import speculative_check

    out = [f"Core: {speculative_check.core_version()}.",
           f"SpeculativeSTT's requests, replayed over {len(items)} sentences (10 English, 2 each in 7 more languages), "
           f"speech {sum(i['off'] - i['on'] for i in items):.0f} s. Both VADs with 0.55 s to END_OF_SPEECH.",
           "requests: per sentence. last: the pause whose transcript becomes the reply, median ms of audio time after "
           "the true end of speech. missed: sentences without such a pause.", ""]
    for name, runs in seen.items():
        ends = [replay(evs, item["off"], 0.1, 0.0)[2] for item, evs in runs]
        segments = sum(sum(ev.type == V.END_OF_SPEECH for ev in evs) for _, evs in runs)
        out.append(f"{name}: END_OF_SPEECH median {np.median([e for e in ends if e is not None]) * 1000:.0f} ms after "
                   f"the end of speech, {segments} segments in {len(runs)} sentences")
        for item, evs in runs:
            marks = [f"{'start' if ev.type == V.START_OF_SPEECH else 'end'} {ev.timestamp - item['off']:+.2f}"
                     for ev in evs if ev.type != V.INFERENCE_DONE]
            if len(marks) != 2:
                out.append(f"  {item['id']} (speech {item['on']:.2f}..{item['off']:.2f} s), relative to its end: {', '.join(marks)}")
        out.append(f"{'pause':>8}{'min_new_speech':>16}{'requests':>10}{'last':>8}{'missed':>8}")
        for pause in PAUSES:
            for min_new in MIN_NEW:
                got = [replay(evs, item["off"], pause, min_new) for item, evs in runs]
                last = [g[1] for g in got if g[1] is not None]
                out.append(f"{pause:>8}{min_new:>16}{sum(g[0] for g in got) / len(got):>10.2f}"
                           f"{np.median(last) * 1000 if last else float('nan'):>8.0f}{len(got) - len(last):>8}")
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
