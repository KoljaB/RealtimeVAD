"""A LiveKit agent whose STT is asked for the transcript at the user's pause, not at END_OF_SPEECH.

Modelled on livekit/agents ``examples/voice_agents/basic_agent.py`` (1.8.11). The only differences
are the ``vad=`` and ``stt=`` lines. Not run by this repository's tests: it needs a LiveKit server
and API keys for the STT, LLM and TTS you pick. The same construction with local models is
measured in ``benchmarks/speculative_check.py``.

    pip install "livekit-agents[openai]" realtimevad ./livekit-plugins-realtimevad
    python examples/speculative_agent.py dev
"""

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import openai, realtimevad

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    vad = realtimevad.VAD.load()  # END_OF_SPEECH where the Silero plugin puts it; the pause is early
    session = AgentSession(
        vad=vad,
        # Any non-streaming STT (LiveKit would put it behind its StreamAdapter and call it at
        # END_OF_SPEECH), or a streaming one with capabilities.manual_flush.
        stt=realtimevad.SpeculativeSTT(stt=openai.STT(model="whisper-1"), vad=vad),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_handling={
            # The reply starts from the transcript of each pause. A pause that was not the end costs
            # one of these tries; LiveKit's defaults are 3 tries and no tries after 10 s of speech.
            "preemptive_generation": {"enabled": True, "max_retries": 6, "max_speech_duration": 20.0},
        },
    )
    # optional: RealtimeVAD does not learn the room from audio that may hold the agent's own voice
    session.on("agent_state_changed", lambda ev: vad.set_agent_speaking(ev.new_state == "speaking"))
    await session.start(agent=Agent(instructions="You are a helpful voice assistant. Keep answers short."),
                        room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
