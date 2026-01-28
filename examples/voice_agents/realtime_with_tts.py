import logging

from dotenv import load_dotenv
from google.genai.types import Modality  # noqa: F401

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.agents.llm import function_tool
from livekit.plugins import google, openai  # noqa: F401

logger = logging.getLogger("realtime-with-tts")
logger.setLevel(logging.INFO)

load_dotenv()

# word lists
IGNORE_WORDS = {"yeah", "ok", "okay", "hmm", "oh", "right"}
INTERRUPT_WORDS = {"stop", "wait", "no","listen","Wrong"}

class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.realtime.RealtimeModel(modalities=["text"]),
            # llm=google.beta.realtime.RealtimeModel(modalities=[Modality.TEXT]),
            tts=openai.TTS(voice="ash"),
        )

    @function_tool
    async def get_weather(self, location: str):
        logger.info(f"getting weather for {location}")
        return f"The weather in {location} is sunny, and the temperature is 20 degrees Celsius."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()

    # agent speaking state
    session.agent_is_speaking = False

    @session.on("audio_segment_start")
    def _on_audio_start():
        session.agent_is_speaking = True

    @session.on("audio_segment_end")
    def _on_audio_end():
        session.agent_is_speaking = False

    #STT-based interruption logic
    @session.on("transcript")
    async def _on_transcript(msg):
        text = msg.text.lower().strip()
        words = text.split()

        if session.agent_is_speaking:
            if any(w in INTERRUPT_WORDS for w in words):
                await session.interrupt()
                return

            if all(w in IGNORE_WORDS for w in words):
                return

            await session.interrupt()
            return

        await session.generate_reply(instructions=text)

    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=True,
            audio_output=True,
        ),
    )

    session.generate_reply(instructions="say hello to the user in English")


if __name__ == "__main__":
    cli.run_app(server)