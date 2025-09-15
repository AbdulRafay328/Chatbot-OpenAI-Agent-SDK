import os
from agents import Agent , Runner , OpenAIChatCompletionsModel , AsyncOpenAI , RunConfig
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent
import dotenv 

dotenv.load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    tracing_disabled=True,
    model=llm,
    model_provider=external_client,
)

agent = Agent(name="AI assistant", instructions="help me answer questions about anything")

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Hello! I'm your AI assistant. Ask me anything!",
        author="Assistant"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    
    result = Runner.run_streamed(
        input=history,
        run_config=config,
        starting_agent=agent,
    )
    msg = cl.Message(content="", author="Assistant")  
    await msg.send()

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
