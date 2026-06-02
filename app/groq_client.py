from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import GROQ_API_KEY, GROQ_MODEL


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.1,
    max_tokens=1024,
    model_kwargs={"response_format": {"type": "json_object"}}
)


async def call_groq(system_prompt: str, user_message: str) -> str:
    """Gọi Groq LLM qua LangChain."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = await llm.ainvoke(messages)
    return response.content
