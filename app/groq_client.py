import asyncio
import logging
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

llm_json = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.0,
    max_tokens=2048,
    model_kwargs={"response_format": {"type": "json_object"}}
)


async def call_groq(system_prompt: str, user_message: str) -> str:
    """Gọi Groq LLM qua LangChain với cơ chế tự động thử lại khi gặp Rate Limit."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    max_retries = 3
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            response = await llm_json.ainvoke(messages)
            return response.content
        except Exception as e:
            # Kiểm tra xem có phải là RateLimitError (429) không
            error_msg = str(e)
            is_rate_limit = "429" in error_msg or "rate_limit" in error_msg.lower() or "RateLimitError" in type(e).__name__
            
            if is_rate_limit and attempt < max_retries - 1:
                # Trích xuất thời gian chờ từ thông điệp lỗi của Groq (nếu có)
                wait_time = base_delay * (attempt + 1)
                match = re.search(r"try again in ([\d\.]+)s", error_msg)
                if match:
                    try:
                        wait_time = float(match.group(1)) + 0.5  # Thêm 0.5s đệm an toàn
                    except ValueError:
                        pass
                
                logger.warning(f"Rate limit hit. Thử lại sau {wait_time:.2f} giây... (Lần thử {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                # Nếu đã hết lượt thử hoặc không phải lỗi Rate Limit, raise lỗi ra ngoài
                raise e


