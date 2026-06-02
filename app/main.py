import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import search_relevant_words, build_prompt
from app.groq_client import call_groq

app = FastAPI(title="BẬU RAG API")

# Cấu hình CORS để cho phép frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thay đổi thành tên miền cụ thể của frontend trong môi trường production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ConvertRequest(BaseModel):
    text: str


class ConvertResponse(BaseModel):
    original: str
    converted: str
    relevant_words: list[dict] = []


@app.post("/api/convert", response_model=ConvertResponse)
async def convert_text(req: ConvertRequest):
    # 1. Tìm từ liên quan nhất từ ChromaDB
    relevant_words = search_relevant_words(req.text)
    
    # 2. Xây dựng System Prompt kèm danh sách từ liên quan
    system_prompt = build_prompt(req.text, relevant_words)
    
    # 3. Gọi mô hình Groq LLM qua LangChain
    result_str = await call_groq(system_prompt, req.text)
    
    # Parse JSON từ LLM
    try:
        import json
        llm_data = json.loads(result_str)
        converted_text = llm_data.get("converted", result_str)
        used_word_list = llm_data.get("used_words", [])
    except Exception:
        # Fallback nếu parse lỗi
        converted_text = result_str
        used_word_list = []
    
    # So khớp danh sách từ LLM trả về với metadata chi tiết trong relevant_words
    used_words_metadata = []
    used_word_set = {w.lower().strip() for w in used_word_list}
    
    for w in relevant_words:
        if w["tu"].lower().strip() in used_word_set:
            used_words_metadata.append(w)
            
    return ConvertResponse(
        original=req.text,
        converted=converted_text,
        relevant_words=used_words_metadata
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
