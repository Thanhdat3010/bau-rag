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
    result = await call_groq(system_prompt, req.text)
    
    # 4. Lọc những từ thực sự xuất hiện trong câu kết quả dịch (converted text)
    used_words = []
    result_lower = result.lower()
    for w in relevant_words:
        word_lower = w["tu"].lower()
        # Định nghĩa regex khớp từ nguyên vẹn hỗ trợ tiếng Việt có dấu
        pattern = rf'(?<![a-zA-Zà-ỹÀ-ỸđĐ]){re.escape(word_lower)}(?![a-zA-Zà-ỹÀ-ỸđĐ])'
        if re.search(pattern, result_lower):
            used_words.append(w)
            
    return ConvertResponse(
        original=req.text,
        converted=result,
        relevant_words=used_words
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
