# RAG Backend — Kế Hoạch Xây Dựng

## Mục tiêu

Xây dựng một **Python API server** làm backend cho tính năng "Chuyển đổi phương ngữ Nam Bộ" trên landing page BẬU (React frontend).

**Flow tổng quan:**
```
[React Frontend] → gửi câu tiếng Việt hiện đại
       ↓
[Python API Server]
  1. Nhận câu input
  2. Embedding câu input → vector
  3. Tìm top-K từ phương ngữ liên quan trong ChromaDB (semantic search)
  4. Ghép prompt: câu gốc + các từ phương ngữ tìm được (nghĩa + ví dụ)
  5. Gọi Groq API (LLM Llama 70B) để chuyển đổi câu
  6. Trả kết quả về frontend
```

---

## Tech Stack

| Thành phần | Công nghệ | Lý do |
|---|---|---|
| Môi trường | **Conda** | Quản lý dependencies sạch, isolate Python version |
| Web framework | **FastAPI** + **Uvicorn** | Nhẹ, nhanh, async, tự tạo Swagger docs |
| RAG framework | **LangChain** | Ecosystem hoàn chỉnh, tích hợp sẵn Groq + ChromaDB |
| Embedding model | **langchain-huggingface** (`paraphrase-multilingual-MiniLM-L12-v2`) | Miễn phí, local, hiểu tiếng Việt |
| Vector store | **langchain-chroma** (ChromaDB persistent) | Đơn giản, nhúng trực tiếp, không cần server riêng |
| LLM API | **langchain-groq** (`llama-3.3-70b-versatile`) | Miễn phí, nhanh, không cần credit card |

---

## Conda Setup

### Tạo môi trường

```bash
conda create -n bau-rag python=3.11 -y
conda activate bau-rag
```

### Cài thư viện

```bash
# LangChain core + integrations
pip install langchain langchain-groq langchain-chroma langchain-huggingface

# Embedding model
pip install sentence-transformers

# ChromaDB vector store
pip install chromadb

# Web server
pip install fastapi "uvicorn[standard]"

# Utilities
pip install python-dotenv httpx
```

### Tổng hợp thư viện

| Package | Vai trò |
|---|---|
| `langchain` | Core framework RAG |
| `langchain-groq` | Tích hợp Groq LLM |
| `langchain-chroma` | Tích hợp ChromaDB vector store |
| `langchain-huggingface` | Tích hợp HuggingFace embeddings |
| `sentence-transformers` | Model embedding (dependency của langchain-huggingface) |
| `chromadb` | Vector database |
| `fastapi` | Web API framework |
| `uvicorn[standard]` | ASGI server cho FastAPI |
| `python-dotenv` | Đọc file `.env` |
| `httpx` | HTTP client async |

---

## Cấu trúc Project

```
bau-rag-backend/
├── .env                        # GROQ_API_KEY=gsk_...
├── .gitignore
├── requirements.txt            # Backup cho pip users
├── README.md
│
├── data/
│   └── tu_dien_nam_bo.jsonl    # Copy từ project React (10,876 entries)
│
├── scripts/
│   └── build_index.py          # Chạy 1 lần: đọc JSONL → tạo embeddings → lưu ChromaDB
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + endpoint POST /api/convert
│   ├── rag.py                  # Logic RAG: load vector store, search, build prompt
│   ├── groq_client.py          # LangChain ChatGroq wrapper
│   └── config.py               # Load .env, constants
│
└── chroma_db/                  # Thư mục ChromaDB persistent (auto-generated bởi build_index.py)
```

---

## Dataset — `tu_dien_nam_bo.jsonl`

- **10,876 entries**, mỗi dòng 1 JSON object
- Encoding: **UTF-8**

Cấu trúc mỗi entry:
```json
{
  "tu": "hổm rày",
  "tu_hien_nay": "",
  "nghia": "mấy ngày vừa qua",
  "pos": ["vi tu"],
  "vi_du": ["hổm rày bận quá không ghé thăm được"]
}
```

| Field | Mô tả |
|---|---|
| `tu` | Từ phương ngữ Nam Bộ |
| `tu_hien_nay` | Từ tương đương hiện đại (thường rỗng) |
| `nghia` | Nghĩa / giải thích |
| `pos` | Loại từ (danh tu, vi tu, quan ngu...) |
| `vi_du` | Danh sách câu ví dụ sử dụng |

> [!IMPORTANT]
> Copy file `tu_dien_nam_bo.jsonl` vào thư mục `data/` của project Python.

---

## Chi tiết từng file

### 1. `.env`

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. `app/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "nam_bo_dictionary"

# Dùng multilingual model vì dataset là tiếng Việt
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 20  # Số từ phương ngữ trả về khi search
```

> [!IMPORTANT]
> Dùng model **multilingual** (`paraphrase-multilingual-MiniLM-L12-v2`) thay vì `all-MiniLM-L6-v2` vì dataset hoàn toàn bằng tiếng Việt.

### 3. `scripts/build_index.py` — Chạy 1 lần để tạo vector index

**Logic:**
1. Đọc từng dòng JSONL
2. Với mỗi entry, tạo document text để embedding:
   ```
   "{tu}" nghĩa là "{nghia}". Ví dụ: "{vi_du[0]}"
   ```
3. Lưu metadata: `tu`, `tu_hien_nay`, `nghia`, `vi_du`
4. Dùng LangChain `HuggingFaceEmbeddings` + `Chroma.from_documents()` để tạo persistent vector store

**Dùng LangChain:**
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Đọc JSONL → tạo LangChain Documents
documents = []
with open("data/tu_dien_nam_bo.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        # Tạo text để embedding
        text = f'"{entry["tu"]}" nghĩa là "{entry["nghia"]}"'
        if entry.get("vi_du"):
            text += f'. Ví dụ: {entry["vi_du"][0]}'

        doc = Document(
            page_content=text,
            metadata={
                "tu": entry["tu"],
                "tu_hien_nay": entry.get("tu_hien_nay", ""),
                "nghia": entry["nghia"],
                "vi_du": json.dumps(entry.get("vi_du", []), ensure_ascii=False),
            }
        )
        documents.append(doc)

# Tạo ChromaDB persistent
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="nam_bo_dictionary"
)

print(f"Đã index {len(documents)} entries vào ChromaDB")
```

**Lưu ý:**
- Một số entry có `tu` trùng nhau nhưng nghĩa khác → giữ tất cả, mỗi nghĩa là 1 document riêng
- Chạy ~2-5 phút tuỳ máy, tạo thư mục `chroma_db/`

### 4. `app/rag.py` — Logic RAG với LangChain

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

# Load vector store (đã build sẵn)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})


def search_relevant_words(query: str) -> list[dict]:
    """Tìm top-K từ phương ngữ liên quan nhất với câu input."""
    docs = retriever.invoke(query)
    results = []
    for doc in docs:
        results.append({
            "tu": doc.metadata["tu"],
            "nghia": doc.metadata["nghia"],
            "tu_hien_nay": doc.metadata.get("tu_hien_nay", ""),
            "vi_du": doc.metadata.get("vi_du", "[]"),
        })
    return results


def build_prompt(user_sentence: str, relevant_words: list[dict]) -> str:
    """Xây dựng system prompt với context từ vector search."""
    context_lines = []
    for w in relevant_words:
        line = f'- "{w["tu"]}" = {w["nghia"]}'
        if w.get("tu_hien_nay"):
            line += f' (hiện nay nói: "{w["tu_hien_nay"]}")'
        context_lines.append(line)

    context = "\n".join(context_lines)

    return f"""Bạn là chuyên gia ngôn ngữ phương ngữ Nam Bộ (miền Nam Việt Nam).

Nhiệm vụ: Chuyển đổi câu tiếng Việt hiện đại sang cách nói phương ngữ Nam Bộ xưa.

Quy tắc:
1. Dựa vào danh sách từ phương ngữ bên dưới để thay thế từ ngữ phù hợp
2. Giữ nguyên ý nghĩa câu gốc
3. Chỉ thay đổi từ ngữ, cách diễn đạt — KHÔNG thêm nội dung mới
4. Giữ giọng văn tự nhiên, mộc mạc như người Nam Bộ nói chuyện
5. CHỈ trả về câu đã chuyển đổi, không giải thích gì thêm

Từ điển phương ngữ Nam Bộ tham khảo:
{context}"""
```

### 5. `app/groq_client.py` — LangChain ChatGroq

```python
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import GROQ_API_KEY, GROQ_MODEL


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.7,
    max_tokens=1024
)


async def call_groq(system_prompt: str, user_message: str) -> str:
    """Gọi Groq LLM qua LangChain."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = await llm.ainvoke(messages)
    return response.content
```

### 6. `app/main.py` — FastAPI server

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import search_relevant_words, build_prompt
from app.groq_client import call_groq

app = FastAPI(title="BẬU RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: đổi thành domain cụ thể
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ConvertRequest(BaseModel):
    text: str


class ConvertResponse(BaseModel):
    original: str
    converted: str


@app.post("/api/convert", response_model=ConvertResponse)
async def convert_text(req: ConvertRequest):
    # 1. Tìm từ liên quan
    relevant_words = search_relevant_words(req.text)
    # 2. Build prompt với context
    system_prompt = build_prompt(req.text, relevant_words)
    # 3. Gọi Groq LLM
    result = await call_groq(system_prompt, req.text)
    return ConvertResponse(original=req.text, converted=result)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## Chạy project

### Bước 1: Tạo conda env + cài thư viện
```bash
conda create -n bau-rag python=3.11 -y
conda activate bau-rag
pip install langchain langchain-groq langchain-chroma langchain-huggingface sentence-transformers chromadb fastapi "uvicorn[standard]" python-dotenv httpx
```

### Bước 2: Tạo file `.env` với Groq API key
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Bước 3: Build vector index (chạy 1 lần)
```bash
python scripts/build_index.py
```
Tạo thư mục `chroma_db/` chứa embeddings. Mất ~2-5 phút.

### Bước 4: Chạy server
```bash
uvicorn app.main:app --reload --port 8000
```

### Bước 5: Test
```bash
curl -X POST http://localhost:8000/api/convert -H "Content-Type: application/json" -d "{\"text\": \"Hôm nay mẹ đi chợ về thấy bố đang ngồi xem ti vi\"}"
```

Swagger docs tự động tại: `http://localhost:8000/docs`

---

## Tích hợp React Frontend

Trong `src/App.js` của project React, sửa hàm `handleConvert`:

```javascript
const handleConvert = async () => {
  if (!inputText.trim()) return;
  try {
    const res = await fetch("http://localhost:8000/api/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText.trim() })
    });
    const data = await res.json();
    setOutputText(data.converted);
  } catch (err) {
    setOutputText("Có lỗi xảy ra, vui lòng thử lại.");
  }
};
```

---

## Deploy miễn phí

| Platform | Free tier | Lưu ý |
|---|---|---|
| **Render.com** | 750 giờ/tháng | Spin down sau 15 phút idle, cold start ~30s |
| **Railway.app** | $5 credit/tháng | Đủ cho demo nhỏ |
| **Fly.io** | 3 shared VMs miễn phí | Cần Dockerfile |

> [!WARNING]
> `sentence-transformers` model download ~500MB lần đầu. Trên free tier (512MB RAM), nên pre-build `chroma_db/` local rồi commit vào repo để tránh chạy `build_index.py` trên server.

---

## Flow tổng quan

```
┌─────────────┐     POST /api/convert      ┌──────────────────────────────┐
│   React     │ ──────────────────────────► │  FastAPI Server (conda env)  │
│  Frontend   │                             │                              │
│             │ ◄────────────────────────── │  1. Nhận câu input           │
│  (Vercel/   │     { converted: "..." }    │  2. LangChain Retriever      │
│   Netlify)  │                             │     → ChromaDB search        │
└─────────────┘                             │  3. Build prompt + context   │
                                            │  4. LangChain ChatGroq       │
                                            │     → Groq API (Llama 70B)   │
                                            │  5. Return converted text    │
                                            └──────────────────────────────┘
```
