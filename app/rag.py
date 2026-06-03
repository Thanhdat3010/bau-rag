import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from app.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

# 1. Load vector store (Dense Retriever)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

# 2. Build BM25 (Sparse Retriever) từ file dữ liệu JSONL
jsonl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tu_dien_nam_bo.jsonl")
bm25_documents = []
if os.path.exists(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Dùng cùng page_content với ChromaDB để đảm bảo tính nhất quán
                text = f'"{entry["tu"]}" nghĩa là "{entry["nghia"]}"'
                vi_du_list = entry.get("vi_du", [])
                if vi_du_list and len(vi_du_list) > 0 and vi_du_list[0]:
                    text += f'. Ví dụ: {vi_du_list[0]}'
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "tu": entry["tu"],
                        "tu_hien_nay": entry.get("tu_hien_nay", ""),
                        "nghia": entry["nghia"],
                        "vi_du": json.dumps(vi_du_list, ensure_ascii=False),
                        "pos": json.dumps(entry.get("pos", []), ensure_ascii=False),
                    }
                )
                bm25_documents.append(doc)
            except Exception:
                continue

if bm25_documents:
    bm25_retriever = BM25Retriever.from_documents(bm25_documents)
    bm25_retriever.k = TOP_K
else:
    # Fallback nếu không có file data
    bm25_retriever = vector_retriever

# 3. Kết hợp hai phương thức tìm kiếm bằng EnsembleRetriever (RRF)
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)



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
            "pos": doc.metadata.get("pos", "[]"),
        })
    return results


def build_prompt(user_sentence: str, relevant_words: list[dict]) -> str:
    """Xây dựng system prompt với context từ vector search."""
    context_lines = []
    for w in relevant_words:
        line = f'- "{w["tu"]}" = {w["nghia"]}'
        
        pos_list = []
        try:
            pos_list = json.loads(w.get("pos", "[]"))
        except:
            pass
        if pos_list:
            line += f' (Từ loại: {", ".join(pos_list)})'
            
        if w.get("tu_hien_nay"):
            line += f' (hiện nay nói: "{w["tu_hien_nay"]}")'
            
        vi_du_list = []
        try:
            vi_du_list = json.loads(w.get("vi_du", "[]"))
        except:
            pass
        if vi_du_list:
            vi_du_str = ", ".join([f'"{vd.strip()}"' for vd in vi_du_list if vd.strip()])
            if vi_du_str:
                line += f' (Ví dụ dùng: {vi_du_str})'
                
        context_lines.append(line)

    context = "\n".join(context_lines)

    return f"""Bạn là chuyên gia ngôn ngữ phương ngữ Nam Bộ (miền Nam Việt Nam).

Nhiệm vụ: Chuyển đổi câu tiếng Việt hiện đại sang cách nói phương ngữ Nam Bộ xưa.

Quy tắc:
1. Dựa vào danh sách từ phương ngữ bên dưới để thay thế từ ngữ phù hợp.
2. Giữ nguyên ý nghĩa và các chi tiết của câu gốc, không tự ý thay đổi các từ ngữ không có trong danh sách phương ngữ (ví dụ: "ăn ké" phải giữ nguyên hoặc thay thế bằng từ Nam Bộ tương đương nếu có, tuyệt đối không tự ý đổi thành "ăn hủ tiếu").
3. Chỉ thay đổi từ ngữ, cách diễn đạt — KHÔNG thêm bớt nội dung mới.
4. Giữ giọng văn tự nhiên, mộc mạc như người Nam Bộ nói chuyện.
5. Viết đúng chính tả tiếng Việt phương ngữ, giữ đúng các dấu thanh (sắc, huyền, hỏi, ngã, nặng) chính xác (ví dụ: "xỉu" chứ không viết thành "xiu").

Yêu cầu định dạng đầu ra:
Bạn BẮT BUỘC phải trả về một đối tượng JSON (không nằm trong thẻ markdown ```json) có cấu trúc chính xác như sau:
{{
  "converted": "câu đã chuyển đổi sang phương ngữ Nam Bộ",
  "used_words": ["từ_phương_ngữ_1", "từ_phương_ngữ_2"]
}}

Trong đó:
- `converted` là câu sau khi chuyển đổi.
- `used_words` là danh sách chứa các từ phương ngữ Nam Bộ (nằm trong cột "từ" của danh sách tham khảo dưới đây) mà bạn ĐÃ THỰC SỰ SỬ DỤNG trong câu chuyển đổi. Không đưa những từ không dùng vào đây.

Từ điển phương ngữ Nam Bộ tham khảo:
{context}"""
