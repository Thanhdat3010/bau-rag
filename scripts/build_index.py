import json
import os
import sys

# Đảm bảo import được module app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def main():
    # Cấu hình stdout sang UTF-8 để in tiếng Việt trên console Windows không bị lỗi
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print(f"Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Đang đọc dữ liệu từ data/tu_dien_nam_bo.jsonl...")
    documents = []
    jsonl_path = "data/tu_dien_nam_bo.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {jsonl_path}")
        sys.exit(1)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as e:
                print(f"Bỏ qua dòng lỗi JSON: {e}")
                continue
            
            # Tạo text để embedding
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
            documents.append(doc)

    print(f"Đọc thành công {len(documents)} mục từ điển. Đang lập chỉ mục và lưu vào ChromaDB tại {CHROMA_DB_PATH} (Quá trình này có thể mất vài phút)...")
    
    # Tạo ChromaDB persistent
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"Hoàn thành! Đã index {len(documents)} entries vào ChromaDB.")

if __name__ == "__main__":
    main()
