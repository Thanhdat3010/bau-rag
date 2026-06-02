import asyncio
import sys
import os

# Đảm bảo import được module app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag import search_relevant_words, build_prompt
from app.groq_client import call_groq

async def test_conversion(text: str):
    print(f"\n[1] Câu gốc: {text}")
    print("-" * 50)
    
    print("[2] Đang tìm kiếm từ phương ngữ liên quan trong ChromaDB...")
    relevant_words = search_relevant_words(text)
    
    if not relevant_words:
        print("Không tìm thấy từ phương ngữ nào liên quan.")
    else:
        print(f"Tìm thấy {len(relevant_words)} từ liên quan. Dưới đây là top 5 từ:")
        for w in relevant_words[:5]:
            print(f"  - {w['tu']}: {w['nghia']}")
            
    print("-" * 50)
    print("[3] Đang gọi Groq LLM để chuyển đổi...")
    system_prompt = build_prompt(text, relevant_words)
    
    try:
        result = await call_groq(system_prompt, text)
        print("\n[4] KẾT QUẢ CHUYỂN ĐỔI:")
        print(f"  => {result}")
    except Exception as e:
        print(f"\n[LỖI]: Không thể gọi Groq API. Chi tiết lỗi: {e}")
        print("=> Vui lòng kiểm tra lại GROQ_API_KEY trong file .env đã chính xác chưa.")

if __name__ == "__main__":
    # Cấu hình stdout sang UTF-8 để in tiếng Việt trên console Windows không bị lỗi
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    import argparse
    parser = argparse.ArgumentParser(description="Test RAG Backend qua Terminal")
    parser.add_argument("--text", type=str, help="Câu tiếng Việt hiện đại cần chuyển đổi", 
                        default="Mấy ngày vừa qua tôi bận quá không ghé thăm cậu được.")
    args = parser.parse_args()
    
    # Chạy hàm bất đồng bộ
    asyncio.run(test_conversion(args.text))
