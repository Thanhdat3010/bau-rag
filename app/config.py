import os
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "nam_bo_dictionary"

# Dùng multilingual model vì dataset là tiếng Việt
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 10  # Số từ phương ngữ trả về khi search
