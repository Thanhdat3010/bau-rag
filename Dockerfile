# Sử dụng Python base image nhẹ
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /code

# Thiết lập biến môi trường để cache Hugging Face models ở thư mục được phép ghi
ENV HF_HOME=/tmp/hf_cache

# Copy file requirements.txt và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Tải trước model embedding trong quá trình build để tối ưu thời gian khởi động (cold start)
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy toàn bộ mã nguồn vào container
COPY . .

# Không chạy build_index.py trong container vì chúng ta sẽ tải thư mục chroma_db sẵn từ local lên.
# Nếu bạn không muốn commit chroma_db, hãy bỏ dấu comment ở dòng dưới để tự build khi deploy:
# RUN python scripts/build_index.py

# Expose cổng mặc định của Hugging Face Spaces (7860)
EXPOSE 7860

# Khởi chạy FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
