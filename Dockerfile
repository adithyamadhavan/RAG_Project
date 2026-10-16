FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY corpus/ ./corpus/
COPY evals/ ./evals/
COPY faiss_index.bin chunk_data.pkl ./

ENV PYTHONUNBUFFERED=1
EXPOSE 8001

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]
