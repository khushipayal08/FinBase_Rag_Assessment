FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

# System dependencies (Tesseract for OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files copy karein (bina .venv, .env, etc.)
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/ 2>/dev/null || true

# Port expose karein
EXPOSE 10000

# Backend start karein
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "10000"]