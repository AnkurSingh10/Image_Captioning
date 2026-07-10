FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TFHUB_CACHE_DIR=/tmp/tfhub_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/tfhub_cache /app/app/uploads

COPY Requirements.txt ./Requirements.txt
RUN python -m pip install --upgrade pip "setuptools<81" wheel \
    && pip install -r Requirements.txt

COPY app ./app
COPY src ./src
COPY data/caption_model.h5 ./data/caption_model.h5
COPY data/tokenizer.pkl ./data/tokenizer.pkl
COPY data/config.json ./data/config.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
