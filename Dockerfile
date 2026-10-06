# E-VIE: Django + PyTorch (CPU) + RAG on Cloudflare Workers AI
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    DATA_DIR=/data

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels keep the image a few GB smaller than the CUDA ones.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the sentiment model into the image so the first request does not
# download ~700 MB from Hugging Face.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
n='nlptown/bert-base-multilingual-uncased-sentiment'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n)"

COPY . .

RUN SECRET_KEY=build DEBUG=False python manage.py collectstatic --noinput \
 && chmod +x docker/entrypoint.sh \
 && mkdir -p /data

VOLUME ["/data"]
EXPOSE 8000

# Probe often so Docker flips to "healthy" right after gunicorn is up; the long
# start period covers the first-boot PDF ingestion (~2 min) before that.
HEALTHCHECK --interval=10s --timeout=5s --start-period=240s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/ > /dev/null || exit 1

ENTRYPOINT ["docker/entrypoint.sh"]
