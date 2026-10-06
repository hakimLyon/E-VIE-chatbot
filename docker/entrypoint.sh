#!/bin/sh
set -e

mkdir -p "$DATA_DIR/media"

python manage.py migrate --noinput

# Idempotent: embeds docs/*.pdf only when the Chroma index is empty.
python manage.py ingest_docs || echo "WARN: ingestion failed (check CF_ACCOUNT_ID/CF_API_TOKEN); it will be retried on the first chat request"

# --preload loads the torch models once and shares them with the workers.
exec gunicorn aura_project.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers "${WEB_CONCURRENCY:-2}" \
  --threads 2 \
  --timeout 180 \
  --preload \
  --access-logfile - \
  --error-logfile -
