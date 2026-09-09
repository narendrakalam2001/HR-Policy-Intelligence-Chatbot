# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr — logs show up immediately
# in `docker logs` / Hugging Face Spaces build logs instead of being lost
# on a crash.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies: PyMuPDF and chromadb's HNSW extension ship manylinux
# wheels for this base image, so no compiler toolchain is needed — only
# minimal runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer) so code-only changes
# don't invalidate the (slow, torch-heavy) dependency install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY config.py ingest.py streamlit_app.py entrypoint.sh ./
COPY app/ ./app/
COPY data/ ./data/

RUN chmod +x entrypoint.sh

# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
EXPOSE 7860

# Non-root user for defense-in-depth — no reason this process needs root.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/_stcore/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
