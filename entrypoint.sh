#!/bin/bash
# entrypoint.sh
#
# Container startup logic:
#   1. If no vector store exists yet (or FORCE_REINGEST=true), run ingestion
#      against whatever PDFs are present in data/policies/.
#   2. Start the Streamlit app.
#
# Ingestion is NOT baked in at `docker build` time because the knowledge-base
# PDFs (data/policies/*.pdf) are expected to be added/updated after the image
# is built (e.g. via a mounted volume or a rebuild with new files) rather
# than assumed present at build time.

set -euo pipefail

CHROMA_DIR="${CHROMA_PERSIST_DIR:-chroma_store}"
FORCE_REINGEST="${FORCE_REINGEST:-false}"

if [ "$FORCE_REINGEST" = "true" ] || [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
    echo "[entrypoint] No existing vector store found (or FORCE_REINGEST=true) — running ingestion..."
    python ingest.py
else
    echo "[entrypoint] Existing vector store found at '$CHROMA_DIR' — skipping ingestion."
    echo "[entrypoint] Set FORCE_REINGEST=true to rebuild it."
fi

echo "[entrypoint] Starting Streamlit on port ${PORT:-7860}..."
exec streamlit run streamlit_app.py \
    --server.port="${PORT:-7860}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
