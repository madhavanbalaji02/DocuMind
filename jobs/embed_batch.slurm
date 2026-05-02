#!/bin/bash
# ============================================================
#  DocuMind — Batch Embedding Job on BigRed200
# ============================================================
#
# SUBMIT:
#   sbatch --export=INPUT_DIR=/path/to/docs,COLLECTION_NAME=documind_chunks \
#          jobs/embed_batch.slurm
#
# INPUT_DIR  : directory containing PDF/TXT/CSV files to ingest
# COLLECTION_NAME : target Qdrant collection (default: documind_chunks)
#
# The job expects:
#   - QDRANT_URL         in environment or .env
#   - DATABASE_URL_ASYNC in environment or .env
#   - vLLM server running (set BIGRED200_VLLM_URL in .env)
#
# ============================================================

#SBATCH --job-name=documind-embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-}"
COLLECTION_NAME="${COLLECTION_NAME:-documind_chunks}"

if [ -z "${INPUT_DIR}" ]; then
    echo "ERROR: INPUT_DIR must be set via --export" >&2
    exit 1
fi

echo "============================================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : $(hostname)"
echo "Input directory : ${INPUT_DIR}"
echo "Collection      : ${COLLECTION_NAME}"
echo "Start time      : $(date)"
echo "============================================================"

module purge
module load python/3.11
module load cuda/12.1

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/.venv"
source "${VENV}/bin/activate"

# Load project env
set -a
# shellcheck disable=SC1091
[ -f "${PROJECT_DIR}/.env" ] && source "${PROJECT_DIR}/.env"
set +a

# Override embedding mode for this job
export EMBEDDING_MODE=bigred200
export QDRANT_COLLECTION="${COLLECTION_NAME}"

# ── GPU utilisation logger (background) ──────────────────────
nvidia-smi dmon -s u -d 30 > "logs/gpu_util_${SLURM_JOB_ID}.csv" &
GPU_MON_PID=$!

# ── Batch ingestion script ────────────────────────────────────
python - <<'PYEOF'
import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("embed_batch")


async def main():
    from src.db import connection as db
    from src.ingest.pipeline import IngestionPipeline

    await db.run_migrations()
    pipeline = IngestionPipeline()

    input_dir = Path(os.environ["INPUT_DIR"])
    extensions = {".pdf", ".txt", ".csv", ".py", ".js", ".ts", ".md"}
    files = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in extensions)
    logger.info("Found %d files to ingest", len(files))

    t_start = time.perf_counter()
    total_chunks = 0
    succeeded = 0
    failed = 0

    for i, filepath in enumerate(files, start=1):
        ext = filepath.suffix.lower().lstrip(".")
        doc_type = (
            "pdf" if ext == "pdf"
            else "csv" if ext == "csv"
            else "code" if ext in ("py", "js", "ts")
            else "text"
        )
        try:
            result = await pipeline.ingest(str(filepath), doc_type)
            total_chunks += result.chunk_count
            succeeded += 1
            elapsed = time.perf_counter() - t_start
            throughput = total_chunks / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] %s — chunks=%d embed_ms=%.0f throughput=%.1f chunks/s",
                i,
                len(files),
                filepath.name,
                result.chunk_count,
                result.embed_time_ms,
                throughput,
            )
        except Exception as exc:
            logger.error("[%d/%d] FAILED %s: %s", i, len(files), filepath.name, exc)
            failed += 1

    elapsed_total = time.perf_counter() - t_start
    logger.info(
        "Batch complete: %d succeeded, %d failed, %d total chunks in %.1fs (%.1f chunks/s)",
        succeeded,
        failed,
        total_chunks,
        elapsed_total,
        total_chunks / elapsed_total if elapsed_total > 0 else 0,
    )
    await pipeline.close()
    await db.close_pool()


asyncio.run(main())
PYEOF

# Stop GPU monitor
kill "${GPU_MON_PID}" 2>/dev/null || true

echo "============================================================"
echo "Batch embedding job complete at $(date)"
echo "============================================================"
