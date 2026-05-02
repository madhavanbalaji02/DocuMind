#!/bin/bash
# ============================================================
#  DocuMind — vLLM Embedding Server on BigRed200
# ============================================================
#
# SUBMIT:
#   sbatch jobs/vllm_server.slurm
#
# AFTER SUBMIT:
#   squeue -u $USER                # find your job and node
#   scontrol show job <JOB_ID>     # get NodeList field
#   # The node hostname appears as e.g. r003n15
#
# SSH TUNNEL (from local machine):
#   ssh -N -L 8000:<node>:8000 <user>@bigred200.indiana.edu
#
# Then set in .env:
#   BIGRED200_VLLM_URL=http://localhost:8000/v1
#
# ============================================================

#SBATCH --job-name=documind-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

set -euo pipefail

echo "============================================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time : $(date)"
echo "============================================================"

# ── Modules ──────────────────────────────────────────────────
module purge
module load python/3.11
module load cuda/12.1

# ── Virtual environment ───────────────────────────────────────
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/.venv"

if [ ! -d "${VENV}" ]; then
    echo "Creating virtualenv…"
    python -m venv "${VENV}"
    "${VENV}/bin/pip" install --upgrade pip uv
    "${VENV}/bin/uv" pip install vllm
fi

source "${VENV}/bin/activate"

# ── vLLM server ───────────────────────────────────────────────
MODEL="BAAI/bge-m3"
echo "Starting vLLM server for model: ${MODEL}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name bge-m3 \
    --task embed \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests

echo "vLLM server exited at $(date)"
