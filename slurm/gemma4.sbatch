#!/bin/bash -l
# Gemma 4 31B on the 96 test rows, 8 shards. Resumable. Then run finalize by hand.
#   GEMMA_MODEL_PATH=/path/to/gemma-4-31B-it sbatch --export=ALL,GEMMA_REPLICATE=0 slurm/gemma4.sbatch
#SBATCH -J gemma4
#SBATCH -p gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-7
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-gemma4/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

python gemma_eval.py --replicate "${GEMMA_REPLICATE:-0}" run \
    --model-path "$GEMMA_MODEL_PATH" --shard-count 8 --shard-index "$SLURM_ARRAY_TASK_ID"
