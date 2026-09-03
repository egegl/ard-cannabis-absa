#!/bin/bash -l
# Final test runs. Needs `run_models.py select` first.
#   sbatch slurm/test.sbatch [--task absa|trad]
#SBATCH -J test
#SBATCH -p gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
echo "=== Started $(date) on $(hostname) (restarts: ${SLURM_RESTART_COUNT:-0}) ==="
nvidia-smi

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

python run_models.py test "$@"

echo "=== Ended $(date) ==="
