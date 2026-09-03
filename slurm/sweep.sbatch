#!/bin/bash -l
# Validation sweep. Idempotent: finished runs are skipped, so just resubmit
# after preemption. Pre-download both pinned checkpoints first (jobs run offline).
#   sbatch slurm/sweep.sbatch --task absa
#   sbatch slurm/sweep.sbatch --task trad
#   sbatch slurm/sweep.sbatch              # after select: augmented_weighted
# Set partition/account/GRES for your cluster; use one bf16 GPU type throughout.
#SBATCH -J sweep
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

python run_models.py sweep "$@"

echo "=== Ended $(date) ==="
