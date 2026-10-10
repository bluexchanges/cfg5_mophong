#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-prepare
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-prepare-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-prepare-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./.venv-data/bin/python -u scripts/prepare_finetune_data.py prepare \
  --sources "$HOME/cfg5-pretrain/downloads/finetune_data" "$HOME/cfg5-pretrain/downloads/dataset_moi" \
  --init-checkpoint outputs/act_pretrain_bs8_to80/best.pt
echo FINETUNE_PREPARE_JOB_OK
