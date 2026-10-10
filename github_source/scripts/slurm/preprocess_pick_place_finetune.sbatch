#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-preprocess
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-preprocess-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-preprocess-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./.venv-data/bin/python -u scripts/prepare_finetune_data.py preprocess \
  --splits train val --workers 4 --resume
./.venv-data/bin/python -u scripts/prepare_finetune_data.py check
echo FINETUNE_PREPROCESS_JOB_OK
