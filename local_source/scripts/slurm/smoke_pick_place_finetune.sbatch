#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-smoke
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-smoke-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-smoke-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./.venv-data/bin/python -u scripts/prepare_finetune_data.py preprocess \
  --splits train val --workers 2 --limit-episodes 1 --max-steps 4 \
  --output-root data/processed/pick_place_finetune_smoke --resume
./.venv-data/bin/python -u scripts/prepare_finetune_data.py check \
  --processed-root data/processed/pick_place_finetune_smoke --smoke
./.venv-data/bin/python -u scripts/train.py \
  --config configs/act/pick_place_finetune.json --device cuda \
  --init-checkpoint outputs/act_pretrain_bs8_to80/best.pt \
  --processed-root data/processed/pick_place_finetune_smoke \
  --smoke --allow-partial-smoke
echo FINETUNE_GPU_SMOKE_OK
