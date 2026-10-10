#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-train
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-train-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-train-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p outputs/pick_place_finetune
exec 9> outputs/pick_place_finetune/.train.lock
flock -n 9 || { echo "Có tiến trình train fine-tune khác; không chạy trùng" >&2; exit 1; }
./.venv-data/bin/python -u scripts/prepare_finetune_data.py check
if [[ -f outputs/pick_place_finetune/latest.pt ]]; then
  ./.venv-data/bin/python -u scripts/train.py \
    --config configs/act/pick_place_finetune.json --device cuda \
    --resume outputs/pick_place_finetune/latest.pt
else
  ./.venv-data/bin/python -u scripts/train.py \
    --config configs/act/pick_place_finetune.json --device cuda \
    --init-checkpoint outputs/act_pretrain_bs8_to80/best.pt
fi
echo FINETUNE_TRAIN_JOB_OK
