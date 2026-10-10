#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-mp4-w16
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-train-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-train-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1 OPENCV_FOR_THREADS_NUM=1
mkdir -p outputs/pick_place_finetune_mp4_w16
exec 9> outputs/pick_place_finetune_mp4_w16/.train.lock
flock -n 9 || { echo "Đang có job MP4; không chạy trùng" >&2; exit 1; }
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK MEM_MiB=$SLURM_MEM_PER_NODE"
hostname
./.venv-data/bin/python -u scripts/prepare_mp4_finetune.py check
if [[ -f outputs/pick_place_finetune_mp4_w16/latest.pt ]]; then
  ./.venv-data/bin/python -u scripts/train.py \
    --config configs/act/pick_place_finetune_mp4_w16.json --device cuda \
    --resume outputs/pick_place_finetune_mp4_w16/latest.pt
else
  ./.venv-data/bin/python -u scripts/train.py \
    --config configs/act/pick_place_finetune_mp4_w16.json --device cuda \
    --init-checkpoint outputs/act_pretrain_bs8_to80/best.pt
fi
echo MP4_TRAIN_JOB_OK
