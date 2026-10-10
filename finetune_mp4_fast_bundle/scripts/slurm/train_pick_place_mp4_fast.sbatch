#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-mp4-fast
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-fast-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-fast-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1 OPENCV_FOR_THREADS_NUM=1
CONFIG=configs/act/pick_place_finetune_mp4_fast.json
CHECKPOINT=outputs/pick_place_finetune_mp4_fast/latest.pt
[[ -f "$CONFIG" && -f "$CHECKPOINT" ]] || { echo 'Chưa benchmark/fork thành công' >&2; exit 1; }
exec 9> outputs/pick_place_finetune_mp4_fast/.train.lock
flock -n 9 || { echo 'Run fast đang chạy, không đăng ký trùng' >&2; exit 1; }
./.venv-data/bin/python -u scripts/prepare_mp4_finetune.py check --config "$CONFIG"
./.venv-data/bin/python -u scripts/train.py --config "$CONFIG" --device cuda --resume "$CHECKPOINT"
echo MP4_FAST_TRAIN_JOB_OK
