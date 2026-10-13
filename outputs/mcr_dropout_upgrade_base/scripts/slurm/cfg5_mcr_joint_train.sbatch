#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-joint-train
#SBATCH --account=user08
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-joint-train-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-joint-train-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
PY=./.venv-data/bin/python
STAGE=${1:-frozen}
CACHE=${2:-data/features/cfg5_robot_all_mcr_features}
OUT=${3:-outputs/cfg5_robot_all_joint_mcr_frozen}
EPOCHS=${4:-5}
BATCH=${5:-16}
WORKERS=${6:-8}
INIT=${7:-}
case "$STAGE" in frozen|full) ;; *) echo 'Stage must be frozen/full' >&2; exit 2;; esac
if [[ -f "$OUT/latest.pt" ]]; then
  START=(--resume "$OUT/latest.pt")
else
  if [[ "$STAGE" == frozen && -z "$INIT" ]]; then
    START=(--from-scratch)
    "$PY" -u scripts/download_mcr.py
  else
    [[ -n "$INIT" ]] || { echo 'Full stage needs a frozen MCR joint checkpoint' >&2; exit 2; }
    START=(--init-checkpoint "$INIT")
  fi
  "$PY" -u scripts/train_mcr_act.py --cache-root "$CACHE" --stage "$STAGE" \
    "${START[@]}" --output "$OUT/smoke_job_$SLURM_JOB_ID" \
    --batch-size "$BATCH" --workers 0 --device cuda --smoke
  echo MCR_JOINT_GPU_SMOKE_OK
fi
"$PY" -u scripts/train_mcr_act.py --cache-root "$CACHE" --stage "$STAGE" \
  "${START[@]}" --output "$OUT" --epochs "$EPOCHS" --batch-size "$BATCH" --workers "$WORKERS" --device cuda
echo MCR_JOINT_TRAIN_JOB_OK
