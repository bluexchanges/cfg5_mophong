#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-train
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-train-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-train-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
PY=./.venv-data/bin/python
STAGE=${1:?Usage: sbatch mcr_train.sbatch frozen/full CACHE INIT OUTPUT EPOCHS [BATCH] [WORKERS]}
CACHE=${2:?Missing cache}
INIT=${3:?Missing initialization checkpoint}
OUT=${4:?Missing output}
EPOCHS=${5:?Missing epoch count}
BATCH=${6:-16}
WORKERS=${7:-8}
case "$STAGE" in frozen|full) ;; *) echo 'Stage must be frozen/full' >&2; exit 2;; esac
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK RAM_MiB=$SLURM_MEM_PER_NODE STAGE=$STAGE"
if [[ -f "$OUT/latest.pt" ]]; then
  START=(--resume "$OUT/latest.pt")
else
  START=(--init-checkpoint "$INIT")
  if [[ "$STAGE" == frozen ]]; then "$PY" -u scripts/download_mcr.py; fi
  # Gate with the same batch size. Smoke writes a distinct job-specific folder.
  "$PY" -u scripts/train_mcr_act.py --cache-root "$CACHE" --stage "$STAGE" \
    --init-checkpoint "$INIT" --output "$OUT/smoke_job_$SLURM_JOB_ID" \
    --batch-size "$BATCH" --workers 0 --device cuda --smoke
  echo MCR_GPU_SMOKE_OK
fi
"$PY" -u scripts/train_mcr_act.py --cache-root "$CACHE" --stage "$STAGE" \
  "${START[@]}" --output "$OUT" --epochs "$EPOCHS" --batch-size "$BATCH" \
  --workers "$WORKERS" --device cuda
echo MCR_TRAIN_JOB_OK
