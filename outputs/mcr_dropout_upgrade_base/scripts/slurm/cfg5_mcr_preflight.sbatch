#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-preflight
#SBATCH --account=user08
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-preflight-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-preflight-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
MODE=${1:-check}
case "$MODE" in check|pilot|build) ;; *) echo 'Use check/pilot/build' >&2; exit 2;; esac
PY=./.venv-data/bin/python
CACHE=data/features/cfg5_robot_all_mcr_features
REPORT="outputs/mcr_checks/mcr_frozen_check_${SLURM_JOB_ID}.json"
echo "JOB_ID=$SLURM_JOB_ID MODE=$MODE CPU=$SLURM_CPUS_PER_TASK RAM_MiB=$SLURM_MEM_PER_NODE"
# The camera report survives a missing-weight/GPU/storage failure; it reads no RGB pixels.
"$PY" -u scripts/report_cfg5_cameras.py
"$PY" -u scripts/download_mcr.py
EXTRA=()
if [[ "$MODE" != check ]]; then EXTRA+=(--require-build-space); fi
"$PY" -u scripts/check_mcr_frozen.py --output "$REPORT" "${EXTRA[@]}"
if [[ "$MODE" == check ]]; then
  echo MCR_CHECK_AND_CAMERA_REPORT_JOB_OK
  exit 0
fi
# Start only after real GPU/freeze and full-build-budget checks pass.
"$PY" -u scripts/prepare_mcr_cache.py pilot \
  --base-config configs/act/cfg5_robot_all_masked.json --cache-root "$CACHE" \
  --features-only --batch-size 8 --decoder-workers 8 --pilot-episodes 3 --home-quota-tb 1.8 --full-check
if [[ "$MODE" == build ]]; then
  "$PY" -u scripts/prepare_mcr_cache.py build \
    --base-config configs/act/cfg5_robot_all_masked.json --cache-root "$CACHE" \
    --features-only --batch-size 8 --decoder-workers 8 --pilot-episodes 3 --home-quota-tb 1.8 --full-check
fi
echo MCR_FROZEN_PREPROCESS_JOB_OK
