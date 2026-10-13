#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-continue
#SBATCH --account=user08
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-continue-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-continue-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
PY=./.venv-data/bin/python
CACHE=${1:-data/features/cfg5_robot_all_mcr_features}
echo "JOB_ID=$SLURM_JOB_ID STORAGE=features_only SCOPE=train,val,test SCRIPT_HOME_BUDGET=disabled"
"$PY" -u scripts/download_mcr.py
ARGS=(--base-config configs/act/cfg5_robot_all_masked.json --cache-root "$CACHE"
      --features-only --include-test --batch-size 8 --decoder-workers 8
      --pilot-episodes 3 --skip-home-budget --full-check)
"$PY" -u scripts/prepare_mcr_cache.py pilot "${ARGS[@]}"
"$PY" -u scripts/prepare_mcr_cache.py build "${ARGS[@]}"
echo MCR_FROZEN_PREPROCESS_JOB_OK
