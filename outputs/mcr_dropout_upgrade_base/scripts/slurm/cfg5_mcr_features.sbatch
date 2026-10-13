#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-features
#SBATCH --account=user08
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-features-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-features-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
PY=./.venv-data/bin/python
MODE=${1:-pilot}
CACHE=${2:-data/features/cfg5_robot_all_mcr_features}
case "$MODE" in pilot|build|check|estimate) ;; *) echo 'Use pilot/build/check/estimate' >&2; exit 2;; esac
echo "JOB_ID=$SLURM_JOB_ID MODE=$MODE STORAGE=features_only reuse_existing_numeric=true"
if [[ "$MODE" == pilot || "$MODE" == build ]]; then "$PY" -u scripts/download_mcr.py; fi
"$PY" -u scripts/prepare_mcr_cache.py "$MODE" \
  --base-config configs/act/cfg5_robot_all_masked.json --cache-root "$CACHE" \
  --features-only --batch-size 8 --decoder-workers 8 --pilot-episodes 3 \
  --home-quota-tb 1.8 --full-check
echo MCR_FEATURES_JOB_OK
