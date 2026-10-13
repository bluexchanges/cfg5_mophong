#!/usr/bin/env bash
#SBATCH --job-name=cfg5-all-mcr-cache
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-all-mcr-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-all-mcr-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
PY=./.venv-data/bin/python
MODE=${1:-pilot}
TARGET=${2:?Specify target: tcp or joint}
case "$MODE" in pilot|build) ;; *) echo 'Usage: sbatch cfg5_robot_all_mcr.sbatch pilot|build' >&2; exit 2;; esac
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK RAM_MiB=$SLURM_MEM_PER_NODE MODE=$MODE"
PREPARE=(--target "$TARGET")
if [[ "$TARGET" == joint ]]; then
  PREPARE+=(--repair-archive "$HOME/cfg5-pretrain/downloads/simulation.zip")
fi
"$PY" -u scripts/prepare_cfg5_masked_mcr.py "${PREPARE[@]}"
"$PY" -u scripts/download_mcr.py
"$PY" -u scripts/prepare_mcr_cache.py "$MODE" \
  --base-config configs/act/cfg5_robot_all_masked.json \
  --cache-root data/features/cfg5_robot_all_masked_mcr \
  --batch-size 8 --decoder-workers 8 --pilot-episodes 3 --home-quota-tb 1.8
echo CFG5_ALL_ROBOT_MCR_JOB_OK
