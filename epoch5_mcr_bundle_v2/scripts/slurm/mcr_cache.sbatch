#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-cache
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-cache-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-cache-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
PY=./.venv-data/bin/python
MODE=${1:?Usage: sbatch mcr_cache.sbatch pilot/build/check/estimate BASE_CONFIG CACHE_ROOT}
BASE=${2:?Missing base config}
CACHE=${3:?Missing cache root}
EXTRA=()
if [[ ${4:-} == --full-check && "$MODE" == check ]]; then EXTRA=(--full-check); fi
if [[ -n ${4:-} && ${#EXTRA[@]} == 0 ]]; then echo 'Optional arg only: check --full-check' >&2; exit 2; fi
case "$MODE" in pilot|build|check|estimate) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK RAM_MiB=$SLURM_MEM_PER_NODE"
if [[ "$MODE" == pilot || "$MODE" == build ]]; then
  # Official pinned checkpoint, 94 MB; local copy is verified or downloaded once.
  "$PY" -u scripts/download_mcr.py
fi
"$PY" -u scripts/prepare_mcr_cache.py "$MODE" --base-config "$BASE" --cache-root "$CACHE" \
  --batch-size 8 --decoder-workers 8 --pilot-episodes 3 --home-quota-tb 1.8 "${EXTRA[@]}"
echo MCR_CACHE_JOB_OK
