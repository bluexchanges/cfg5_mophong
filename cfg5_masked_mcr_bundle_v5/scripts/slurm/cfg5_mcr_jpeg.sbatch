#!/usr/bin/env bash
#SBATCH --job-name=cfg5-mcr-jpeg
#SBATCH --account=user08
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-mcr-jpeg-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-mcr-jpeg-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
MODE=${1:-pilot}
CACHE=${2:-data/features/cfg5_robot_all_mcr_features}
INIT=${3:-outputs/cfg5_robot_all_joint_mcr_frozen/best.pt}
case "$MODE" in pilot|build|check) ;; *) echo 'Use pilot/build/check' >&2; exit 2;; esac
OPTIONS=()
if [[ "$MODE" == build ]]; then OPTIONS+=(--release-features); fi
echo "JOB_ID=$SLURM_JOB_ID MODE=$MODE GPU_REQUIRED=false"
./.venv-data/bin/python -u scripts/prepare_mcr_jpeg.py "$MODE" \
  --cache-root "$CACHE" --frozen-checkpoint "$INIT" "${OPTIONS[@]}" \
  --batch-size 8 --decoder-workers 8 --pilot-episodes 3 --home-quota-tb 1.8 --full-check
echo MCR_JPEG_JOB_OK
