#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-mp4-bench
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-bench-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-bench-%j.err
set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1 OPENCV_FOR_THREADS_NUM=1
mkdir -p outputs/pick_place_finetune_mp4_w16
exec 9> outputs/pick_place_finetune_mp4_w16/.train.lock
flock -n 9 || { echo 'Run nguồn vẫn đang train' >&2; exit 1; }
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK MEM_MiB=$SLURM_MEM_PER_NODE"
./.venv-data/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA gate cannot be skipped"'
./.venv-data/bin/python -m unittest tests.test_mp4_fast
./.venv-data/bin/python -u scripts/prepare_mp4_finetune.py check
./.venv-data/bin/python -u scripts/benchmark_mp4_fast.py \
  --report-dir "outputs/mp4_fast_benchmark/job_$SLURM_JOB_ID"
echo MP4_FAST_CHECK_JOB_OK
