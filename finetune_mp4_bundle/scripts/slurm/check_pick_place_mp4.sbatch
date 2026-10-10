#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-mp4-check
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-check-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-mp4-check-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1 OPENCV_FOR_THREADS_NUM=1
mkdir -p outputs/pick_place_finetune_mp4_w16
exec 9> outputs/pick_place_finetune_mp4_w16/.train.lock
flock -n 9 || { echo "Đang có job MP4; không chạy trùng" >&2; exit 1; }
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK MEM_MiB=$SLURM_MEM_PER_NODE"
hostname
df -h "$HOME" .
quota -s || true
echo "Quota home theo tài liệu: soft 1.8 TB / hard 2 TB; df là dung lượng chung"
./.venv-data/bin/python -u scripts/prepare_mp4_finetune.py prepare --workers 16 --resume
./.venv-data/bin/python -u scripts/prepare_mp4_finetune.py check
# A fresh job-specific smoke directory permits re-running checks without deleting checkpoints.
./.venv-data/bin/python - "$SLURM_JOB_ID" <<'PY'
import sys
from src.data.common import read_json
from src.training.pretrain import write_json
run = read_json('configs/act/pick_place_finetune_mp4_smoke.json')
run['output_dir'] = 'outputs/pick_place_finetune_mp4_w16/smoke_job_' + sys.argv[1]
write_json('configs/act/pick_place_finetune_mp4_smoke_job.json', run)
PY
./.venv-data/bin/python -u scripts/train.py \
  --config configs/act/pick_place_finetune_mp4_smoke_job.json --device cuda --smoke \
  --init-checkpoint outputs/act_pretrain_bs8_to80/best.pt
echo MP4_GPU_SMOKE_OK
./.venv-data/bin/python -u scripts/prepare_mp4_finetune.py benchmark --workers 16 --batches 100
echo MP4_CHECK_JOB_OK
