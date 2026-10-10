#!/usr/bin/env bash
#SBATCH --job-name=cfg5-act-bs16-g2-to80
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-act-bs16-g2-to80-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-act-bs16-g2-to80-%j.err

set -euo pipefail
BUNDLE=__BUNDLE__
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
mkdir -p "$HOME/cfg5-pretrain/tmp"
export TMPDIR="$HOME/cfg5-pretrain/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9> "$HOME/cfg5-pretrain/act_fast40.lock"
flock -n 9 || { echo "Một job ACT fast khác đang chạy" >&2; exit 1; }
ACTIVE=$(squeue -u "$USER" -h -o '%i %j' |
    awk -v id="$SLURM_JOB_ID" '$1 != id && $2 ~ /^cfg5-act/ {print}')
if [[ -n "$ACTIVE" ]]; then
    printf 'Có job ACT khác; chưa cài code/chạy:\n%s\n' "$ACTIVE" >&2
    exit 1
fi
PY="./.venv-data/bin/python"
"$PY" -c 'import torch; assert torch.cuda.device_count() == 2, "Cần đúng 2 GPU"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])'
python3 "$BUNDLE/scripts/slurm/install_to80.py" "$BUNDLE" "$PWD"
PYTHONPATH="$PWD" CFG5_REQUIRE_TWO_GPU=1 "$PY" -u "$BUNDLE/tests/test_multi_gpu_training.py"
df -h /dev/shm
"$PY" -u "$BUNDLE/scripts/slurm/continue_bs16_gpu2.py" --probe
"$PY" -u "$BUNDLE/scripts/slurm/continue_bs16_gpu2.py"
