#!/usr/bin/env bash
#SBATCH --job-name=cfg5-ft-pre16
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-ft-pre16-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-ft-pre16-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
mkdir -p outputs/pick_place_finetune
exec 9>outputs/pick_place_finetune/preprocess.lock
flock -n 9 || { echo "Đã có job check/preprocess khác" >&2; exit 1; }
df -h "$HOME" .
if command -v quota >/dev/null 2>&1; then quota -s || true; fi
./.venv-data/bin/python -u scripts/preprocess_finetune_parallel.py --workers 16 --splits train val
echo FINETUNE_PREPROCESS16_JOB_OK
