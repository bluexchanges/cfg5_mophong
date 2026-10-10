#!/usr/bin/env bash
#SBATCH --job-name=cfg5-act-test
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-act-test-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-act-test-%j.err

set -euo pipefail
BASE="$HOME/cfg5-pretrain"
REPO="$BASE/edabk-physical-ai-robot/04-Src/software/cfg5_act"
BUNDLE="${CFG5_TEST_BUNDLE:?Launcher must export CFG5_TEST_BUNDLE}"
cd "$REPO"
mkdir -p "$BASE/tmp"
export TMPDIR="$BASE/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9> "$BASE/pretrain_test.lock"
flock -n 9 || { echo "Đã có job test đang chạy" >&2; exit 1; }
exec 8> "$BASE/act_fast40.lock"
flock -n 8 || { echo "Training vẫn giữ lock; kiểm job trước khi test" >&2; exit 1; }
TEST_ARGS=()
if [[ "${CFG5_TEST_STOPPED_EARLY:-0}" == 1 ]]; then
    TEST_ARGS+=(--stopped-early)
fi
./.venv-data/bin/python -u "$BUNDLE/scripts/evaluate_pretrain_test.py" --repo "$REPO" "${TEST_ARGS[@]}"
