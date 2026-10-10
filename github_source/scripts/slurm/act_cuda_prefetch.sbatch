#!/usr/bin/env bash
#SBATCH --job-name=cfg5-act-prefetch
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-act-prefetch-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-act-prefetch-%j.err

set -euo pipefail
BUNDLE=__BUNDLE__
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
mkdir -p "$HOME/cfg5-pretrain/tmp"
export TMPDIR="$HOME/cfg5-pretrain/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9> "$HOME/cfg5-pretrain/act_cuda_prefetch.lock"
flock -n 9 || { echo "Một job prefetch khác đang chạy" >&2; exit 1; }
ACTIVE=$(squeue -u "$USER" -h -o '%i %j' |
    awk -v id="$SLURM_JOB_ID" '$1 != id && $2 ~ /^cfg5-act/ {print}')
if [[ -n "$ACTIVE" ]]; then
    printf 'Có job ACT khác; chưa cài code/chạy:\n%s\n' "$ACTIVE" >&2
    exit 1
fi
PY="./.venv-data/bin/python"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
# Gate the bundled candidate on real CUDA before touching the installed trainer.
CFG5_PREFETCH_SOURCE="$BUNDLE/src/training/pretrain.py" PYTHONPATH="$PWD" \
    "$PY" -u "$BUNDLE/tests/test_cuda_prefetch.py"
python3 "$BUNDLE/scripts/slurm/install_worker_bundle.py" "$BUNDLE" "$PWD"
CONFIG="configs/act/cfg5_all_pretrain_prefetch_40epoch.json"
CHILD="outputs/act_pretrain_prefetch/latest.pt"
"$PY" -u - <<'PY'
import json
from pathlib import Path
import torch

child = Path("outputs/act_pretrain_prefetch/latest.pt")
source = child if child.exists() else Path("outputs/act_pretrain_bs4_w2/latest.pt")
saved = torch.load(source, map_location="cpu", weights_only=True)
assert saved["kind"] == "train_state" and saved["smoke"] is False
cfg = dict(saved["metadata"]["run_config"])
assert cfg["batch_size"] == 4 and cfg["num_workers"] == 2
assert cfg["save_every_epochs"] == 1 and saved["epoch"] + 1 < 40
if child.exists():
    assert cfg.get("cuda_prefetch") is True
    assert cfg["output_dir"] == "outputs/act_pretrain_prefetch"
cfg.update(epochs=40, cuda_prefetch=True, output_dir="outputs/act_pretrain_prefetch")
Path("configs/act/cfg5_all_pretrain_prefetch_40epoch.json").write_text(
    json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print(f"Nguồn={source}; đã lưu {saved['epoch'] + 1} epoch; CUDA prefetch bật", flush=True)
PY
if [[ ! -f "$CHILD" ]]; then
    "$PY" -u scripts/fork_training_run.py --config "$CONFIG" \
        --from-checkpoint outputs/act_pretrain_bs4_w2/latest.pt
fi
"$PY" -u scripts/train.py --config "$CONFIG" --device cuda --resume "$CHILD"
echo "ACT_CUDA_PREFETCH_OK"
