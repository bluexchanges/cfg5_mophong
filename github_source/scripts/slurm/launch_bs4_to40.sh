#!/usr/bin/env bash
# Run this on the head node after uploading/extracting the small source bundle.
set -euo pipefail
REPO="$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
BUNDLE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$REPO"
ACTIVE=$(squeue -u "$USER" -h -o '%i %j %t' | awk '$2 ~ /^cfg5-act/ {print}')
if [[ -n "$ACTIVE" ]]; then
    printf 'Có job ACT đang chạy/chờ, chưa gửi thêm:\n%s\n' "$ACTIVE"
    exit 1
fi
test -f outputs/act_pretrain/latest.pt
test -x .venv-data/bin/python
mkdir -p "$HOME/cfg5-pretrain/logs" "$HOME/cfg5-pretrain/tmp"
if [[ "$BUNDLE" != "$REPO" ]]; then
    BACKUP=$(mktemp -d "$HOME/cfg5-pretrain/trainer-backup-XXXXXXXX")
    for FILE in src/training/pretrain.py src/training/fork_run.py scripts/fork_training_run.py; do
        mkdir -p "$BACKUP/$(dirname "$FILE")" "$(dirname "$FILE")"
        if [[ -f "$FILE" ]]; then cp -p -- "$FILE" "$BACKUP/$FILE"; fi
    done
    # Validate source bytes and install atomically; do not import torch on the head node.
    python3 - "$BUNDLE" "$REPO" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
files = ["src/training/pretrain.py", "src/training/fork_run.py", "scripts/fork_training_run.py"]
payloads = {name: (source / name).read_bytes() for name in files}
for name, payload in payloads.items():
    compile(payload, name, "exec")
for name, payload in payloads.items():
    target = destination / name
    temporary = target.with_name(target.name + ".install-partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        installed = target.read_bytes()
        assert installed == payload, f"File cài không khớp nguồn: {target}"
        compile(installed, name, "exec")
        print(f"Đã kiểm file: {name}; SHA256={hashlib.sha256(installed).hexdigest()}")
    finally:
        temporary.unlink(missing_ok=True)
PY
    echo "Backup code cũ: $BACKUP"
fi
cat > "$HOME/cfg5-pretrain/act_bs4_to40.sbatch" <<'BATCH'
#!/usr/bin/env bash
#SBATCH --job-name=cfg5-act-bs4-to40
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-act-bs4-to40-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-act-bs4-to40-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
mkdir -p "$HOME/cfg5-pretrain/tmp"
export TMPDIR="$HOME/cfg5-pretrain/tmp"
exec 9> "$HOME/cfg5-pretrain/act_bs4_to40.lock"
flock -n 9 || { echo "Một lượt batch 4 khác đang chạy" >&2; exit 1; }
ACTIVE=$(squeue -u "$USER" -h -o '%i %j' | awk -v id="$SLURM_JOB_ID" '$1 != id && $2 ~ /^cfg5-act/ {print}')
if [[ -n "$ACTIVE" ]]; then
    printf 'Có job ACT khác; dừng để tránh ghi cùng checkpoint:\n%s\n' "$ACTIVE" >&2
    exit 1
fi
PY="./.venv-data/bin/python"
CONFIG="configs/act/cfg5_all_pretrain_bs4_40epoch.json"
CHILD="outputs/act_pretrain_bs4/latest.pt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u - <<'PY'
import json
from pathlib import Path
import torch

assert torch.cuda.is_available(), "PyTorch không thấy CUDA"
child = Path("outputs/act_pretrain_bs4/latest.pt")
source = child if child.exists() else Path("outputs/act_pretrain/latest.pt")
saved = torch.load(source, map_location="cpu", weights_only=True)
assert saved["kind"] == "train_state" and saved["smoke"] is False
cfg = dict(saved["metadata"]["run_config"])
if child.exists():
    assert cfg["batch_size"] == 4 and cfg["output_dir"] == "outputs/act_pretrain_bs4"
assert saved["epoch"] + 1 < 40, "Checkpoint đã đạt 40 epoch; không cần gửi lại"
cfg.update(epochs=40, batch_size=4, output_dir="outputs/act_pretrain_bs4")
Path("configs/act/cfg5_all_pretrain_bs4_40epoch.json").write_text(
    json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print(f"GPU: {torch.cuda.get_device_name(0)}; nguồn={source}; "
      f"đã lưu {saved['epoch'] + 1} epoch; tiếp epoch {saved['epoch'] + 2}; "
      f"batch_size=4; mục tiêu tổng 40 epoch", flush=True)
PY
if [[ ! -f "$CHILD" ]]; then
    "$PY" -u scripts/fork_training_run.py --config "$CONFIG" \
        --from-checkpoint outputs/act_pretrain/latest.pt
fi
"$PY" -u scripts/train.py --config "$CONFIG" --device cuda --resume "$CHILD"
echo "ACT_BS4_TO40_OK"
BATCH
SUBMITTED=$(sbatch --parsable "$HOME/cfg5-pretrain/act_bs4_to40.sbatch")
JOB_ID=${SUBMITTED%%;*}
printf '%s\n' "$JOB_ID" > "$HOME/cfg5-pretrain/act_bs4_to40.jobid"
echo "Submitted batch job $JOB_ID"
squeue -j "$JOB_ID" -o '%.18i %.22j %.2t %.10M %R'
echo "Xem log: tail -n 15 \"\$HOME/cfg5-pretrain/logs/cfg5-act-bs4-to40-$JOB_ID.out\""
