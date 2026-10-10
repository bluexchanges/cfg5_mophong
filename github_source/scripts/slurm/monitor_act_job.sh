#!/usr/bin/env bash
# Observe the active batch-4 ACT job and its worker children for about 30 seconds.
set -euo pipefail
J=${1:-}
if [[ -z "$J" ]]; then
    J=$(squeue -u "$USER" -h -o '%i %j %t' |
        awk '$2 ~ /^cfg5-act-bs4/ && $3 == "R" {print $1; exit}')
fi
if [[ -z "$J" ]]; then
    echo "Chưa có job ACT batch 4 đang R"
    exit 0
fi
srun --mpi=none --jobid="$J" --overlap --exact -N1 -n1 -c1 bash -s <<'BASH'
PID=""
for P in $(scontrol listpids "$SLURM_JOB_ID" | awk '$1 ~ /^[0-9]+$/ {print $1}'); do
    ARGS=$(ps -p "$P" -o args=)
    if [[ "$ARGS" == *python*scripts/train.py* ]]; then PID="$P"; break; fi
done
if [[ -z "$PID" ]]; then echo "Job đang chuẩn bị, chưa thấy train.py"; exit 0; fi
GPU=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader |
    awk -F', *' -v pid="$PID" '$1==pid {print $2; exit}')
if [[ -z "$GPU" ]]; then echo "Train chưa xuất hiện trên GPU"; exit 0; fi
PIDS="$PID"
for P in $(ps --ppid "$PID" -o pid=); do PIDS="$PIDS,$P"; done
echo "Train PID=$PID; CPU theo dõi gồm train và các tiến trình con: $PIDS; GPU=$GPU"
nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv -l 2 &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
top -b -d 2 -n 15 -p "$PIDS" |
    awk -v ids=",$PIDS," '$1=="PID" || index(ids, ","$1",") {print; fflush()}'
BASH
