#!/usr/bin/env bash
#SBATCH --job-name=cfg5-check-alloc16
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/check-alloc16-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/check-alloc16-%j.err

set -euo pipefail
hostname
echo "JOB_ID=$SLURM_JOB_ID"
echo "REQUEST: 1 node, 1 task, 16 Slurm CPUs, 128G RAM, 1 GPU"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} MiB"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
scontrol show job "$SLURM_JOB_ID"

# The batch script already runs on the allocated compute node.
# Direct Python avoids creating an srun step when compute-node DNS is unavailable.
python3 - <<'PY'
import json
import os
from pathlib import Path

allowed = sorted(os.sched_getaffinity(0))
physical = set()
for cpu in allowed:
    base = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
    try:
        physical.add(((base / 'physical_package_id').read_text().strip(),
                      (base / 'core_id').read_text().strip()))
    except OSError:
        pass
print('BATCH_AFFINITY_LOGICAL_CPU_COUNT=' + str(len(allowed)))
print('BATCH_AFFINITY_LOGICAL_CPU_IDS=' + json.dumps(allowed))
print('BATCH_AFFINITY_PHYSICAL_CORE_COUNT=' + (str(len(physical)) if physical else 'unknown'))
print('NOTE: batch affinity can exceed the allocated CPU count if cluster binding is not enforced; use Slurm NumCPUs/AllocCPUS for allocation.')
print('CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'))

# Report enforced RAM limits when exposed by Linux cgroups, without allocating RAM.
limit_files = []
for line in Path('/proc/self/cgroup').read_text().splitlines():
    _, controllers, relative = line.split(':', 2)
    if not controllers:
        folder = Path('/sys/fs/cgroup') / relative.lstrip('/')
        limit_files.append(folder / 'memory.max')
    elif 'memory' in controllers.split(','):
        folder = Path('/sys/fs/cgroup/memory') / relative.lstrip('/')
        limit_files.append(folder / 'memory.limit_in_bytes')
print('CGROUP_RAM_LIMITS_BEGIN')
seen = set()
for leaf in limit_files:
    current = leaf.parent
    # The effective limit can be applied on the parent job cgroup.
    for _ in range(12):
        file = current / leaf.name
        if file not in seen:
            seen.add(file)
            try:
                value = file.read_text().strip()
                gib = (' %.2f GiB' % (int(value) / 2**30)) if value.isdigit() else ''
                print(str(file) + ': ' + value + gib)
            except OSError:
                pass
        if current == Path('/sys/fs/cgroup') or current == current.parent:
            break
        current = current.parent
print('CGROUP_RAM_LIMITS_END')
print('ALLOCATION_BATCH_OK')
PY

echo ALLOCATION_CHECK_OK
