#!/usr/bin/env bash
#SBATCH --job-name=cfg5-e5-probe
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-e5-probe-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-e5-probe-%j.err

set -euo pipefail
cd "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENCV_FFMPEG_THREADS=1
export MUJOCO_GL=egl
PY=./.venv-data/bin/python
echo "JOB_ID=$SLURM_JOB_ID CPU=$SLURM_CPUS_PER_TASK RAM_MiB=$SLURM_MEM_PER_NODE"
"$PY" - <<'PY'
import torch, mujoco, cv2
assert torch.cuda.device_count() == 1, 'Cần một GPU được Slurm cấp'
assert mujoco.__version__ == '3.14.0', 'Dùng MuJoCo 3.14.0 như demo'
print('GPU:', torch.cuda.get_device_name(0), 'MuJoCo:', mujoco.__version__, 'OpenCV:', cv2.__version__)
PY
# Bundle includes pinned Panda assets, so this job needs no asset download.
"$PY" -u scripts/probe_epoch5.py --checkpoint outputs/act_pretrain/best.pt \
  --expected-epoch 5 --split val --episodes 10 --mode both --device cuda \
  --max-steps 600 --output "outputs/epoch5_transfer_probe_$SLURM_JOB_ID"
echo EPOCH5_PROBE_JOB_OK
