#!/usr/bin/env bash
#SBATCH --job-name=cfg5-disk-audit
#SBATCH --account=user08
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:30:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/disk-audit-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/disk-audit-%j.err
set -euo pipefail
du -h --max-depth=2 "$HOME" | sort -h
du -h --max-depth=1 "$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act/data" | sort -h
echo DISK_AUDIT_OK
