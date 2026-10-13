#!/usr/bin/env bash
#SBATCH --job-name=cfg5-import-missing
#SBATCH --partition=defq
#SBATCH --account=user08
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/home/user08/cfg5-pretrain/logs/cfg5-import-missing-%j.out
#SBATCH --error=/home/user08/cfg5-pretrain/logs/cfg5-import-missing-%j.err
set -euo pipefail
export CFG5_IMPORT_DESTINATION="${1:-robot_missing_data}"
case "$CFG5_IMPORT_DESTINATION" in
  training_data|robot_missing_data) ;;
  missing_data) export CFG5_IMPORT_DESTINATION=robot_missing_data ;;
  *) echo 'Usage: sbatch import_missing_cfg5_drive.sbatch [robot_missing_data|training_data]' >&2; exit 2 ;;
esac
export CFG5_IMPORT_REPO="$HOME/cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
export CFG5_IMPORT_ZIP="$HOME/cfg5-pretrain/downloads/cfg5_missing_data_drive.zip"
mkdir -p "$HOME/cfg5-pretrain/downloads"
test -d "$CFG5_IMPORT_REPO/RH20T_cfg5"
echo "JOB_ID=$SLURM_JOB_ID"
df -h "$CFG5_IMPORT_REPO/RH20T_cfg5"
rclone backend copyid mydrive: \
  1OUXIrwCQXboVz7mfZfT9w0iPbAjEsfpT \
  "$CFG5_IMPORT_ZIP" \
  --checksum --multi-thread-streams 4 --stats 30s --stats-one-line \
  --retries 5 --low-level-retries 10 -v
echo 'DRIVE_DOWNLOAD_OK'
python3 -u - <<'PY'
import json, os, re, shutil, stat, zlib, zipfile
from pathlib import Path, PurePosixPath
repo = Path(os.environ['CFG5_IMPORT_REPO']).resolve()
archive = Path(os.environ['CFG5_IMPORT_ZIP'])
mode = os.environ['CFG5_IMPORT_DESTINATION']
target = repo / 'RH20T_cfg5' / mode
report_target = repo / ('outputs/raw_import_%s_reports' % mode)
pattern = re.compile(r'task_\d{4}_user_\d{4}_scene_\d{4}_cfg_0005')
existing_episodes = {p.name for p in target.rglob('task_*_cfg_0005') if p.is_dir() and pattern.fullmatch(p.name)} if target.exists() else set()
if not zipfile.is_zipfile(archive):
    raise ValueError('File Drive tải về không phải ZIP; dừng, chưa giải nén')
def mapped(info):
    name = info.filename.replace('\\', '/')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or any(':' in x for x in p.parts):
        raise ValueError('Đường dẫn ZIP không an toàn: ' + name)
    if '__MACOSX' in p.parts:
        return None
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError('ZIP chứa symlink: ' + name)
    if info.is_dir():
        return None
    parts = p.parts
    if 'robot_missing_data' in parts:
        parts = parts[parts.index('robot_missing_data') + 1:]
    elif 'missing_data' in parts:
        parts = parts[parts.index('missing_data') + 1:]
    if parts and parts[0] == 'robot':
        parts = parts[1:]
    if not parts or info.is_dir():
        return None
    if parts[0] not in ('README_VI.md', 'README.md') and not re.fullmatch(r'task_\d{4}', parts[0]):
        raise ValueError('ZIP này phải chỉ chứa robot/task_XXXX; thành phần không nhận diện: ' + name)
    # Support the older per-task robot_incomplete wrapper, too.
    if len(parts) >= 2 and parts[1] == 'robot_incomplete':
        parts = (parts[0],) + parts[2:]
    if any(re.fullmatch(r'task_\d{4}_user_\d{4}_scene_\d{4}_cfg_0005_human(?:_2)?', x) for x in parts):
        raise ValueError('Có human episode trong ZIP robot: ' + name)
    # Keep source audit documents in outputs so they do not collide with existing task reports.
    has_episode = any(pattern.fullmatch(part) for part in parts)
    destination_root = target if has_episode else report_target
    dest = destination_root.joinpath(*parts)
    if destination_root.resolve() not in dest.resolve().parents:
        raise ValueError('Đích nằm ngoài folder đã chỉ định: ' + name)
    return dest
with zipfile.ZipFile(archive) as z:
    entries = []
    seen = set()
    for info in z.infolist():
        dest = mapped(info)
        if dest is None:
            continue
        if str(dest) in seen:
            raise ValueError('Nhiều entry ZIP cùng đích: ' + str(dest))
        seen.add(str(dest)); entries.append((info, dest))
    if not entries:
        raise ValueError('ZIP không có dữ liệu robot')
    incoming_names = {part for info, _ in entries for part in PurePosixPath(info.filename.replace('\\', '/')).parts if pattern.fullmatch(part)}
    if not incoming_names:
        raise ValueError('ZIP không chứa robot episode CFG5')
    required = sum(info.file_size for info, _ in entries)
    print('ZIP_CHECK_OK files=%d uncompressed_GiB=%.3f' % (len(entries), required / 1024**3), flush=True)
    if shutil.disk_usage(repo).free < required + 1024**3:
        raise OSError('Filesystem không đủ dung lượng để giải nén')
    skipped = 0
    for i, (info, dest) in enumerate(entries, 1):
        if dest.exists():
            if not dest.is_file() or dest.stat().st_size != info.file_size:
                raise ValueError('File cũ khác file ZIP, dừng không ghi đè: ' + str(dest))
            crc = 0
            with dest.open('rb') as f:
                for block in iter(lambda: f.read(4 * 1024**2), b''):
                    crc = zlib.crc32(block, crc)
            if crc & 0xffffffff != info.CRC:
                raise ValueError('CRC file cũ khác ZIP, dừng không ghi đè: ' + str(dest))
            skipped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + '.cfg5-import-partial')
            if partial.exists():
                raise ValueError('Có file import dở; cần kiểm tra trước: ' + str(partial))
            with z.open(info) as source, partial.open('xb') as output:
                shutil.copyfileobj(source, output, 4 * 1024**2)
            # ZipFile verifies CRC when the member is read to EOF.
            if partial.stat().st_size != info.file_size:
                raise ValueError('Giải nén sai dung lượng: ' + str(partial))
            partial.rename(dest)
        if i % 500 == 0 or i == len(entries):
            print('EXTRACT %d/%d skipped_identical=%d' % (i, len(entries), skipped), flush=True)
episodes = [p for p in target.rglob('task_*_cfg_0005*') if p.is_dir() and pattern.fullmatch(p.name)]
summary = dict(destination=str(target), archive=str(archive), robot_episodes=len(episodes),
               destination_mode=mode, incoming_robot_episodes=len(incoming_names),
               existing_robot_episodes=len(existing_episodes),
               newly_added_robot_episodes=len(incoming_names - existing_episodes),
               source_reports_directory=str(report_target),
               scope='Robot archive only',
               note='Import raw only; existing training split/config is unchanged. Re-audit before training.')
(repo / 'outputs').mkdir(exist_ok=True)
(repo / ('outputs/robot_drive_import_%s.json' % mode)).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
print('MISSING_DATA_IMPORT_OK', json.dumps(summary, ensure_ascii=False), flush=True)
PY
echo 'CFG5_MISSING_IMPORT_JOB_OK'
