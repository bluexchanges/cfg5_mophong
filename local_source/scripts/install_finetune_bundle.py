"""Install the small fine-tune bundle after checking known source versions; preserve backups."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def checksum(path, normalized=False):
    data = Path(path).read_bytes()
    if normalized:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def inside_root(path, root):
    """Path.is_relative_to is unavailable on the headnode's older system Python."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def plan_install(bundle, repo, manifest):
    operations = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Bundle path không hợp lệ")
        source, target = (bundle / relative).resolve(), (repo / relative).resolve()
        if not inside_root(source, bundle) or not inside_root(target, repo):
            raise ValueError("Bundle path đi ra ngoài root")
        if checksum(source) != row["sha256"]:
            raise ValueError(f"Checksum bundle sai: {relative}")
        if target.exists():
            current = checksum(target, normalized=True)
            if current == row["normalized_sha256"]:
                continue
            if not row.get("previous_normalized_sha256") or current != row["previous_normalized_sha256"]:
                raise ValueError(f"Code server khác bản đã kiểm: {relative}; chưa sửa file nào. Gửi output để áp patch phù hợp.")
        operations.append((relative, source, target))
    return operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo, bundle = Path(args.repo).resolve(), Path(__file__).resolve().parent
    if not (repo / "scripts/train.py").is_file() or not (repo / "src/data/common.py").is_file():
        raise ValueError("--repo phải trỏ tới software/cfg5_act")
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    operations = plan_install(bundle, repo, manifest)
    if shutil.which("squeue"):
        result = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T"],
                                check=True, capture_output=True, text=True)
        running = [line for line in result.stdout.splitlines()
                   if any(word in line.lower() for word in ("cfg5-act", "cfg5-ft", "pick_place"))]
        if running:
            raise RuntimeError(f"Có job ACT/fine-tune; chờ kết thúc trước khi cài code: {running}")
    print("File cần cài:", [str(relative) for relative, _, _ in operations])
    if args.dry_run:
        print("DRY_RUN_OK; chưa sửa file nào")
        return
    backup = repo / "outputs/install_backups" / datetime.now(timezone.utc).strftime("finetune_%Y%m%dT%H%M%S%fZ")
    for relative, source, target in operations:
        if target.exists():
            previous = backup / relative
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, previous)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".install_partial")
        try:
            temporary.write_bytes(source.read_bytes())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"FINETUNE_BUNDLE_INSTALLED; backups={backup}")


if __name__ == "__main__":
    main()
