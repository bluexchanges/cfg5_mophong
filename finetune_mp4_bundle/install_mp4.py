"""Cài bản MP4 đã kiểm checksum; sao lưu file thay đổi, không sửa dataset/split."""
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
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalized else data).hexdigest()


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def plan_install(bundle, repo, manifest):
    for relative, expected in manifest["prerequisites"].items():
        path = (repo / relative).resolve()
        if not inside(path, repo) or not path.is_file() or checksum(path, True) != expected:
            raise ValueError(f"Prerequisite server khác bản đã kiểm: {relative}; chưa sửa file nào")
    operations = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Bundle path không hợp lệ")
        source, target = (bundle / relative).resolve(), (repo / relative).resolve()
        if not inside(source, bundle) or not inside(target, repo):
            raise ValueError("Bundle path ra ngoài root")
        if checksum(source) != row["sha256"]:
            raise ValueError(f"Checksum bundle sai: {relative}")
        if target.exists():
            current = checksum(target, True)
            if current == row["normalized_sha256"]:
                continue
            if current != row.get("previous_normalized_sha256"):
                raise ValueError(f"Code server khác bản đã kiểm: {relative}; không ghi đè custom code")
        elif row.get("previous_normalized_sha256"):
            raise FileNotFoundError(f"Thiếu baseline cần thay: {target}")
        operations.append((relative, source, target))
    return operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo, bundle = Path(args.repo).resolve(), Path(__file__).resolve().parent
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    operations = plan_install(bundle, repo, manifest)
    if shutil.which("squeue"):
        result = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T"],
                                check=True, capture_output=True, text=True)
        active = [line for line in result.stdout.splitlines()
                  if any(word in line.lower() for word in ("cfg5-act", "cfg5-ft", "pick_place"))]
        if active:
            raise RuntimeError(f"Chờ job ACT/fine-tune kết thúc trước khi thay code: {active}")
    print("File cần cài:", [str(relative) for relative, _, _ in operations], flush=True)
    if args.dry_run:
        print("MP4_INSTALL_DRY_RUN_OK")
        return
    backup = repo / "outputs/install_backups" / datetime.now(timezone.utc).strftime("mp4_%Y%m%dT%H%M%S%fZ")
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
            if temporary.exists():
                temporary.unlink()
    print(f"MP4_BUNDLE_INSTALLED; backups={backup}")


if __name__ == "__main__":
    main()
