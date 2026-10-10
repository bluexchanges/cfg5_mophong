"""Install additive preprocessing scripts with version and SHA checks (Python 3.8+)."""
import argparse
import ast
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def checksum(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def contained(root, name):
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Bundle path không hợp lệ: " + name)
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Bundle path ra ngoài root: " + name)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve(strict=True)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    allowed = {
        "scripts/preprocess_finetune_parallel.py",
        "scripts/slurm/check_pick_place_preprocess16.sbatch",
        "scripts/slurm/preprocess_pick_place16.sbatch",
        "docs/PREPROCESS16_VI.md",
        "tests/test_parallel_finetune_preprocess.py",
    }
    entries = manifest["files"]
    if len(entries) != len(allowed) or {row["path"] for row in entries} != allowed:
        raise ValueError("Danh sách file bundle khác bản dự kiến")
    for row in manifest["prerequisites"]:
        target = contained(repo, row["path"])
        if not target.is_file() or checksum(target.read_bytes()) != row["normalized_sha256"]:
            raise ValueError("Phiên bản server khác prerequisite: " + row["path"] + "; chưa sửa file")
    locked = repo / "data/splits/pick_place_finetune_splits.json"
    if not locked.is_file():
        raise ValueError("Chưa có split fine-tune đã khóa; chạy prepare thành công trước")
    planned = []
    for row in entries:
        source = contained(bundle, row["path"])
        target = contained(repo, row["path"])
        data = source.read_bytes().replace(b"\r\n", b"\n")
        if checksum(data) != row["normalized_sha256"]:
            raise ValueError("Checksum bundle sai: " + row["path"])
        if target.suffix == ".py":
            ast.parse(data.decode("utf-8"))
        if target.exists():
            if not target.is_file() or checksum(target.read_bytes()) != row["normalized_sha256"]:
                raise ValueError("File đã tồn tại khác nội dung: " + row["path"] + "; không ghi đè")
        else:
            planned.append((target, data))
    if planned and shutil.which("squeue"):
        result = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T"],
                                check=True, capture_output=True, text=True)
        active = [line for line in result.stdout.splitlines()
                  if len(line.split("|")) >= 2 and line.split("|")[1].startswith("cfg5-ft")]
        if active:
            raise RuntimeError("Có job fine-tune/preprocess khác; chưa sửa file: " + "; ".join(active))
    for target, data in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: additive installer never overwrites a concurrent change.
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    print("PREPROCESS16_BUNDLE_INSTALLED; added=" + str(len(planned)))
    print("CPU=16, RAM=128G, workers=16, GPU not requested; split/config/raw unchanged")


if __name__ == "__main__":
    main()
