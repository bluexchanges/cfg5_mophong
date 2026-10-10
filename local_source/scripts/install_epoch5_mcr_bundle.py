"""Cài các file mới và dependency MuJoCo đúng checksum, không ghi đè custom code."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def checksum(path, normalized=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalized else data).hexdigest()


def safe_path(root, relative):
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Path bundle không hợp lệ: {relative}")
    target = (root / part).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path ngoài root: {relative}")
    return target


def plan_install(bundle, repo, manifest):
    missing = [relative for relative in manifest["prerequisites"] if not safe_path(repo, relative).is_file()]
    if missing:
        print(f"REPO={repo}", file=sys.stderr, flush=True)
        for relative in missing:
            print(f"MISSING: {relative}", file=sys.stderr, flush=True)
        raise FileNotFoundError("Server thiếu prerequisite: " + ", ".join(missing))
    operations = []
    for row in manifest["files"]:
        source, target = safe_path(bundle, row["path"]), safe_path(repo, row["path"])
        if checksum(source) != row["sha256"]:
            raise ValueError(f"SHA bundle sai: {row['path']}")
        if target.exists():
            if row.get("install_if_missing"):
                continue  # Test-only fixture; preserve an existing server version.
            if checksum(target, row["text"]) != row["normalized_sha256"]:
                raise ValueError(f"File server khác: {row['path']}; không ghi đè, cần xem bản khác trước")
        else:
            operations.append((source, target))
    return operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle, repo = Path(__file__).resolve().parent, Path(args.repo).resolve()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    plan = plan_install(bundle, repo, manifest)
    print("File mới cần cài:", [str(target.relative_to(repo)) for _, target in plan], flush=True)
    if args.dry_run:
        print("EPOCH5_MCR_INSTALL_DRY_RUN_OK")
        return
    for source, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".install_partial")
        data = source.read_bytes()
        if target.suffix in (".py", ".sbatch", ".md", ".json"):
            data = data.replace(b"\r\n", b"\n")
        with temporary.open("xb") as handle:
            handle.write(data)
        os.replace(temporary, target)
    print("EPOCH5_MCR_BUNDLE_INSTALLED; dataset/split/checkpoint cũ được giữ nguyên", flush=True)


if __name__ == "__main__":
    main()
