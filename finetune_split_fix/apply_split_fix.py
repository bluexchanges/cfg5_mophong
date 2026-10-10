"""Install the group-preserving fine-tune split repair (Python 3.8+)."""
import argparse
import ast
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def normalized(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve(strict=True)
    manifest = json.loads((bundle / "patch_manifest.json").read_text(encoding="utf-8"))
    if manifest["target"] != "src/data/sim_demos.py":
        raise ValueError("Unexpected patch target")
    target = (repo / manifest["target"]).resolve(strict=True)
    try:
        target.relative_to(repo)
    except ValueError:
        raise ValueError("Patch target escapes repository")
    if not (repo / "scripts/train.py").is_file():
        raise ValueError("Wrong repository: scripts/train.py missing")
    source = normalized(bundle / "sim_demos.py")
    if sha(source) != manifest["new_sha256"]:
        raise ValueError("Patch source checksum mismatch")
    ast.parse(source.decode("utf-8"))
    current = normalized(target)
    if sha(current) == manifest["new_sha256"]:
        print("FINETUNE_SPLIT_FIX_ALREADY_INSTALLED")
        return
    if sha(current) != manifest["previous_sha256"]:
        raise ValueError("Installed sim_demos.py differs from the supported version; no file changed")
    locked_paths = [
        "data/splits/pick_place_finetune_splits.json",
        "data/splits/pick_place_finetune_manifest.json",
        "configs/act/pick_place_finetune.json",
    ]
    if any((repo / name).exists() for name in locked_paths):
        raise ValueError("Fine-tune artifacts already exist; preserve the locked split and inspect before changing code")
    if shutil.which("squeue"):
        result = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T"],
            check=True, capture_output=True, text=True,
        )
        active = [row for row in result.stdout.splitlines()
                  if len(row.split("|")) >= 2
                  and row.split("|")[1].startswith(("cfg5-ft", "cfg5-act"))]
        if active:
            raise RuntimeError("Relevant Slurm job still active; no file changed: " + "; ".join(active))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = repo / "outputs/install_backups" / ("split_fix_" + stamp) / manifest["target"]
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, backup)
    partial = target.with_name(target.name + ".split_fix_partial")
    created_partial = False
    try:
        with partial.open("xb") as stream:
            created_partial = True
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(partial), str(target))
    finally:
        if created_partial and partial.exists():
            partial.unlink()
    print("FINETUNE_SPLIT_FIX_INSTALLED")
    print("Backup: " + str(backup))


if __name__ == "__main__":
    main()
