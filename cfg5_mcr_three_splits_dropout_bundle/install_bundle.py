"""Install only recognized code versions; preflight all files, then back up changed code."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def safe(root, relative):
    if Path(relative).is_absolute():
        raise ValueError("Absolute bundle path")
    path = root / relative
    path.relative_to(root)
    if path.resolve() != path:
        raise ValueError("Symlink/escaping bundle path: " + relative)
    return path


def install(bundle, repo, dry_run=False):
    bundle, repo = Path(bundle).absolute(), Path(repo).absolute()
    if bundle.resolve(strict=True) != bundle or repo.resolve(strict=True) != repo:
        raise ValueError("Bundle/repo must be actual paths")
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name in manifest["prerequisites"]:
        if not safe(repo, name).is_file():
            raise FileNotFoundError("Missing prerequisite: " + name)
    plan = []
    for row in manifest["files"]:
        relative = row["path"]
        if not relative.startswith(("scripts/", "src/", "tests/", "docs/")):
            raise ValueError("Installer may only update code/docs")
        data = safe(bundle, relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Bundle checksum differs: " + relative)
        if relative.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=relative)
        path = safe(repo, relative)
        if path.exists():
            original = path.read_text(encoding="utf-8").encode("utf-8")
            if original == data:
                continue
            if hashlib.sha256(original).hexdigest() not in row.get("before_sha256", []):
                raise ValueError("Server file differs; nothing installed. Send this file for adaptation: " + relative)
        elif row.get("required_existing", False):
            raise FileNotFoundError("Missing V5 code: " + relative)
        temporary = path.with_name(path.name + ".dropout_install_partial")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError("Inspect incomplete install before retry: " + str(temporary))
        plan.append((relative, data))
    print("MCR_DROPOUT_PATCH_PLAN", [name for name, _ in plan], flush=True)
    if dry_run:
        print("MCR_DROPOUT_DRY_RUN_OK", flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    relative_backup = "outputs/install_backups/mcr_dropout_" + stamp
    backup = safe(repo, relative_backup)
    backup.mkdir(parents=True, exist_ok=False)
    for name, data in plan:
        target = safe(repo, name)
        if target.exists():
            old = safe(backup, name)
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".dropout_install_partial")
        with temporary.open("xb") as stream:
            stream.write(data)
        os.replace(temporary, target)
    print("MCR_DROPOUT_BUNDLE_INSTALLED backups=" + str(backup), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    install(Path(__file__).resolve().parent, args.repo, args.dry_run)


if __name__ == "__main__":
    main()
