"""Install three validated source files atomically, retaining a code backup."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

FILES = ("src/training/pretrain.py", "src/training/fork_run.py", "scripts/fork_training_run.py")


def validate_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "bundle_hashes.json").read_text(encoding="utf-8"))
    payloads = {}
    for name in FILES:
        payload = (bundle / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest[name]:
            raise ValueError(f"Bundle sai checksum: {name}")
        compile(payload, name, "exec")
        payloads[name] = payload
    return payloads


def install(bundle, repo):
    payloads = validate_bundle(bundle)
    repo = Path(repo)
    backup = Path(tempfile.mkdtemp(prefix="workers2-code-backup-", dir=repo.parents[3]))
    for name, payload in payloads.items():
        target = repo / name
        old = backup / name
        old.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".install-partial")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            installed = target.read_bytes()
            if installed != payload:
                raise IOError(f"File cài không khớp bundle: {target}")
            compile(installed, name, "exec")
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Code đã kiểm checksum/cú pháp; backup: {backup}", flush=True)


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]))
