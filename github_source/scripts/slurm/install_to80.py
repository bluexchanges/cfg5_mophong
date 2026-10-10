"""Verify payload before atomically installing it, retaining original code."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

FILES = ("src/training/pretrain.py", "src/training/fork_run.py", "src/training/multi_gpu.py")


def install(bundle, repo):
    hashes = json.loads((bundle / "bundle_hashes.json").read_text())
    payloads = {}
    for name in FILES:
        payload = (bundle / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != hashes[name]:
            raise ValueError(f"Checksum sai: {name}")
        compile(payload, name, "exec")
        payloads[name] = payload
    backup = Path(tempfile.mkdtemp(prefix="to80-code-backup-", dir=Path.home() / "cfg5-pretrain"))
    for name, payload in payloads.items():
        target = repo / name
        if target.exists():
            original = backup / name
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, original)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".install-partial")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Code multi-GPU đã cài; backup={backup}", flush=True)


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]))
