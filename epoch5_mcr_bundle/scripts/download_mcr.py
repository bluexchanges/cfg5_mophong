"""Download one pinned official MCR ResNet50 checkpoint, with SHA256 verification."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import _bootstrap
from src.data.common import ROOT

REVISION = "057651ca0b6a27aba390449603f674cd0977f87c"
SHA256 = "21560db9e795384a7b73b0626d33d6525f9d4c5de11fca6ded52089aab14f97c"
SIZE = 94345861
URL = f"https://huggingface.co/GqJiang/robots-pretrain-robots/resolve/{REVISION}/mcr_resnet50.pth"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def download(destination):
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size != SIZE or sha(destination) != SHA256:
            raise ValueError("Checkpoint hiện có sai checksum; không ghi đè")
        print(f"MCR_VERIFIED {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    # A failed attempt is retried from the beginning. Completed assets are reused.
    checksum, total = hashlib.sha256(), 0
    with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as f:
        while True:
            block = response.read(2**20)
            if not block:
                break
            total += len(block)
            if total > SIZE:
                raise ValueError("Download vượt kích thước chính thức")
            checksum.update(block)
            f.write(block)
    if total != SIZE or checksum.hexdigest() != SHA256:
        raise ValueError("Download thiếu/sai SHA256; không dùng checkpoint")
    temporary.replace(destination)
    print(f"MCR_VERIFIED {destination} bytes={total} sha256={SHA256}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/backbones/mcr_resnet50.pth")
    args = parser.parse_args()
    download(ROOT / args.output)
