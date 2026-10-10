"""Tải source RH20T ACT cố định commit, kiểm Git blob SHA; không tải weights."""
import argparse
import concurrent.futures
import hashlib
from pathlib import PurePosixPath
import urllib.request
import json
import _bootstrap
from src.data.common import ROOT, write_json

COMMIT = "107ce1287a46be3a1a7ea954f8bbdec3bb103288"
REPO = "rh20t/act_baseline"
DEST = ROOT / "third_party/rh20t_act"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "RH20T-reproduction"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def download():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated") or tree["sha"] != COMMIT:
        raise ValueError("Tree không đầy đủ hoặc khác commit")
    entries = [e for e in tree["tree"] if e["type"] == "blob"]
    def save(entry):
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["mode"] not in ("100644", "100755"):
            raise ValueError("Path/mode upstream không hợp lệ")
        path = DEST.joinpath(*relative.parts)
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{relative.as_posix()}"
        payload = path.read_bytes() if path.exists() else fetch(url)
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob != entry["sha"]:
            raise ValueError(f"Hash sai, không ghi đè: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return {"path": relative.as_posix(), "url": url, "bytes": len(payload),
                "git_blob_sha": blob, "sha256": hashlib.sha256(payload).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(save, entries))
    write_json(DEST.parent / "rh20t_act_manifest.json", {"repository": REPO, "commit": COMMIT,
               "weights_downloaded": False, "files": files})
    print(f"Verified {len(files)} source files, {sum(f['bytes'] for f in files)} bytes; no weights.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    download()
