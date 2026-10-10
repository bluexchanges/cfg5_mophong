"""Tải riêng Panda Menagerie tại commit cố định; không sửa upstream XML."""
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
import _bootstrap
from src.data.common import ROOT, read_json, write_json

COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/franka_emika_panda/"
DEST = ROOT / "mujoco/models/franka_emika_panda"


def download():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "manifest.json"
    prior = read_json(manifest_path) if manifest_path.exists() else None
    expected = {r["path"]: r["sha256"] for r in prior["files"]} if prior else {}
    records = []

    def get(name):
        path = (DEST / name).resolve()
        if not path.is_relative_to(DEST.resolve()):
            raise ValueError("Asset path ngoài model folder")
        if path.exists():
            content = path.read_bytes()
        else:
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        if name in expected and sha != expected[name]:
            raise ValueError(f"Hash không khớp: {name}")
        records.append({"path": name, "sha256": sha, "bytes": len(content), "url": BASE+name})
        print(name, len(content), flush=True)
        return content

    xml = ET.fromstring(get("panda.xml"))
    for name in ("LICENSE", "README.md", "CHANGELOG.md"):
        get(name)
    for name in sorted({m.attrib["file"] for m in xml.findall("./asset/mesh")}):
        get("assets/" + name)
    write_json(manifest_path, {"repository": "google-deepmind/mujoco_menagerie", "commit": COMMIT,
                              "license": "Apache-2.0", "files": records,
                              "total_bytes": sum(r["bytes"] for r in records)})
    print("Verified model files:", len(records))


if __name__ == "__main__":
    download()
