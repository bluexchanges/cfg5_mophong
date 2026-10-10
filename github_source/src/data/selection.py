"""Lọc episode từ manifest đã kiểm, không sửa raw hoặc tự sửa split cũ."""
from copy import deepcopy
from .common import digest, require_contract


def select_manifest(manifest, selection, cfg):
    require_contract(manifest, cfg)
    content = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(content) != manifest["manifest_hash"]:
        raise ValueError("Manifest hash sai")
    if selection["source_manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Selection thuộc manifest khác")
    selected = selection["episodes"]
    if not isinstance(selected, list) or not selected or not all(isinstance(x, str) for x in selected):
        raise ValueError("episodes phải là danh sách ID không rỗng")
    if len(selected) != len(set(selected)):
        raise ValueError("Selection có episode trùng")
    valid = {record["episode"] for record in manifest["episodes"]}
    unknown = set(selected) - valid
    if unknown:
        raise ValueError(f"ID chưa qua bộ lọc manifest: {sorted(unknown)}")
    result = deepcopy(content)
    result["episodes"] = [record for record in result["episodes"] if record["episode"] in selected]
    result["selection"] = {"source_manifest_hash": manifest["manifest_hash"],
                           "selection_hash": digest(selection), "note": selection.get("note", ""),
                           "excluded_accepted_episodes": sorted(valid - set(selected))}
    result["manifest_hash"] = digest(result)
    return result
