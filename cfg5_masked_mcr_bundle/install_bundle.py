"""Apply reviewed MCR patches with preflight and backups; leave raw/checkpoints alone."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def adapt_pretrain_mask(content):
    """Adapt only the two recognized batch/policy interfaces, preserving server code."""
    tree = ast.parse(content)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    move, run = functions.get("move_batch"), functions.get("run_epoch")
    if move is None or run is None:
        raise ValueError("Cannot recognize move_batch/run_epoch; provide server pretrain.py")
    if not {"batch", "device"}.issubset(arg.arg for arg in move.args.args):
        raise ValueError("move_batch interface differs; provide server pretrain.py")
    lines = content.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    def location(lineno, byte_col):
        # AST column offsets are UTF-8 bytes, while Python string slices use chars.
        return offsets[lineno-1] + len(lines[lineno-1].encode("utf-8")[:byte_col].decode("utf-8"))
    def subscript(node, name, key):
        return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == name and isinstance(node.slice, ast.Constant)
                and node.slice.value == key)
    changes = []
    move_text = ast.get_source_segment(content, move)
    transfer = 'if "camera_valid" in batch:\n        tensors["camera_valid"] = batch["camera_valid"].to(device, non_blocking=True)'
    if 'camera_valid' not in move_text:
        returns = [node for node in ast.walk(move) if isinstance(node, ast.Return)]
        if len(returns) != 1 or returns[0] not in move.body:
            raise ValueError("move_batch return differs; provide server pretrain.py")
        direct_return = isinstance(returns[0].value, ast.DictComp)
        if direct_return:
            # Exact interface found in the downloaded user08 server file.
            if len(move.body) != 1:
                raise ValueError("Direct-return move_batch has extra logic; provide server pretrain.py")
            value = returns[0].value
        else:
            if not isinstance(returns[0].value, ast.Name) or returns[0].value.id != "tensors":
                raise ValueError("move_batch return differs; provide server pretrain.py")
            assignments = [node for node in move.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "tensors" for t in node.targets)]
            if len(assignments) != 1 or not isinstance(assignments[0].value, ast.DictComp):
                raise ValueError("move_batch tensor construction differs; provide server pretrain.py")
            value = assignments[0].value
        keys = ["state", "images", "actions", "is_pad"]
        if (len(value.generators) != 1 or not isinstance(value.generators[0].iter, (ast.Tuple, ast.List))
                or [getattr(n, "value", None) for n in value.generators[0].iter.elts] != keys
                or not isinstance(value.key, ast.Name) or not isinstance(value.generators[0].target, ast.Name)
                or value.key.id != value.generators[0].target.id
                or not isinstance(value.value, ast.Call) or not isinstance(value.value.func, ast.Attribute)
                or value.value.func.attr != "to" or not isinstance(value.value.func.value, ast.Subscript)
                or not isinstance(value.value.func.value.value, ast.Name) or value.value.func.value.value.id != "batch"):
            raise ValueError("move_batch tensor schema differs; provide server pretrain.py")
        if direct_return:
            returned = returns[0]
            start = location(returned.lineno, returned.col_offset)
            end = location(returned.end_lineno, returned.end_col_offset)
            replacement = 'tensors = ' + ast.get_source_segment(content, value) + '\n    ' + transfer + '\n    return tensors'
            changes.append((start, end, replacement))
        else:
            at = offsets[returns[0].lineno-1]
            changes.append((at, at, '    ' + transfer + '\n'))
    elif transfer not in move_text:
        raise ValueError("Existing camera transfer differs; provide server pretrain.py")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "policy"]
    if len(calls) != 1:
        raise ValueError("run_epoch policy call differs; provide server pretrain.py")
    call = calls[0]
    if len(call.args) != 4 or not all(subscript(arg, "tensors", key) for arg, key in zip(call.args, ("state", "images", "actions", "is_pad"))):
        raise ValueError("run_epoch policy inputs differ; provide server pretrain.py")
    extra = '({"camera_valid": tensors["camera_valid"]} if "camera_valid" in tensors else {})'
    if call.keywords:
        expected = ast.parse(extra, mode="eval").body
        if (len(call.keywords) != 1 or call.keywords[0].arg is not None
                or ast.dump(call.keywords[0].value) != ast.dump(expected)):
            raise ValueError("Existing policy keywords need review; provide server pretrain.py")
    else:
        at = location(call.end_lineno, call.end_col_offset) - 1
        if content[at] != ')':
            raise ValueError("Cannot locate policy call closing parenthesis")
        changes.append((at, at, ', **' + extra))
    for start, end, replacement in sorted(changes, reverse=True):
        content = content[:start] + replacement + content[end:]
    ast.parse(content)
    return content


def safe(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Unsafe bundle path: {relative}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    bundle, repo = Path(__file__).resolve().parent, Path(args.repo).resolve()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    missing = [r for r in manifest["prerequisites"] if not safe(repo, r).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing server prerequisites: {missing}")
    operations = []
    for row in manifest["patches"]:
        target = safe(repo, row["path"])
        original = content = target.read_text(encoding="utf-8")
        try:
            for hunk in row["hunks"]:
                if content.count(hunk["new"]) == 1:
                    continue
                if content.count(hunk["old"]) != 1:
                    raise ValueError(f"Server code differs at {row['path']}; no files installed. Send this server file for patch adaptation.")
                content = content.replace(hunk["old"], hunk["new"], 1)
        except ValueError:
            if row["path"] != "src/training/pretrain.py":
                raise
            content = adapt_pretrain_mask(original)
            print("ADAPTED_SERVER_FUNCTIONS", row["path"], flush=True)
        if row["path"].endswith(".py"):
            ast.parse(content, filename=row["path"])
        if content != original:
            operations.append((row["path"], content.encode("utf-8")))
    for row in manifest["new_files"]:
        source = safe(bundle, row["path"])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Bundle checksum wrong: {row['path']}")
        target = safe(repo, row["path"])
        if target.exists():
            if target.read_text(encoding="utf-8") != data.decode("utf-8"):
                raise FileExistsError(f"Custom new file already exists: {target}")
        else:
            operations.append((row["path"], data))
    print("PATCH_PLAN", [r for r, _ in operations], flush=True)
    if args.dry_run:
        print("CFG5_MASKED_INSTALL_DRY_RUN_OK", flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = repo / "outputs/install_backups" / ("cfg5_masked_" + stamp)
    backup.mkdir(parents=True)
    for relative, data in operations:
        target = safe(repo, relative)
        if target.exists():
            saved = safe(backup, relative)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".masked_install_partial")
        with temporary.open("xb") as handle:
            handle.write(data)
        os.replace(temporary, target)
    print(f"CFG5_MASKED_BUNDLE_INSTALLED backups={backup}", flush=True)


if __name__ == "__main__":
    main()
