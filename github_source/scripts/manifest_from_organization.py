"""Lập manifest đủ104 task từ plan đã verify, không decode video lần hai."""
import argparse
from collections import Counter
import _bootstrap
from src.data.common import ROOT, load_config, read_json, write_json, outside_raw, digest, episode_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/data/cfg5_all8.json")
    p.add_argument("--plan", default="docs/audit/cfg5_all_organization_plan.json")
    p.add_argument("--verification", default="docs/audit/cfg5_all_organization_verified.json")
    p.add_argument("--output", default="data/splits/cfg5_all_manifest.json")
    a = p.parse_args(); cfg = load_config(a.config)
    plan, verified = read_json(a.plan), read_json(a.verification)
    plan_hash = plan.pop("plan_hash")
    if digest(plan) != plan_hash or verified["plan_hash"] != plan_hash or not verified["directory_identity_and_file_count_bytes_verified"]:
        raise ValueError("Organization plan/verification không khớp")
    if plan["data_config_hash"] != digest(cfg):
        raise ValueError("Data config không khớp plan")
    out = outside_raw(a.output, cfg)
    if out.exists(): raise FileExistsError("Manifest đã có; không ghi đè")
    accepted, rejected = [], []
    for record in plan["episodes"]:
        if record["kind"] == "human": continue
        if record["category"] == "missing_data":
            rejected.append({"episode": record["episode"], "reason": record["reason"]})
            continue
        name=record["episode"]
        folder=episode_path(cfg,name)
        metadata=read_json(folder / "metadata.json")
        words=name.split("_")
        accepted.append({"episode":name,"task":int(words[1]),"user":int(words[3]),"scene":int(words[5]),
                         "calib":str(metadata["calib"]),"rating":metadata["rating"],
                         "source_fingerprint":record["source_fingerprint"],"quality":record["quality"]})
    manifest={"schema_version":1,"config_hash":digest(cfg),"config":cfg,"episodes":accepted,
              "rejected":rejected,"skipped_counts":{"human_or_alternate_name":plan["kinds"]["human"]},
              "organization_plan_hash":plan_hash,"note":"All104 CFG5 tasks, robot8D ACT; human videos retained separately for future paired/one-shot work."}
    manifest["manifest_hash"]=digest(manifest)
    write_json(out,manifest)
    print(f"{len(accepted)} accepted / {len(rejected)} rejected robot episodes; {len(set(e['task'] for e in accepted))} tasks with ACT-ready data",flush=True)
