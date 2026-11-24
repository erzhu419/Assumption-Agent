#!/usr/bin/env python3
"""Recompute capability counts from committed receipts; no Docker or model calls.

HEGEL_AUDIT_REPO accepts either repository root or its Hegel Machine directory.
HEGEL_AUDIT_BUNDLE optionally selects a different evidence bundle directory name.
"""
from collections import Counter
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("HEGEL_AUDIT_REPO", HERE.parents[1] / "repo_m691_4001bef56"))
HM = REPO if REPO.name == "Hegel Machine" else REPO / "Hegel Machine"
BUNDLE = os.environ.get("HEGEL_AUDIT_BUNDLE", "m6_9_1_a_two_branch_frontier_40ep_20261123")
B = HM / "docs/evidence" / BUNDLE
assert B.is_dir(), f"Missing committed evidence bundle: {B}"

qualification = []
for p in sorted(B.glob("branch_trajectories/*/branch_2*/01_decision_provenance_dag.json")):
    for rec in json.loads(p.read_text())["records"]:
        if rec.get("decision_type") == "D1C_SKILL_BINDING_AND_SMOKE_QUALIFICATION":
            qualification.append({"episode": p.parents[1].name,
                                  "qualified": rec["chosen_option"].get("qualified_count"),
                                  "modules": rec.get("candidates_evaluated", [])})

branch_manifests = sorted(B.glob("branch_trajectories/*/branch_*/04b_producer_skill_manifest/executable_skill_manifest.json"))
producer_counts = {}
empty_build = 0
for p in branch_manifests:
    m = json.loads(p.read_text())
    producer_counts[str(p.relative_to(B))] = len(m["producer"]["sources"])
    empty_build += not m["producer"].get("build_and_invoke_commands")

wave1_lineage = []
for p in sorted(B.glob("episodes/wave_1*/04b_producer_skill_manifest/executable_skill_manifest.json")):
    m = json.loads(p.read_text())
    wave1_lineage.append({"episode": p.parents[1].name, "identity": m["identity"],
                          "producer_count": len(m["producer"]["sources"])})

tasks = {}
by_function = Counter()
records = 0
total = 0
for p in sorted(B.glob("branch_trajectories/*/branch_2*/04c_full_skill_activations.jsonl")):
    counts = Counter()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records += 1
        if row.get("status") != "completed" or row.get("caller_role") == "smoke_test":
            continue
        count = int(row.get("call_count", 1))
        key = " / ".join(str(row.get(k, "")) for k in ("module", "function", "caller_role"))
        counts[key] += count
        by_function[key] += count
        total += count
    if counts:
        tasks[p.parents[1].name] = {"count": sum(counts.values()), "by_function": dict(counts)}

refs = []
missing = []
for p in sorted(B.glob("**/04b_producer_skill_manifest/executable_skill_manifest.json")):
    m = json.loads(p.read_text())
    for blob in m.get("state", {}).get("artifact_blob_refs", []):
        ref = blob.get("content_addressed_ref")
        if not ref:
            continue
        refs.append(ref)
        # Current bundle stores blobs under branch_trajectories; also accept explicit root store.
        if not (B / ref).is_file() and not (B / "branch_trajectories" / ref).is_file():
            missing.append({"manifest": str(p.relative_to(B)), **blob})

result = {"bundle": BUNDLE, "qualification": qualification,
          "manifest_stats": {"count": len(branch_manifests), "empty_build_commands": empty_build,
                             "producer_counts": producer_counts},
          "wave1_lineage": wave1_lineage,
          "activations": {"records": records, "count": total, "by_function": dict(by_function), "tasks": tasks},
          "blob_refs": {"occurrences": len(refs), "unique": len(set(refs)),
                        "missing_occurrences": len(missing),
                        "missing_unique": len({x["content_addressed_ref"] for x in missing}), "missing": missing}}
(HERE / "empirical_capabilities_recomputed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"completed_non_smoke_calls": total, "tasks_with_calls": len(tasks),
                  "qualification_episodes": len(qualification),
                  "staged_modules": sum(len(x["modules"]) for x in qualification),
                  "qualified_modules": sum(x["qualified"] or 0 for x in qualification),
                  "branch_manifests": len(branch_manifests), "empty_build_commands": empty_build,
                  "wave1_source_branches": Counter(x["identity"]["source_episode_id"] for x in wave1_lineage),
                  "missing_blob_unique": result["blob_refs"]["missing_unique"]}, indent=2))
