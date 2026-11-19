"""Derive candidate consumption, timing, contributions and matched Recall from receipts.

No solver execution, network request, or repository edit. Run after the M3 replay.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean

REPO = Path(os.environ.get("HEGEL_REPO", "/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine"))
OUT = Path(os.environ.get("AUDIT_OUTPUT_DIR", Path(__file__).resolve().parent))
ORIG = REPO / "docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


sources = read_jsonl(ORIG / "source_episodes.jsonl")
targets = read_jsonl(ORIG / "target_episodes.jsonl")
clusters = sorted({r["cluster_id"] for r in targets})
reachability = {}
for cid in clusters:
    rows = [r for r in sources if r.get("cluster_id") == cid and r["arm"] == "O_HExA_CodeSkills" and r["strict_pass"]]
    skills, mistakes = defaultdict(list), defaultdict(list)
    for r in rows:
        if not r.get("selected_schema"):
            continue
        bank = skills if r["phase"] == "source" else mistakes
        bank[r["active_arity"]].append(r["selected_schema"])
    result = []
    for ar in sorted(set(skills) | set(mistakes)):
        tags = skills[ar]
        source_skills = sorted(set(tags), key=lambda s: (-tags.count(s), s))
        ordered = list(dict.fromkeys([*source_skills, *mistakes[ar]]))
        prefix = ordered[:2]
        without = source_skills[:2]
        result.append({
            "arity": ar,
            "source_skills": source_skills,
            "mistakes": mistakes[ar],
            "actual_prefix": prefix,
            "prefix_if_mistake_bank_removed": without,
            "equal": prefix == without,
            "mistake_positions_one_based": {s: ordered.index(s) + 1 for s in mistakes[ar]},
        })
    reachability[cid] = result

wall = {}
for label, folder in (("published", ORIG), ("replay", OUT / "replay")):
    cs = json.loads((folder / "paired_results.json").read_text())["heldout_cluster_summaries"]
    wall[label] = {}
    for arm in cs[0]["arms"]:
        avg = mean(c["arms"][arm]["target_measured_python_wall_seconds"] for c in cs)
        wall[label][arm] = {
            "mean_python_wall_seconds_per_24_target_cluster": avg,
            "mean_python_wall_seconds_per_target": avg / 24,
        }
baseline = {"hexa_mistake_reachability": reachability, "wall_time_units": wall}
(OUT / "baseline_consumption_and_timing.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")

h, b, u = "Hegel_M3_HippoPPR", "Flat_BM25_EqualMetadata", "HippoPPR_UncheckedRecall"
paired = json.loads((ORIG / "paired_results.json").read_text())
by_id = {(r["cluster_id"], r["arm"], r["target_name"]): r for r in targets}
t13_name = "T13_guard_two_hop_strict_rollover"
t13 = {
    cid: {
        k: by_id[(cid, h, t13_name)]["costs"][k] - by_id[(cid, b, t13_name)]["costs"][k]
        for k in ("measurement_cost", "compute_cost", "composite_cost")
    }
    for cid in clusters
}
other_target_diffs, other_e2e_diffs, other_meas = [], [], []
for cs in paired["heldout_cluster_summaries"]:
    cid = cs["cluster_id"]
    diff = cs["arms"][h]["target_composite_cost"] - cs["arms"][b]["target_composite_cost"] - t13[cid]["composite_cost"]
    other_target_diffs.append(diff)
    other_e2e_diffs.append(diff + cs["arms"][h]["source_and_index_composite_cost"] - cs["arms"][b]["source_and_index_composite_cost"])
    other_meas.append(cs["arms"][h]["target_measurement_cost"] - cs["arms"][b]["target_measurement_cost"] - t13[cid]["measurement_cost"])

eligible = {(r["cluster_id"], r["target_name"]) for r in targets if r["arm"] == h and not r["retrieval_diagnostic"].get("blocked_by_scale_map_pre_pilot")}
matched = {}
for arm in (h, u):
    rows = [r["retrieval_diagnostic"] for r in targets if r["arm"] == arm and (r["cluster_id"], r["target_name"]) in eligible]
    matched[arm] = {
        "query_count": len(rows),
        "recall1_numerator": sum(bool(r.get("authorized_top_schemas")) and r["authorized_top_schemas"][0] == r.get("truth_schema_audit_only") for r in rows),
        "recall2_numerator": sum(r.get("truth_schema_audit_only") in r.get("authorized_top_schemas", [])[:2] for r in rows),
    }
contribution = {
    "T13_per_cluster_differences": t13,
    "T13_mean_composite": mean(r["composite_cost"] for r in t13.values()),
    "other_23_mean_target_composite_difference": mean(other_target_diffs),
    "other_23_mean_e2e_difference": mean(other_e2e_diffs),
    "other_23_mean_measurement_difference": mean(other_meas),
    "scope": "Algebraic contribution of existing tasks; not a new preregistered suite.",
    "matched_recall_denominator": matched,
}
(OUT / "contribution_and_recall_scope.json").write_text(json.dumps(contribution, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"hexa_prefixes_unchanged": all(x["equal"] for rows in reachability.values() for x in rows), "matched_recall": matched}, ensure_ascii=False, indent=2))
