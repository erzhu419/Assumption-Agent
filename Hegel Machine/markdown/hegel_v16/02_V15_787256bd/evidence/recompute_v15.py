#!/usr/bin/env python3
"""Recompute V15 descriptive results from saved episode MSE/cost fields.

No model calls, scientific reruns, or mutation of the input bundle. The saved
external MSE values are inputs, not independently recomputed predictions.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean, median


E1_NAME = "deliverable_3_prospective_e1_same_history_organization_20261118.json"
E2_NAME = "deliverable_4_prospective_e2_end_to_end_strategy_20261118.json"
K = "K_eager_persistent"
R = "R_same_history_lazy"
RM = "K_true_repaired_version_removed_raw_rebuild_allowed"
COLD = "K_cold"
K2 = "K_autonomous_end_to_end"
ORD = "Ordinary_autonomous_end_to_end"
TOL = 1e-7
COSTS = ("compute_cost", "observation_cost", "measurement_cost", "agent_run_wall_seconds")


def sign_flip(values):
    nz = [float(x) for x in values if abs(x) > 1e-12]
    if not nz:
        return 1.0
    threshold = abs(sum(nz))
    return sum(abs(sum(s * v for s, v in zip(signs, nz))) >= threshold - 1e-9
               for signs in product((-1, 1), repeat=len(nz))) / (2 ** len(nz))


def distribution(values):
    return {"values": list(values), "mean": mean(values), "median": median(values),
            "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "zero": sum(v == 0 for v in values), "descriptive_sign_flip_p": sign_flip(values)}


def gates(ep):
    is_target = ":T" in ep["episode_tag"]
    conflict = ":S4_" in ep["episode_tag"]
    comp_budget = 840000 if is_target else 880000 if conflict else 680000
    costs = ep["measured_costs"]
    budget_ok = costs["compute_cost"] <= comp_budget and costs["measurement_cost"] <= 12200 and costs["observation_cost"] <= 600
    conf = ep.get("confirmation_loss")
    confirmed = ep["status"] == "completed" and conf is not None and float(conf) <= TOL
    accurate = float(ep["external_fresh48_mse"]) <= TOL
    return {"strict_pass": bool(confirmed and accurate and budget_ok), "internally_confirmed": confirmed,
            "accurate_external": accurate, "false_internal_confirmation": bool(confirmed and not accurate),
            "unresolved_or_unconfirmed": not confirmed,
            "unconfirmed_yet_accurate_on_external": bool(not confirmed and accurate),
            "within_predeclared_budget": budget_ok}


def summarize(episodes):
    gs = [gates(e) for e in episodes]
    errors = []
    for e, g in zip(episodes, gs):
        for key, value in g.items():
            if key in e and e[key] != value:
                errors.append({"episode_tag": e["episode_tag"], "key": key, "stored": e[key], "recomputed": value})
    decomp = []
    for e in episodes:
        parts = e["cost_decomposition_sec7_1"]
        summed = sum(v for k, v in parts.items() if k != "total_compute_cost")
        extra = summed - e["measured_costs"]["compute_cost"]
        if extra:
            decomp.append({"episode_tag": e["episode_tag"], "bucket_sum": summed,
                           "compute_total": e["measured_costs"]["compute_cost"], "excess": extra})
    return {
        "n": len(episodes),
        **{key: sum(int(g[key]) for g in gs) for key in gs[0]},
        "status_counts": dict(Counter(e["status"] for e in episodes)),
        "costs": {key: {"sum": sum(e["measured_costs"][key] for e in episodes),
                         "mean": mean(e["measured_costs"][key] for e in episodes),
                         "median": median(e["measured_costs"][key] for e in episodes)} for key in COSTS},
        "stored_gate_mismatches": errors,
        "decomposition_nonadditive_episode_count": len(decomp),
        "decomposition_excess_sum": sum(d["excess"] for d in decomp),
        "decomposition_examples": decomp[:6],
        "raw_reconstruction_event_statuses": dict(Counter(ev["status"] for e in episodes for ev in e.get("raw_history_reconstruction_events", []))),
        "raw_reconstruction_direct_compute": sum(ev.get("compute_cost", 0) for e in episodes for ev in e.get("raw_history_reconstruction_events", [])),
        "raw_reconstruction_exported_compute": sum(e.get("on_demand_or_raw_reconstruction_compute", 0) for e in episodes),
        "adopted_repaired_or_rebuilt_focal_count": sum(bool(e.get("adopted_repaired_or_rebuilt_focal_source")) for e in episodes),
        "wrong_knowledge_transfer_narrow_count": sum(bool(e.get("wrong_knowledge_transfer")) for e in episodes),
    }


def compact(ep):
    return {"episode_tag": ep["episode_tag"], "arm": ep["arm"], "episode_uid": ep["episode_uid"],
            **gates(ep), "status": ep["status"], "confirmation_loss": ep.get("confirmation_loss"),
            "external_fresh48_mse": ep["external_fresh48_mse"], "costs": ep["measured_costs"],
            "adopted_source_ids": ep.get("confirmed_adopted_source_ids", []),
            "reconstruction_events": ep.get("raw_history_reconstruction_events", [])}


def cost_total(episodes, cost):
    return sum(e["measured_costs"][cost] for e in episodes)


def program_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.get("children", []):
            yield from program_nodes(child)


def expression(node):
    op = node.get("op")
    if op == "const":
        return str(node.get("value"))
    if op in ("arg", "feature"):
        return str(node.get("name"))
    name = "[" + str(node["name"]) + "]" if node.get("name") else ""
    return str(op) + name + "(" + ",".join(expression(c) for c in node.get("children", [])) + ")"


def event_digest(event):
    result = {k: v for k, v in event.items() if k not in ("adopted_program", "executed_probe_ids")}
    result["executed_probe_ids"] = event.get("executed_probe_ids", [])
    nodes = list(program_nodes(event.get("adopted_program", {})))
    result["adopted_program_operator_counts"] = dict(Counter(n.get("op") for n in nodes if n.get("op")))
    result["relation_gates"] = [
        {"direction": n.get("name"), "radius": expression(n["children"][0]),
         "condition": expression(n["children"][1])}
        for n in nodes if n.get("op") == "relation_sum" and len(n.get("children", [])) > 1
    ]
    return result


def case_digest(ep):
    d = compact(ep)
    d["consumption"] = [event_digest(ev) for ev in ep.get("structured_5stage_consumption_events", [])]
    d["first_dispatch"] = ep.get("branch_dispatch_telemetry")
    return d


def pair_compare(focal_clusters, ordinary_clusters):
    """Signs are ordinary-minus-focal costs; positive means focal cheaper.

    Common-success costs are explicitly descriptive, post-selection summaries.
    """
    flat_f = [e for c in focal_clusters for e in c]
    flat_o = [e for c in ordinary_clusters for e in c]
    for a, b in zip(flat_f, flat_o):
        assert a["episode_uid"] == b["episode_uid"]
    discordant, ff, oo = [], [], []
    wins = losses = both = neither = 0
    for f, o in zip(flat_f, flat_o):
        fs, os = gates(f)["strict_pass"], gates(o)["strict_pass"]
        wins += fs and not os
        losses += os and not fs
        both += fs and os
        neither += not fs and not os
        if fs != os:
            discordant.append({"focal": compact(f), "ordinary": compact(o)})
        if fs and os:
            ff.append(f); oo.append(o)
    return {"paired_tasks": len(flat_f), "focal_only_success": wins, "ordinary_only_success": losses,
            "both_success": both, "neither_success": neither,
            "cluster_success_difference_focal_minus_ordinary": distribution([
                sum(gates(e)["strict_pass"] for e in f) - sum(gates(e)["strict_pass"] for e in o)
                for f, o in zip(focal_clusters, ordinary_clusters)]),
            "cluster_cost_difference_ordinary_minus_focal": {cost: distribution([
                cost_total(o, cost) - cost_total(f, cost) for f, o in zip(focal_clusters, ordinary_clusters)]) for cost in COSTS},
            "descriptive_postselected_common_success_mean_compute_difference_ordinary_minus_focal": (
                mean(o["measured_costs"]["compute_cost"] - f["measured_costs"]["compute_cost"] for f, o in zip(ff, oo)) if ff else None),
            "discordant_success_pairs": discordant}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("recomputed_v15.json"))
    args = ap.parse_args()
    receipts = args.bundle / "receipts/v15_true_intervention_and_strategy_20261118"
    e1 = json.loads((receipts / E1_NAME).read_text())
    e2 = json.loads((receipts / E2_NAME).read_text())
    cs1, cs2 = e1["prospective_clusters"], e2["prospective_clusters"]
    arms1 = {a: [[e for e in c["arms_target_results"][a]] for c in cs1] for a in (K, R, RM, COLD)}
    arms2 = {a: [[e for e in c[a]["target_episodes"]] for c in cs2] for a in (K2, ORD)}
    src2 = {a: [[e for e in c[a]["source_episodes"]] for c in cs2] for a in (K2, ORD)}
    flatten = lambda clusters: [e for c in clusters for e in c]
    repeats = {
        "K_source_same_record_in_E1_and_E2": all(c1["shared_source_formation"]["episodes"] == c2[K2]["source_episodes"] for c1, c2 in zip(cs1, cs2)),
        "K_target_same_record_in_E1_and_E2": all(c1["arms_target_results"][K] == c2[K2]["target_episodes"] for c1, c2 in zip(cs1, cs2)),
    }
    unique_run_eps = flatten(src2[K2]) + flatten(src2[ORD]) + flatten(arms2[ORD])
    unique_run_eps += [e for clusters in arms1.values() for e in flatten(clusters)]
    assert len({(e["arm"], e["episode_tag"]) for e in unique_run_eps}) == len(unique_run_eps)
    e1_prefix = {}
    e2_prefix = {}
    for k in (1, 3, 6):
        target_delta = [cost_total(r[:k], "compute_cost") - cost_total(kk[:k], "compute_cost") for r, kk in zip(arms1[R], arms1[K])]
        maint = [c["shared_source_formation"]["source_reconciliation_maintenance_compute"] for c in cs1]
        obs_extra = [c["shared_source_formation"]["source_discriminating_probe_extra_observations"] for c in cs1]
        e1_prefix[str(k)] = {
            "observed_target_compute_R_minus_K": distribution(target_delta),
            "author_conditional_maintenance_allocation_R_minus_K": distribution([x-m for x, m in zip(target_delta, maint)]),
            "observed_target_observation_R_minus_K": distribution([cost_total(r[:k], "observation_cost")-cost_total(kk[:k], "observation_cost") for r, kk in zip(arms1[R], arms1[K])]),
            "author_observation_allocation_R_minus_K": distribution([cost_total(r[:k], "observation_cost")-cost_total(kk[:k], "observation_cost")-extra for r, kk, extra in zip(arms1[R], arms1[K], obs_extra)]),
            "note": "The author maintenance allocation is not an independently run R source trajectory. Acquisition is shared in same-history design; source probe observation debit to K alone is an allocation assumption.",
        }
    for k in (0, 1, 3, 6):
        e2_prefix[str(k)] = {cost: distribution([
            cost_total(os + ot[:k], cost) - cost_total(ks + kt[:k], cost)
            for os, ot, ks, kt in zip(src2[ORD], arms2[ORD], src2[K2], arms2[K2])]) for cost in COSTS}
    source_by_slot = {}
    for i in range(5):
        k_eps, o_eps = [c[i] for c in src2[K2]], [c[i] for c in src2[ORD]]
        label = k_eps[0]["episode_tag"].split(":", 1)[1]
        source_by_slot[label] = {"K": summarize(k_eps), "Ordinary": summarize(o_eps),
                               "paired_compute_Ordinary_minus_K": distribution([o["measured_costs"]["compute_cost"]-k["measured_costs"]["compute_cost"] for k, o in zip(k_eps, o_eps)])}
    tree = ast.parse((args.bundle / "bundled_sources/scripts/run_v15_prospective_e1_e2_studies.py").read_text())
    specs = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                 and node.target.id == "PROSPECTIVE_V15_CLUSTERS")
    specs = {s["base_seed"]: s for s in specs}
    cases = {"focal_adoption": [], "false_confirmation_K": [], "compatible_A_without_focal": []}
    for c1, c2 in zip(cs1, cs2):
        focal = c1["shared_source_formation"]["repaired_s4_object"]["concept_id"]
        for ep in c1["arms_target_results"][K]:
            spec = specs[c1["base_seed"]]
            relevant = ep.get("adopted_repaired_or_rebuilt_focal_source") or ep["false_internal_confirmation"]
            if relevant:
                entry = {"task_parameters_in_frozen_script": spec["target_params"][ep["slot_id"]],
                         "K": case_digest(ep), "other_arms": {},
                         "source_history": [{"episode_tag": z["episode_tag"], "concept_id": z["concept_id"],
                                             "strict_pass": gates(z)["strict_pass"], "compute_cost": z["measured_costs"]["compute_cost"]}
                                            for z in c1["shared_source_formation"]["episodes"]],
                         "S4_source": case_digest(c1["shared_source_formation"]["episodes"][-1]),
                         "repaired_object": c1["shared_source_formation"]["repaired_s4_object"]}
                for arm, eps in c1["arms_target_results"].items():
                    if arm != K:
                        entry["other_arms"][arm] = compact(next(z for z in eps if z["episode_uid"] == ep["episode_uid"]))
                entry["other_arms"][ORD] = compact(next(z for z in c2[ORD]["target_episodes"] if z["episode_uid"] == ep["episode_uid"]))
                cases["focal_adoption" if ep.get("adopted_repaired_or_rebuilt_focal_source") else "false_confirmation_K"].append(entry)
            if ep["slot_id"] in ("A1", "A2") and not ep.get("adopted_repaired_or_rebuilt_focal_source"):
                tel = ep.get("branch_dispatch_telemetry") or {}
                cases["compatible_A_without_focal"].append({"episode_tag": ep["episode_tag"],
                    "strict_pass": gates(ep)["strict_pass"], "sources_confirmed_adopted": ep["confirmed_adopted_source_ids"],
                    "first_dispatch_focal_empirically_excluded": "branch:" + focal in tel.get("empirically_excluded_on_context", []),
                    "first_dispatch_focal_scope_unknown": "branch:" + focal in tel.get("scope_not_established", []),
                    "confirmed_consumption": [event_digest(ev) for ev in ep["structured_5stage_consumption_events"] if ev.get("adopted_confirmed")]})
    by_role = {}
    all_arms = {**arms1, ORD: arms2[ORD]}
    for a, clusters in all_arms.items():
        by_role[a] = {role: summarize([e for e in flatten(clusters) if e["target_regime_role"] == role])
                      for role in sorted({e["target_regime_role"] for e in flatten(clusters)})}
    cluster_rows = []
    for i, (c1, c2) in enumerate(zip(cs1, cs2)):
        row = {"cluster_id": c1["cluster_id"], "order": c1["target_order_pattern"],
               "source_maintenance_compute": c1["shared_source_formation"]["source_reconciliation_maintenance_compute"],
               "source_probe_observations": c1["shared_source_formation"]["source_discriminating_probe_extra_observations"],
               "E1_targets": {a: summarize(arms1[a][i]) for a in arms1},
               "E2_sources": {a: summarize(src2[a][i]) for a in src2},
               "E2_Ordinary_targets": summarize(arms2[ORD][i]),
               "E2_end_to_end_compute_saving_Ordinary_minus_K": e2_prefix["6"]["compute_cost"]["values"][i],
               "E2_target_compute_saving_Ordinary_minus_K": cost_total(arms2[ORD][i], "compute_cost")-cost_total(arms2[K2][i], "compute_cost"),
               "removed_arm_direct_reconstruction_events": [{"target": e["episode_tag"], **ev} for e in arms1[RM][i] for ev in e["raw_history_reconstruction_events"]]}
        cluster_rows.append(row)
    hashes = []
    for path, claimed in e1["code_provenance"]["frozen_source_sha256"].items():
        got = hashlib.sha256((args.bundle / "bundled_sources" / path).read_bytes()).hexdigest()
        hashes.append({"path": path, "claimed": claimed, "actual": got, "match": got == claimed})
    outcomes = {
        "verification_scope": "Recomputed classifications, sums, paired differences and sign flips from saved confirmation/external MSE and measured-cost fields. Did not recompute underlying predictions or run science/model calls.",
        "tolerance": TOL,
        "provenance_hash_checks": hashes,
        "episode_counts": {"E1_rows_shared_source_plus_4_target_arms": 30+144, "E2_rows_two_source_and_two_target_arms": 60+72,
                           "sum_receipt_rows": 306, "duplicated_K_source_and_target_rows": 66,
                           "unique_actual_arm_episode_runs": len(unique_run_eps),
                           "unique_task_instances_by_episode_uid": len({e["episode_uid"] for e in unique_run_eps}),
                           "independent_cluster_n": len(cs1), **repeats},
        "all_unique_run_summary": summarize(unique_run_eps),
        "E1_target_summaries": {a: summarize(flatten(cs)) for a, cs in arms1.items()},
        "E2_source_summaries": {a: summarize(flatten(cs)) for a, cs in src2.items()},
        "E2_target_summaries": {a: summarize(flatten(cs)) for a, cs in arms2.items()},
        "E2_end_to_end_summaries": {a: summarize(flatten(src2[a])+flatten(arms2[a])) for a in arms2},
        "E1_source_maintenance": distribution([c["shared_source_formation"]["source_reconciliation_maintenance_compute"] for c in cs1]),
        "E1_prefixes": e1_prefix, "E2_prefixes": e2_prefix,
        "E2_sources_by_slot": source_by_slot,
        "trace_cases": cases,
        "target_roles": by_role,
        "target_pairs": {"K_vs_R": pair_compare(arms1[K], arms1[R]),
                         "K_vs_cold": pair_compare(arms1[K], arms1[COLD]),
                         "K_vs_removed": pair_compare(arms1[K], arms1[RM]),
                         "K_vs_Ordinary": pair_compare(arms2[K2], arms2[ORD])},
        "nonpassing_unique_runs": [compact(e) for e in unique_run_eps if not gates(e)["strict_pass"]],
        "clusters": cluster_rows,
        "interpretation_limits": ["n=6 source clusters, not independent n=36 target samples",
                                  "E1 and E2 share the exact same K source+target trajectory; no cross-study independent replication",
                                  "E1 removal turns on compulsory entry-time full raw-history reconstruction; 5/36 does not isolate information removal from this policy change",
                                  "Target success and total end-to-end cost are separate estimands; source S3 savings cannot be relabeled as unseen target gains",
                                  "Same-success costs are descriptive postselection, not a causal treatment estimate",
                                  "Seven reported cost buckets overlap; ledger total remains the usable total"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "counts": outcomes["episode_counts"],
                      "all": {k: outcomes["all_unique_run_summary"][k] for k in ("n","strict_pass","false_internal_confirmation","unresolved_or_unconfirmed","stored_gate_mismatches")},
                      "E1": {a: {k: outcomes["E1_target_summaries"][a][k] for k in ("n","strict_pass","accurate_external","false_internal_confirmation")} for a in arms1},
                      "E2_source": {a: outcomes["E2_source_summaries"][a]["strict_pass"] for a in src2},
                      "E2_target": {a: outcomes["E2_target_summaries"][a]["strict_pass"] for a in arms2}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
