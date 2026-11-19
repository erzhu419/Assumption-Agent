"""Read-only audit controls for commit 11961f702, independent scratch outputs.

Run from Hegel Machine with PYTHONPATH=src. No source modifications, remote calls,
or complete research study. All functions use real paid-lab simulator interfaces.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import time
from unittest.mock import patch

import hegel_machine.cognition.m4_multiscale_renormalization as m
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex

OUT = Path(__file__).resolve().parent
WORK = OUT / "probe_workspaces"


def clean_result(r):
    r = deepcopy(r)
    r["cost"].pop("wall_clock_seconds", None)
    return r


def evaluate(target, forest, label, *, no_cg=False, arm="Hegel_M4_MultiScale_HippoWalk"):
    idx = HippoRAGWormholeIndex(
        forest, damping_alpha=0.65, wormhole_equivalence_pooling=True,
        scale_aware=True, include_coarse_graining_edges=not no_cg,
    )
    r = m.evaluate_m4_target_under_arm(
        target, arm_name=arm, forest=forest, index=idx, stage1_gate=None,
        work_dir=WORK / label,
    )
    return clean_result(r)


def method_audit_evaluator(node, q):
    """Exact evaluator used by the lifecycle study (copied with line reference)."""
    req_ar = int(node["observation_map"]["parameter_semantics"]["active_arity"])
    if q.get("active_arity") != req_ar:
        return "UNK"
    for pre in node["claim_contract"].get("preconditions", ()):
        f_k = pre["field"]
        if f_k not in q:
            return "UNK"
        if (q[f_k] == pre["value"]) is not (pre.get("op", "==") == "=="):
            return "REJ"
    return "ACC"


def main():
    started = time.perf_counter()
    forest, source_meta = m.build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=12)
    targets, _ = m.build_m4_cluster_targets(cluster_idx=0)
    chosen = [targets[i] for i in (0, 3, 6, 10, 14)]
    out = {"commit": "11961f702739ab272661ff59b77fb5bc8e3589b8", "source_meta": source_meta}

    # Positive check: real closure values change with empirical point selection.
    positive = []
    for t in chosen:
        lab = deepcopy(t.lab)
        probe = lab.run_active_scale_closure_probe(t.scale_boundary_point)
        positive.append({"target": t.target_id, "probe": probe, "query_log": lab.query_log})
    out["real_paired_closure_probes"] = positive

    # Keep all contracts, graphs, targets identical; replace only instruction text.
    nonsense = deepcopy(forest)
    for nd in nonsense.nodes.values():
        nd["program_steps"] = [{"step": "NOT_AN_IMPLEMENTED_OPCODE", "fresh_points": -999}]
    controls = []
    for t in chosen:
        base = evaluate(t, forest, "base_" + t.target_id)
        absurd = evaluate(t, nonsense, "nonsense_" + t.target_id)
        no_cg = evaluate(t, forest, "no_cg_" + t.target_id, no_cg=True)
        controls.append({
            "target": t.target_id, "base": base, "nonsense_program_steps": absurd,
            "nonsense_same_result_except_runtime": base == absurd,
            "without_coarse_graining_edges": no_cg,
        })
    out["program_and_coarse_edge_controls"] = controls

    # The child success path should require the same final authorization receipt.
    original_auth = m.authorize_abstraction_use
    auth_calls = []
    def deny_final(**kwargs):
        if kwargs.get("requested_tier") == "local_solver_guidance":
            auth_calls.append({"tier": kwargs["requested_tier"], "granted": False})
            return {"granted": False, "reason": "audit_denies_final_permission"}
        res = original_auth(**kwargs)
        auth_calls.append({"tier": kwargs["requested_tier"], "granted": res.get("granted")})
        return res
    with patch.object(m, "authorize_abstraction_use", deny_final):
        child_result = evaluate(targets[10], forest, "deny_final_walk_child")
    out["final_authorization_denied_child_path"] = {"result": child_result, "auth_calls": auth_calls}

    hard_scope_forest = deepcopy(forest)
    child_cc = hard_scope_forest.nodes["bio_vascular_macro_capped"]["claim_contract"]
    child_cc["preconditions"].append({"field": "audit_boundary_measured", "op": "==", "value": True})
    scoped_child_result = evaluate(targets[10], hard_scope_forest, "unverified_child_scope")
    model_sha = hashlib.sha256((WORK / "unverified_child_scope" / "cand_00.py").read_bytes()).hexdigest()
    child_node = hard_scope_forest.nodes["bio_vascular_macro_capped"]
    auth = original_auth(
        requested_tier="local_solver_guidance",
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": targets[10].lab.tolerance,
                   "target_id": targets[10].target_id, "node_version": child_node["version"],
                   "expected_model_sha256": model_sha, "expected_confirmation_digest": "audit_fresh_confirmation"},
        target_features={"active_arity": 1, "scale_closure_verified": True, "high_flow_ceiling_regime": True},
        claim_contract=child_cc, scale_map=child_node["scale_map"],
        evidence_receipt={"passed": True, "observed_error": 0.0, "claim_id": child_cc["claim_id"],
                          "target_id": targets[10].target_id, "node_version": child_node["version"],
                          "model_sha256": model_sha, "confirmation_digest": "audit_fresh_confirmation"},
    )
    out["unverified_child_scope_bypass"] = {"result": scoped_child_result, "direct_final_authorization": auth}

    # Fair frozen memory retains exactly the Stage-1 contents; compare the
    # study's manually deleted child under the same arm code and target.
    full_base, _ = m.build_m4_multiscale_forest(cluster_seed=9101, extra_distractor_count=6)
    evals9, _ = m.build_m4_cluster_targets(cluster_idx=9)
    freeze_target = evals9[10]
    fair_frozen = evaluate(freeze_target, full_base, "fair_frozen_walk", arm="Frozen_Stage1_Memory")
    study_frozen = full_base.without_node_version("bio_vascular_macro_capped", 1)
    unequal_frozen = evaluate(freeze_target, study_frozen, "study_frozen_walk", arm="Frozen_Stage1_Memory")
    out["frozen_starting_information_control"] = {"full_memory_result": fair_frozen, "deleted_child_result": unequal_frozen}

    # Synthetic narrow violation: closure agrees at the declared boundary and
    # confirmation locations but differs at an independently evaluated region.
    # This is a scope test, not a claim that the committed benchmark has this data.
    shifted = deepcopy(targets[0])
    orig_micro = shifted.lab._micro_fn
    shifted.lab._micro_fn = lambda p: orig_micro(p) + math.exp(-((p["q_flow"] - 5.8) / 0.20) ** 2)
    shifted.target_id += "_audit_local_micro_bump"
    shifted.lab.task_id = shifted.target_id
    lab = deepcopy(shifted.lab)
    shifted_probe = lab.run_active_scale_closure_probe(shifted.scale_boundary_point)
    shifted_result = evaluate(shifted, forest, "local_micro_bump")
    out["single_point_scope_counterexample"] = {"probe": shifted_probe, "result": shifted_result}

    # Exact new revision sequence used at every lifecycle stage. Horizontal
    # revalidation does not rebind the vertical coarse edges' endpoint versions.
    rev = deepcopy(full_base)
    before = HippoRAGWormholeIndex(rev, damping_alpha=0.65).edge_counts
    for i, nid in enumerate(("aq_macro_river_tree", "aq_meso_hex_cell")):
        cc = deepcopy(rev.nodes[nid]["claim_contract"])
        cc["protocol_spec"]["last_verified_stage"] = 2
        ep = f"s2_rev_{i}"
        rev.empirical_revise_node(nid, witness_episode_id=ep, cause="protocol_or_scope_mismatch", updated_claim_contract=cc)
        for wid, wh in list(rev.horizontal_wormhole_edges.items()):
            if wh.get("stale_endpoint_revision"):
                rev.revalidate_wormhole_after_revision(wid, witness_episode_id=ep, reason="verified_pure_ast_and_scope_preserved_at_stage_revision")
    out["revision_coarse_edge_liveness"] = {
        "before": before, "after": HippoRAGWormholeIndex(rev, damping_alpha=0.65).edge_counts,
        "live_fine_children_of_aq_meso": [n["node_id"] for n in rev.fine_scale_children_of("aq_meso_hex_cell")],
        "live_coarse_parents_of_aq_meso": [n["node_id"] for n in rev.coarse_scale_parents_of("aq_meso_hex_cell")],
        "all_horizontal_edges_not_stale": all(not wh.get("stale_endpoint_revision") for wh in rev.horizontal_wormhole_edges.values()),
    }

    # Current evaluator checks only arity + preconditions, not predictive law,
    # scale, observation map, or program. It can certify unrelated same-arity laws.
    bad_merge_f = deepcopy(forest)
    queries = [
        {"active_arity": 1, "scale_closure_verified": True, "high_flow_ceiling_regime": False},
        {"active_arity": 1, "scale_closure_verified": True, "high_flow_ceiling_regime": True},
        {"active_arity": 2, "scale_closure_verified": True},
        {"active_arity": 2, "scale_closure_verified": False},
    ]
    bad = bad_merge_f.compress_compatible_nodes(
        keeper_node_id="aq_macro_river_tree", redundant_node_id="dist_node_00",
        evaluation_queries=queries, evaluator_fn=method_audit_evaluator,
    )
    out["fidelity_evaluator_wrong_law_control"] = {
        "keeper_tag": forest.nodes["aq_macro_river_tree"]["claim_contract"]["conclusion_or_action"]["operator_tag"],
        "redundant_tag": forest.nodes["dist_node_00"]["claim_contract"]["conclusion_or_action"]["operator_tag"],
        "accepted": bad["accepted"], "conflicting_verdict_indices": bad["conflicting_verdict_indices"],
    }

    # Wrapper commits signatures and archival but not the merged rule that the
    # fidelity helper actually evaluated. Show loss even with a truthful evaluator.
    branch_f = deepcopy(forest)
    for nid, val in (("aq_macro_river_tree", "A"), ("dist_node_00", "B")):
        branch_f.nodes[nid]["claim_contract"]["audit_case"] = val
    def branch_eval(node, q):
        cc = node["claim_contract"]
        branches = cc.get("disjunctive_branches", [cc])
        return "ACC" if any(b.get("audit_case") == q["case"] for b in branches) else "UNK"
    branch_report = branch_f.compress_compatible_nodes(
        keeper_node_id="aq_macro_river_tree", redundant_node_id="dist_node_00",
        evaluation_queries=[{"case": "A"}, {"case": "B"}], evaluator_fn=branch_eval,
    )
    out["fidelity_committed_node_vs_validated_node"] = {
        "accepted": branch_report["accepted"],
        "validated_merged_B": branch_eval(branch_report["merged_node"], {"case": "B"}),
        "committed_keeper_B": branch_eval(branch_f.nodes["aq_macro_river_tree"], {"case": "B"}),
        "B_source_archived": branch_f.nodes["dist_node_00"]["utility_ledger"]["archived_low_utility"],
    }

    out["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (OUT / "probe_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT / "probe_results.json"), "elapsed_seconds": out["elapsed_seconds"],
        "nonsense_all_equal": all(r["nonsense_same_result_except_runtime"] for r in controls),
        "child_recall_despite_final_deny": child_result["confirmed_via_recall"],
        "single_point_counterexample": shifted_result,
        "fidelity_committed_node_vs_validated_node": out["fidelity_committed_node_vs_validated_node"],
        "revision_coarse_edge_liveness": out["revision_coarse_edge_liveness"],
    }, indent=2))


if __name__ == "__main__":
    main()
