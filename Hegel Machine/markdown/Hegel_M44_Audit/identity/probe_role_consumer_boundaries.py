"""Finite M4.4 role / selected-action checks from committed immutable snapshots.

This script modifies only independently loaded in-memory forests and its own output
directory. It never builds a forest, replaces a repository file, or calls a model API.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(os.environ.get("HEGEL_AUDIT_REPO", "/workspace/scratch/3f14ad2da0ed/repo_m44_b965f022b/Hegel Machine")).resolve()
if (ROOT / "Hegel Machine" / "src").is_dir():
    ROOT /= "Hegel Machine"
OUT = Path(os.environ.get("HEGEL_AUDIT_OUT", "/workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/identity")).resolve()
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition import m4_2_unified_closure_and_pi_lifecycle as m42
from hegel_machine.cognition import m4_3_channel_bound_lifecycle_and_audit_closure as m43
from hegel_machine.cognition import m4_4_instance_role_closure_and_online_writeback as m44
from hegel_machine.cognition import m4_compositional_discovery as cd
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m4_multiscale_renormalization import MultiScaleWorkspaceTaskLab
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use

BUNDLE = ROOT / "docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120"


def load_forest(name="post_stage2_forest_snapshot.json"):
    return m44.load_location_independent_forest(BUNDLE / "relocatable_forest_snapshots" / name, BUNDLE)


def evaluate(target, label, forest=None):
    forest = load_forest() if forest is None else forest
    index = HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
    acqs, fitted, loads, gates = [], [], [], []
    orig_acq = MultiScaleWorkspaceTaskLab.acquire_channel
    orig_fit = m43._autonomous_fit_candidate_specs
    orig_load = m43._load_node_verified_operator_spec
    orig_gate = m43.authorize_abstraction_use

    def spy_acq(self, points, *, channel="coarse_macro", phase="pilot"):
        r = orig_acq(self, points, channel=channel, phase=phase)
        acqs.append({"phase": phase, "channel": channel, "points": deepcopy(points)})
        return r

    def spy_fit(*args, **kwargs):
        r, cnt = orig_fit(*args, **kwargs)
        fitted.append({"allowed_role_orders": kwargs.get("allowed_role_orders"), "count": cnt,
                       "best_role_order": r.get("role_order") if r else None,
                       "train_rmse": r.get("train_rmse") if r else None})
        return r, cnt

    def spy_load(node, **kwargs):
        r = orig_load(node, **kwargs)
        a = node["claim_contract"]["conclusion_or_action"]
        loads.append({"node_id": node["node_id"], "requested_instance": a.get("confirmed_instance_digest"),
                      "path": a.get("authored_artifact_path"), "success": r is not None})
        return r

    def spy_gate(**kwargs):
        r = orig_gate(**kwargs)
        a = r.get("selected_action") or {}
        gates.append({"claim_id": kwargs["claim_contract"]["claim_id"], "tier": kwargs["requested_tier"],
                      "granted": r["granted"], "selected_instance": a.get("confirmed_instance_digest")})
        return r

    with patch.object(MultiScaleWorkspaceTaskLab, "acquire_channel", spy_acq), \
            patch.object(m43, "_autonomous_fit_candidate_specs", spy_fit), \
            patch.object(m43, "_load_node_verified_operator_spec", spy_load), \
            patch.object(m43, "authorize_abstraction_use", spy_gate):
        rec = m43.evaluate_m4_3_target_under_arm(
            target, arm_name=label, seed=20301120, work_dir=OUT / label,
            forest=forest, index=index, ordinary_history_specs=None,
            fallback_pool_specs=[], allow_pool_fallback=False, open_code_agent_fallback=False)
    return {"receipt": rec, "acquisitions": acqs, "fits": fitted, "loads": loads, "gates": gates,
            "scope": json.loads((OUT / label / "model_scope_contract.json").read_text())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {"commit": "b965f022b214f47a1341559f20041da6a224d66d", "no_build": True,
                "immutable_bundle": str(BUNDLE), "online_model_calls": 0}
    forest = load_forest()
    anchors = []
    for nid, node in sorted(forest.nodes.items()):
        cc = node["claim_contract"]
        for name, action in [("top", cc["conclusion_or_action"]),
                             *[(f"branch_{i}", b["conclusion_or_action"])
                               for i, b in enumerate(cc.get("disjunctive_branches", []))]]:
            if not action.get("authored_artifact_path"):
                continue
            p = forest.resolve_artifact_path(action["authored_artifact_path"])
            loaded = cd.load_verified_authored_method(p,
                    expected_ast_digest=action.get("ast_skeleton_digest"),
                    expected_instance_digest=action.get("confirmed_instance_digest"),
                    require_external_manifest=True, verify_frozen_only=True)
            wrong = cd.load_verified_authored_method(p,
                    expected_ast_digest=action.get("ast_skeleton_digest"),
                    expected_instance_digest="wrong_instance_anchor", require_external_manifest=True,
                    verify_frozen_only=True)
            anchors.append({"node_id": nid, "action": name, "has_expected_instance": bool(action.get("confirmed_instance_digest")),
                            "frozen_verified": loaded is not None, "wrong_expected_rejected": wrong is None,
                            "mode": loaded.get("_verification_mode") if loaded else None})
    evidence["anchors"] = anchors

    # Canonical names versus an explicitly contradictory caller-provided role map.
    pt = {"u_drive": 2.7, "T_horizon": 0.47}
    truth = float(m42._eval_pi_s_fine_scalar(pt["u_drive"], pt["T_horizon"])["y_micro"])
    dispatcher = m42.ForestScopeDispatcher(forest)
    tree = forest.nodes["ode_meso_schur_pi_thermal"]["tree_id"]
    shared = {"candidate_tree_id": tree, "features": pt,
              "target_metadata": {"active_arity": 2, "modality": "thermal"}, "ground_truth": truth}
    evidence["r3_roles"] = {
        "normal_canonical": dispatcher.dispatch_predict(query_id="canonical_without_map", **shared),
        "contradictory_canonical_map": dispatcher.dispatch_predict(query_id="canonical_with_swapped_map", **shared,
                role_map={"u_drive": "T_horizon", "T_horizon": "u_drive"}),
        "valid_anonymous_map": dispatcher.dispatch_predict(query_id="anonymous_short_time", candidate_tree_id=tree,
                features={"x0": 2.7, "x1": 0.47}, target_metadata={"active_arity": 2, "modality": "thermal"},
                role_map={"x0": "u_drive", "x1": "T_horizon"}, ground_truth=truth),
    }

    base = m43.build_m4_3_target_suite(seed=20301120)[0][20]
    short_pilot = [(2.45, .40), (2.90, .52), (2.55, .67), (2.80, .85), (2.65, 1.05), (2.95, 1.20)]
    short_confirm = [(2.70, .47)]
    short_heldout = [(2.50, .42), (2.85, .55), (2.75, .75)]
    short_evidence = {}
    for label, names in [("partial_canonical_names", ("u_drive", "x1")),
                         ("anonymous_short_time", ("x0", "x1"))]:
        def named(pts):
            return [{names[0]: x, names[1]: y} for x, y in pts]
        target = replace(base, target_id=label, feature_names=names,
            feature_bounds={names[0]: (2.4, 3.0), names[1]: (.4, 1.3)},
            pilot_points=named(short_pilot), confirm_points=named(short_confirm), heldout_points=named(short_heldout),
            oracle_fn=lambda p, ns=names: base.oracle_fn({"u_drive": p[ns[0]], "T_horizon": p[ns[1]]}),
            coarse_oracle_fn=lambda p, ns=names: base.coarse_oracle_fn({"u_drive": p[ns[0]], "T_horizon": p[ns[1]]}))
        short_evidence[label] = evaluate(target, label)
    evidence["anonymous_and_partial_alias_short_time"] = short_evidence

    # Real two-instance OR contract: initial thermal instance versus re-certified hydraulic instance.
    # No artifact bytes are changed; both files are original, host-confirmed archived instances.
    or_f = load_forest()
    s0 = load_forest("true_stage0_forest_snapshot.json")
    nid = "ode_meso_schur_pi_thermal"
    cc = or_f.nodes[nid]["claim_contract"]
    current_action = deepcopy(cc["conclusion_or_action"])
    stage0_action = deepcopy(s0.nodes[nid]["claim_contract"]["conclusion_or_action"])
    cc["conclusion_or_action"] = stage0_action
    for branch in cc["disjunctive_branches"]:
        conditions = branch["preconditions"]
        modality = next((c["value"] for c in conditions if c["field"] == "modality"), None)
        branch["conclusion_or_action"] = deepcopy(current_action if modality == "hydraulic" else stage0_action)
    evidence["or_distinct_valid_instances"] = {
        "scope_limit": "Exploratory in-memory Pi fixture only: production coarse-edge revalidation requires all OR branches to share the certificate instance. Do not treat this two-instance Pi forest as a production-reachable certified state. Use generic_or_actions.json for the valid ordinary-OR integration control.",
        "top_instance": stage0_action["confirmed_instance_digest"],
        "hydraulic_branch_instance": current_action["confirmed_instance_digest"],
        "r3": m42.ForestScopeDispatcher(or_f).dispatch_predict(query_id="or_hydraulic_branch", candidate_tree_id=tree,
                features={"u_drive": 1.6, "T_horizon": 3.2}, target_metadata={"active_arity": 2, "modality": "hydraulic"}),
        "m43": evaluate(base, "or_two_genuine_instances", or_f),
    }
    # Availability boundary: only the unselected thermal branch's artifact locator is unavailable.
    # The selected hydraulic branch and its certified instance remain byte-for-byte intact.
    unavailable_a = deepcopy(or_f)
    unavailable_a.nodes[nid]["claim_contract"]["conclusion_or_action"]["authored_artifact_path"] = str(
        OUT / "intentionally_absent_unselected_thermal_artifact.json")
    evidence["or_unselected_artifact_unavailable"] = {
        "scope_limit": "Exploratory Pi state inherits the mixed-instance certificate limitation above; generic_or_actions.json provides the valid ordinary-OR control.",
        "boundary": "Independent OR branch must not require another branch's absent artifact",
        "r3": m42.ForestScopeDispatcher(unavailable_a).dispatch_predict(
            query_id="or_hydraulic_unselected_thermal_missing", candidate_tree_id=tree,
            features={"u_drive": 1.6, "T_horizon": 3.2},
            target_metadata={"active_arity": 2, "modality": "hydraulic"}),
        "m43": evaluate(base, "or_unselected_thermal_missing", unavailable_a),
    }
    (OUT / "role_consumer_boundaries.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"anchors": len(anchors), "all_anchors_pass": all(a["frozen_verified"] and a["wrong_expected_rejected"] for a in anchors),
                      "r3_roles": {k: {f: v.get(f) for f in ("status", "selected_node_id", "prediction", "abs_error", "within_tolerance")}
                                   for k, v in evidence["r3_roles"].items()},
                      "short_time": {k: v["receipt"] for k,v in short_evidence.items()},
                      "or_m43": evidence["or_distinct_valid_instances"]["m43"]["receipt"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
