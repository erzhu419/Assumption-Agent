"""Selected-action integration control on ordinary (non-Pi, no scale certificate) OR nodes.

Uses two genuinely calibrated, confirmed same-AST Monod instances ALREADY committed
in b965f022b, and production register_node to preserve semantic digests. No mutation
of any repository file; no new calibration, full study, or online model call.
"""
from copy import deepcopy
import json
from pathlib import Path

from probe_role_consumer_boundaries import ROOT, OUT, BUNDLE, load_forest, evaluate, m42, m43, m44, cd
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest, authorize_abstraction_use


def main():
    original = load_forest()
    node = original.nodes["stage0_monod_2d"]
    action_a = deepcopy(node["claim_contract"]["conclusion_or_action"])
    path_b = ROOT / "markdown/Hegel_M4_3_Audit/instance/same_ast_distinct_gain_calibration/authored_method.json"
    artifact_b = cd.load_verified_authored_method(path_b, require_external_manifest=True, verify_frozen_only=True)
    assert artifact_b is not None
    action_b = m42._make_verified_instance_action(
        operator_tag=action_a["operator_tag"], operator_spec=artifact_b["operator_spec"],
        artifact_dict=artifact_b, artifact_relpath=str(path_b))
    assert action_a["ast_skeleton_digest"] == action_b["ast_skeleton_digest"]
    assert action_a["confirmed_instance_digest"] != action_b["confirmed_instance_digest"]
    cc = deepcopy(node["claim_contract"])
    cc["preconditions"] = [{"field": "active_arity", "op": "==", "value": 2}]
    cc["disjunctive_branches"] = [
        {"preconditions": [{"field": "secondary_bleed_active", "op": "==", "value": False}],
         "conclusion_or_action": action_a, "permission_tier": "local_solver_guidance", "evidence_status": "SUPPORTED_IN_SCOPE"},
        {"preconditions": [{"field": "secondary_bleed_active", "op": "==", "value": True}],
         "conclusion_or_action": action_b, "permission_tier": "local_solver_guidance", "evidence_status": "SUPPORTED_IN_SCOPE"},
    ]

    def registered(claim):
        f = PhilosophicalForest(artifact_root=str(BUNDLE))
        f.register_node(node_id=node["node_id"], tree_id=node["tree_id"], layer_role=node["layer_role"],
            task_view=node["task_view"], observation_map=node["observation_map"], scale_map=node["scale_map"],
            claim_contract=claim, probe_signatures=[*node["probe_signatures"], "sig:cue_or_instance_control"],
            program_steps=node["program_steps"])
        assert m44.verify_forest_snapshot_digests(f)["semantic_digest_mismatch_count"] == 0
        return f

    oracle = lambda p: float(cd.predict_with_frozen_instance(artifact_b, [p])["predictions"][0])
    target = m43.M43TargetSpec(
        target_id="generic_OR_selected_B", mechanism_family="Audit_Same_AST_Different_Instance",
        feature_names=("s", "b"), feature_bounds={"s": (.5, 6.0), "b": (0., 1.)},
        pilot_points=[{"s": s, "b": b} for s,b in [(0.5,.1),(1.,.5),(2.,.3),(3.,.7),(4.,.2),(5.5,.9)]],
        confirm_points=[{"s": 2.5, "b": .45}],
        heldout_points=[{"s": 1.6, "b": .6}, {"s": 3.7, "b": .25}, {"s": 4.5, "b": .85}],
        tolerance=.05, oracle_fn=oracle, coarse_oracle_fn=None, domain_cue="cue_or_instance_control", scale_regime="macro")

    def dispatch(f):
        point = {"s": 1.6, "b": .6}
        return m42.ForestScopeDispatcher(f).dispatch_predict(query_id="generic_or_B", candidate_tree_id=node["tree_id"],
                features=point, target_metadata={"active_arity": 2, "secondary_bleed_active": True}, ground_truth=oracle(point))

    f = registered(cc)
    baseline = {"r3": dispatch(f), "m43": evaluate(target, "generic_OR_intact_A", f)}
    absent_cc = deepcopy(cc)
    absent_cc["conclusion_or_action"]["authored_artifact_path"] = str(OUT / "intentionally_absent_A.json")
    absent_cc["disjunctive_branches"][0]["conclusion_or_action"]["authored_artifact_path"] = str(OUT / "intentionally_absent_A.json")
    absent = registered(absent_cc)
    availability = {"r3": dispatch(absent), "m43": evaluate(target, "generic_OR_unselected_A_unavailable", absent)}
    evidence = {"commit": "b965f022b214f47a1341559f20041da6a224d66d", "fixture": str(path_b),
        "construction": "production register_node; one ordinary node; no coarse edges or Pi certificates",
        "same_ast": True, "distinct_instance": True, "unselected_A_instance": action_a["confirmed_instance_digest"],
        "selected_B_instance": action_b["confirmed_instance_digest"], "baseline": baseline, "availability": availability}
    (OUT / "generic_or_actions.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        label: {"r3": {k: v["r3"].get(k) for k in ("status","prediction","within_tolerance","confirmed_instance_digest")},
                "m43": {k: v["m43"]["receipt"].get(k) for k in ("status","strict_pass","confirmed_via_recall","heldout_rmse")},
                "loads": v["m43"]["loads"], "gates": v["m43"]["gates"]}
        for label, v in [("baseline", baseline), ("availability", availability)]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
