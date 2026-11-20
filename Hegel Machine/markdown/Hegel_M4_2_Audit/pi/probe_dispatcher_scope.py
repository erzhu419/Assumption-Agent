"""Production-forest controls, with all mutations limited to in-memory copies.

Input is the same unmodified builder result shared by the independent evaluation auditor.
No source, committed evidence, or paid APIs are changed/called.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle

from hegel_machine.cognition import m4_2_unified_closure_and_pi_lifecycle as mod
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex

OUT = Path(__file__).resolve().parent
BUILT = OUT.parent / "unified" / "shared_built.pkl"
built = pickle.loads(BUILT.read_bytes())
base = built["forest"]
NID = "ode_meso_schur_pi_thermal"
FID = "ode_micro_2mode_node"
TREE = "tree_multiscale_pi_s"
results = {"input": str(BUILT)}


def dispatch(forest, tag, modality="thermal", t=0.12):
    p = {"u_drive": 2.0, "T_horizon": t}
    return mod.ForestScopeDispatcher(forest).dispatch_predict(
        query_id=tag, candidate_tree_id=TREE, features=p,
        target_metadata={"active_arity": 2, "modality": modality},
        ground_truth=mod._eval_pi_s_fine_scalar(p["u_drive"], t, mod.DEFAULT_PI_S_ODE_PARAMS)["y_micro"],
    )


# A: The production post-Stage2 forest vs the R3 verification function's additional in-place restriction.
pre = deepcopy(base)
results["production_optical_before_stream"] = dispatch(pre, "optical_before", "optical", 3.6)
before_contract = deepcopy(pre.nodes[FID]["claim_contract"])
before_version = pre.nodes[FID]["version"]
before_digest = pre.nodes[FID]["semantic_digest"]
stream_with = mod.run_interleaved_zero_fit_dispatcher_stream(pre)
results["optical_after_stream_guard"] = dispatch(pre, "optical_after", "optical", 3.6)
results["stream_mutation"] = {
    "fine_contract_changed": before_contract != pre.nodes[FID]["claim_contract"],
    "fine_version_before": before_version,
    "fine_version_after": pre.nodes[FID]["version"],
    "semantic_digest_unchanged": before_digest == pre.nodes[FID]["semantic_digest"],
    "added_disjunctive_branches": pre.nodes[FID]["claim_contract"].get("disjunctive_branches"),
}

# B: Dispatcher does not traverse or validate coarse edges; delete all of them only for this control.
without = deepcopy(base)
without.coarse_graining_edges = []
stream_without = mod.run_interleaved_zero_fit_dispatcher_stream(without)
signature = lambda rep: [
    {k: x.get(k) for k in ("query_id", "status", "selected_node_id", "selected_version", "prediction", "escalated_from_coarse_node")}
    for x in rep["receipts"]
]
results["dispatcher_edges_deleted"] = {
    "edge_count_before": len(base.coarse_graining_edges),
    "edge_count_after": len(without.coarse_graining_edges),
    "identical_routing_and_prediction": signature(stream_with) == signature(stream_without),
    "with_edges": stream_with,
    "without_edges": stream_without,
}

# C: Legitimate narrow-domain certificate, but accidentally widened ScaleMap. No forged measurements.
cert = json.loads((OUT / "projection_dependency_results.json").read_text())["baseline"]
widened = deepcopy(base)
new_sm = deepcopy(widened.nodes[NID]["scale_map"])
new_sm["window_or_partition"]["verified_input_box"]["T_horizon"] = [0.05, 5.2]
reval = widened.revalidate_vertical_edges_with_certificate(
    NID, witness_episode_id="audit_legitimate_narrow_cert_broader_scope",
    closure_certificate=cert, updated_scale_map=new_sm,
)
results["certificate_scope_binding"] = {
    "legitimate_certificate_box": cert["verified_input_box"],
    "certificate_defect": cert["max_closure_defect"],
    "requested_new_box": new_sm["window_or_partition"]["verified_input_box"],
    "revalidation": reval,
    "original_short_query": dispatch(deepcopy(base), "quench_original"),
    "widened_short_query": dispatch(widened, "quench_widened"),
    "old_node_box": base.nodes[NID]["scale_map"]["window_or_partition"]["verified_input_box"],
    "current_edge_box_after_revalidation": [e["scale_map"]["window_or_partition"].get("verified_input_box") for e in widened.coarse_graining_edges if e["coarse_node_id"] == NID],
}

# D: Real Stage3 caller. The ordinary current suite is in-box and uses the exact certified coordinate names.
# One continuation query at T=.12 changes only evaluation points; the oracle/mechanism/fit protocol remain.
targets, _ = mod.build_m4_2_target_suite(seed=20_261_120)
t = next(x for x in targets if x.mechanism_family == "PiS_Adiabatic_CrossDomain_2D")


def run_stage3(target, name):
    f = deepcopy(base)
    ix = HippoRAGWormholeIndex(f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
    return mod.evaluate_m4_2_target_under_arm(
        target, arm_name=name, seed=20_261_120,
        work_dir=OUT / "stage3_scope_workspaces" / name,
        forest=f, index=ix, ordinary_history_specs=None,
        fallback_pool_specs=built["ordinary_history_forward"],
    )


stage3 = {}
stage3["unchanged_production_target"] = run_stage3(t, "unchanged")
outside = replace(t, heldout_points=[{"u_drive": 2.0, "T_horizon": 0.12}])
stage3["out_of_box_heldout_only"] = run_stage3(outside, "heldout_outside")

# E: Semantically identical role rename. Confirm T=1.2 is outside Pi certificate but physically valid.
conf_target = replace(t, feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": (0.05, 5.2)},
                      confirm_points=[{"u_drive": 2.6, "T_horizon": 1.2}, t.confirm_points[1]])
stage3["named_out_of_box_confirm"] = run_stage3(conf_target, "named_confirm_outside")
rename = lambda p: {"u_probe": p["u_drive"], "t_probe": p["T_horizon"]}
orig_oracle = conf_target.oracle_fn
alias_target = replace(
    conf_target, feature_names=("u_probe", "t_probe"),
    feature_bounds={"u_probe": (0.4, 3.0), "t_probe": (0.05, 5.2)},
    pilot_points=[rename(p) for p in conf_target.pilot_points],
    confirm_points=[rename(p) for p in conf_target.confirm_points],
    heldout_points=[rename(p) for p in conf_target.heldout_points],
    oracle_fn=lambda p: orig_oracle({"u_drive": p["u_probe"], "T_horizon": p["t_probe"]}),
)
stage3["renamed_out_of_box_confirm"] = run_stage3(alias_target, "renamed_confirm_outside")
stage3["alias_box_helper_result"] = mod.ForestScopeDispatcher._point_in_box(
    {"u_probe": 2.6, "t_probe": 1.2}, base.nodes[NID]["scale_map"]["window_or_partition"]["verified_input_box"]
)
results["stage3_scope"] = stage3

results["production_pi_provenance"] = {
    "node_version": base.nodes[NID]["version"],
    "node_box": base.nodes[NID]["scale_map"]["window_or_partition"]["verified_input_box"],
    "node_artifact": base.nodes[NID]["claim_contract"]["conclusion_or_action"]["authored_artifact_path"],
    "edge_records": [e for e in base.coarse_graining_edges if e["coarse_node_id"] == NID],
}
(OUT / "dispatcher_scope_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps({
    "optical_before": results["production_optical_before_stream"],
    "optical_after": results["optical_after_stream_guard"],
    "stream_mutation": results["stream_mutation"],
    "edge_deletion_identical": results["dispatcher_edges_deleted"]["identical_routing_and_prediction"],
    "certificate_scope_binding": results["certificate_scope_binding"],
    "stage3_scope": stage3,
}, ensure_ascii=False, indent=2))
