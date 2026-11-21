"""Focused audit controls for commit 63b940a03; no repository modifications."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

from hegel_machine.cognition.wormhole_forest import (
    compute_order_invariant_source_diversity,
    build_warm_start_parameter_prior,
    validate_consistent_joint_subtree_binding,
)
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    serialize_location_independent_forest, verify_forest_snapshot_digests,
)
from hegel_machine.cognition.hipporag_wormhole_index import (
    HippoRAGWormholeIndex, FlatEqualMetadataIndex,
)
from hegel_machine.cognition import m5_temporal_multiscale_and_base_subtree as m5


out = {}
orig_sources = [
    {"source_id": "a1", "provenance_lineage_id": "A", "surface_cues": ["same"]},
    {"source_id": "b1", "provenance_lineage_id": "B", "surface_cues": ["same"]},
]
clone_sources = [*deepcopy(orig_sources),
    {"source_id": "a2", "provenance_lineage_id": "A", "surface_cues": ["renamed"]}]
orig_div = compute_order_invariant_source_diversity(orig_sources)
clone_div = compute_order_invariant_source_diversity(clone_sources)
out["lineage_clone_changes_diversity"] = {
    "original": orig_div, "same_lineage_renaming_added": clone_div,
    "original_prior_mode": build_warm_start_parameter_prior(
        [{"p": .4}, {"p": .5}], diversity_metrics=orig_div,
        hard_domain_bounds={"p": (0., 1.)})["mode"],
    "clone_prior_mode": build_warm_start_parameter_prior(
        [{"p": .4}, {"p": .5}], diversity_metrics=clone_div,
        hard_domain_bounds={"p": (0., 1.)})["mode"],
}

def star_sources(sim):
    return [
        {"provenance_lineage_id": "A", "pairwise_surface_sim": {"B": sim, "C": sim}},
        {"provenance_lineage_id": "B", "pairwise_surface_sim": {"C": 0.0}},
        {"provenance_lineage_id": "C"},
    ]

out["non_psd_similarity"] = {
    "all_pairwise_similarities_in_unit_interval": True,
    "star_similarity_1": compute_order_invariant_source_diversity(star_sources(1.0)),
    "star_similarity_087": compute_order_invariant_source_diversity(star_sources(.87)),
}
try:
    out["non_psd_similarity"]["singular_case"] = compute_order_invariant_source_diversity(
        star_sources(1.25 / math.sqrt(2)))
except Exception as exc:
    out["non_psd_similarity"]["singular_case"] = {
        "exception": type(exc).__name__, "message": str(exc)}

forest, priors = m5._build_seed_forest_and_priors()
sub = forest.shared_base_subtrees["sub_2layer_siphon_base"]
roles = {"u_drive": "u_now", "x_pool": "x_now", "s_grad": "s_grad", "y_out": "y"}
out["joint_binding"] = {
    "empty_confirmed_condition_list": validate_consistent_joint_subtree_binding(
        sub, {"role_bindings": roles, "satisfied_shared_node_ids": []}),
    "only_c1_claimed": validate_consistent_joint_subtree_binding(
        sub, {"role_bindings": roles, "satisfied_shared_node_ids": ["C1_base_capacity_field"]}),
    "no_relation_edges_bound": validate_consistent_joint_subtree_binding(
        sub, {"role_bindings": roles,
              "satisfied_shared_node_ids": ["C1_base_capacity_field", "C2_adjacent_high_gradient_sink"],
              "edge_bindings": []}),
    "missing_intermediate_role": validate_consistent_joint_subtree_binding(
        sub, {"role_bindings": {k:v for k,v in roles.items() if k != "x_pool"},
              "satisfied_shared_node_ids": ["C1_base_capacity_field", "C2_adjacent_high_gradient_sink"]}),
}

snapshot = serialize_location_independent_forest(forest)
snapshot["shared_base_subtrees"]["sub_2layer_siphon_base"]["diversity_metrics"]["d_eff_set"] = 1000000.0
snapshot["shared_base_subtrees"]["sub_2layer_siphon_base"]["supporting_sources"] = []
snapshot["shared_base_subtrees"]["sub_2layer_siphon_base"]["probe_signatures"] = ["arbitrary_changed_signature"]
out["persistence_uncaptured_fields"] = verify_forest_snapshot_digests(snapshot)
forest.revise_shared_base_subtree("sub_2layer_siphon_base", witness_episode_id="audit_only",
                                  cause="protocol_or_scope_mismatch")
snapshot2 = serialize_location_independent_forest(forest)
snapshot2["shared_base_subtree_history"]["sub_2layer_siphon_base"][0]["shared_relations"] = []
out["history_tamper_undetected"] = verify_forest_snapshot_digests(snapshot2)

ds = m5._build_disjoint_dataset(
    "audit_same_task", "audit", "C_two_layer_conjunctive_subtree", False,
    ["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"], False,
    {"vmax": 6.8, "rho": 0.70, "local_branch_coeff": .4, "tau_scale": 1.0}, random.Random(123))

def run_one(mutate):
    f, p = m5._build_seed_forest_and_priors()
    mutate(f.shared_base_subtrees["sub_2layer_siphon_base"])
    warm_calls = []
    orig_fit = m5.fit_with_soft_warm_start_and_wide_fallback
    def record_fit(*args, **kwargs):
        warm_calls.append({"warm_start_box": kwargs["prior_spec"].get("warm_start_box")})
        return orig_fit(*args, **kwargs)
    m5.fit_with_soft_warm_start_and_wide_fallback = record_fit
    try:
        receipt = m5._run_genuine_solver_pipeline(
            ds, "Hegel_M5_Full_TemporalSubtree_Forest", forest=f,
            priors_bundle=p, hippo_idx=HippoRAGWormholeIndex(f),
            flat_idx=FlatEqualMetadataIndex(f), rng=random.Random(123))
    finally:
        m5.fit_with_soft_warm_start_and_wide_fallback = orig_fit
    return {"receipt": receipt, "warm_fit_calls": warm_calls}

baseline = run_one(lambda sb: None)
mutated = run_one(lambda sb: sb.update({
    "permission_tier": "retrieve_candidate", "stale_endpoint_revision": True,
    "open_completion_slots": [], "shared_relations": [], "supporting_sources": [],
    "diversity_metrics": {"independent_lineage_count": 0, "d_eff_set": 0},
}))
out["stale_retrieve_only_empty_subtree_consumer"] = {
    "baseline": baseline, "mutated": mutated,
    "receipt_and_warm_calls_identical": baseline == mutated,
}

dest = Path(__file__).with_name("contracts_controls_results.json")
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
