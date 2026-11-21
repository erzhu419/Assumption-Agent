"""Ablate only restored M5 state; run three small target continuations, not full J."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess

from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    load_location_independent_forest, verify_forest_snapshot_digests,
)
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex, FlatEqualMetadataIndex
from hegel_machine.cognition import m5_temporal_multiscale_and_base_subtree as m5

repo_root = Path(os.environ.get('HEGEL_AUDIT_REPO_ROOT') or subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel'], text=True).strip())
evidence_root = Path(os.environ.get('HEGEL_AUDIT_EVIDENCE_DIR') or (
    repo_root / 'Hegel Machine/docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120'))
snapshot_path = evidence_root / 'track1_anonymous_benchmark/joint_stream_J/relocatable_m5_joint_snapshot/post_j16_m5_forest_snapshot.json'

def restore():
    return load_location_independent_forest(snapshot_path, snapshot_path.parent)

rng = random.Random(999)
j17 = m5._build_disjoint_dataset('J17_Cold_TwoLayerReuse', 'J_Cold', 'C_two_layer_conjunctive_subtree', False, ['sig:base_field_C1', 'sig:sink_gradient_C2', 'sig:monotone_concave'], False, {'vmax': 6.4, 'rho': 0.70, 'local_branch_coeff': 0.38, 'tau_scale': 1.0}, rng)
j18 = m5._build_disjoint_dataset('J18_Cold_MemoryKernel', 'J_Cold', 'B_memory_kernel_fold', False, ['sig:slow_horizon_query', 'sig:coupled_transient_lag'], True, {'a_slow': 1.8, 'b_couple': 1.2, 'c_drive': 1.3, 'tau_fast': 0.62, 'g_feedback': 0.52, 'horizon_H': 0.42}, rng)
j19 = m5._build_disjoint_dataset('J19_Cold_LeakReuse', 'J_Cold', 'C_upstream_premise_failure', False, ['sig:base_field_C1', 'sig:sink_gradient_C2', 'sig:mass_sum_drift'], False, {'vmax': 6.9, 'rho': 0.70, 'external_leak_rate': 0.72}, rng)

def run(forest, ds):
    # This is the same cold prior constructor as the production subprocess, line 2036.
    _, priors = m5._build_seed_forest_and_priors()
    calls = []
    fit_orig = m5.fit_with_soft_warm_start_and_wide_fallback
    def fit_capture(*args, **kwargs):
        result = fit_orig(*args, **kwargs)
        calls.append({"prior_spec": deepcopy(kwargs["prior_spec"]),
                      "used_warm_start": result["used_warm_start"],
                      "escaped_to_wide_fallback": result["escaped_to_wide_fallback"]})
        return result
    m5.fit_with_soft_warm_start_and_wide_fallback = fit_capture
    try:
        receipt = m5._run_genuine_solver_pipeline(
            ds, 'Hegel_M5_Full_TemporalSubtree_Forest', forest=forest,
            priors_bundle=priors, hippo_idx=HippoRAGWormholeIndex(forest),
            flat_idx=FlatEqualMetadataIndex(forest), rng=random.Random(999))
    finally:
        m5.fit_with_soft_warm_start_and_wide_fallback = fit_orig
    return {"receipt": receipt, "fit_calls": calls}

base = restore()
erased = restore()
old_slots = erased.shared_base_subtrees['sub_2layer_siphon_base']['open_completion_slots']
remaining_slots = [s for s in old_slots if s.get('branch_id') != 'confirmed_external_sink_leak']
erased.revise_shared_base_subtree(
    'sub_2layer_siphon_base', witness_episode_id='audit_remove_J09_branch',
    cause='protocol_or_scope_mismatch', updated_open_completion_slots=remaining_slots)
del erased.shared_base_subtrees['sub_joint_fast_slow']

j19_base, j19_erased = run(base, j19), run(erased, j19)
out = {
    "source_snapshot": str(snapshot_path),
    "baseline_digest": verify_forest_snapshot_digests(base),
    "erased_digest": verify_forest_snapshot_digests(erased),
    "deleted_J09_branch_count": len(old_slots) - len(remaining_slots),
    "joint_subtree_deleted": 'sub_joint_fast_slow' not in erased.shared_base_subtrees,
    "j19_original": j19_base, "j19_erased": j19_erased,
    "j19_identical_receipts_and_fit_calls": j19_base == j19_erased,
}

metadata_erased = restore()
sb = metadata_erased.shared_base_subtrees['sub_2layer_siphon_base']
sb['supporting_sources'] = []
sb['diversity_metrics'] = {"independent_lineage_count": 0, "d_eff_set": 0.0}
sb['unique_evidence_dependency_ids'] = []
j17_base, j17_erased = run(restore(), j17), run(metadata_erased, j17)
out.update({
    "metadata_erased_digest": verify_forest_snapshot_digests(metadata_erased),
    "j17_original": j17_base, "j17_source_metadata_erased": j17_erased,
    "j17_identical_receipts_and_fit_calls": j17_base == j17_erased,
    "snapshot_contains_parameter_prior": any('warm_start' in str(k) or 'parameter_prior' in str(k)
                                              for k in sb),
})

# Exercise public revision API to obtain a digest-valid low-permission subtree.
low_permission = restore()
low_permission.revise_shared_base_subtree(
    'sub_2layer_siphon_base', witness_episode_id='audit_permission_downgrade',
    cause='protocol_or_scope_mismatch', updated_permission_tier='retrieve_candidate')
j17_low = run(low_permission, j17)
out['permission_downgrade'] = {
    "new_permission_tier": low_permission.shared_base_subtrees['sub_2layer_siphon_base']['permission_tier'],
    "digest": verify_forest_snapshot_digests(low_permission),
    "result": j17_low, "identical_to_baseline": j17_low == j17_base,
}

Path(__file__).with_name('cold_subtree_controls_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({
    "j19_identical_receipts_and_fit_calls": out['j19_identical_receipts_and_fit_calls'],
    "j19_original_receipt": j19_base['receipt'],
    "j19_erased_receipt": j19_erased['receipt'],
    "j17_identical_receipts_and_fit_calls": out['j17_identical_receipts_and_fit_calls'],
    "j17_original_receipt": j17_base['receipt'],
    "j17_source_metadata_erased_receipt": j17_erased['receipt'],
    "low_permission_identical_to_baseline": j17_low == j17_base,
    "all_changed_snapshots_have_zero_digest_mismatches": all(
        v.get('subtree_digest_mismatch_count') == 0 for v in [
            out['erased_digest'], out['metadata_erased_digest'], out['permission_downgrade']['digest']]),
}, ensure_ascii=False, indent=2))
