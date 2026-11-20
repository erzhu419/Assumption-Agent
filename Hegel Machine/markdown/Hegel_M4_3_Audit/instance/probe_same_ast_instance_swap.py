"""Two genuine calibrations, same law/source, different parameters; no fabricated confirmation rows."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

# HEGEL_AUDIT_REPO accepts either the checkout root or its "Hegel Machine" directory.
ROOT = Path(os.environ.get('HEGEL_AUDIT_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m43_346153536/Hegel Machine')).expanduser().resolve()
if (ROOT / 'Hegel Machine' / 'src').is_dir():
    ROOT = ROOT / 'Hegel Machine'
OUT = Path(os.environ.get('HEGEL_AUDIT_OUT', '/workspace/scratch/3f14ad2da0ed/audit_m43_346153536')).expanduser().resolve() / 'instance'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import _run_metered_atomic_calibration, ForestScopeDispatcher
from hegel_machine.cognition.m4_compositional_discovery import load_verified_authored_method, predict_with_frozen_instance, _VERIFIED_AUTHORED_CACHE

sp = importlib.util.spec_from_file_location('m43_study_runner', ROOT / 'scripts/run_v17_5_m4_3_unified_study.py')
runner = importlib.util.module_from_spec(sp)
sp.loader.exec_module(runner)
bundle = ROOT / 'docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120'
forest = runner.load_relocated_forest_from_bundle(bundle / 'relocatable_forest_snapshots/post_stage2_forest_snapshot.json', bundle)
node_id = 'stage0_monod_2d'
action = forest.nodes[node_id]['claim_contract']['conclusion_or_action']
had_expected_instance = bool(action.get('confirmed_instance_digest'))
old_path = Path(action['authored_artifact_path'])
old_art = json.loads(old_path.read_text())
old_loaded = load_verified_authored_method(old_path, expected_ast_digest=action['ast_skeleton_digest'], require_external_manifest=True, verify_frozen_only=True)
assert old_loaded is not None
copy_ws = OUT / 'same_ast_original_location'
shutil.copytree(old_path.parent, copy_ws, dirs_exist_ok=True)
action['authored_artifact_path'] = str(copy_ws / 'authored_method.json')

def predict_current(query_id):
    _VERIFIED_AUTHORED_CACHE.clear()
    return ForestScopeDispatcher(forest).dispatch_predict(
        query_id=query_id, candidate_tree_id='tree_monod_reactor',
        features={'s': 1.6, 'b': 0.0},
        target_metadata={'active_arity': 2, 'haldane_turnover_shunt_regime': False, 'secondary_bleed_active': False},
        ground_truth=6.0 * 1.6 / (2.0 + 1.6),
    )

base = predict_current('baseline_original_instance')
orig_bundle = old_art['host_confirmation_bundle']
cal = _run_metered_atomic_calibration(
    OUT / 'same_ast_distinct_gain_calibration', task_id='genuine_same_law_double_gain', surface_domain='audit_same_law',
    spec=old_art['operator_spec'], role_names=old_art['role_names'],
    pilot_points=[r['features'] for r in orig_bundle['training_prefix_rows']],
    confirm_points=[r['features'] for r in orig_bundle['confirm_rows']],
    oracle_fn=lambda pt: 2.0 * predict_with_frozen_instance(old_loaded, [pt])['predictions'][0],
    tolerance=float(orig_bundle['tolerance']),
)
new_art = cal['artifact']
assert cal['confirmed_ok']
new_loaded = load_verified_authored_method(Path(cal['artifact_path']), require_external_manifest=True, verify_frozen_only=True)
assert new_loaded is not None
assert old_art['ast_skeleton_digest'] == new_art['ast_skeleton_digest']
assert old_art['confirmed_instance_digest'] != new_art['confirmed_instance_digest']

# Replace only the copied artifact bundle, retaining the already loaded forest node/version/AST.
for fname in ('authored_method.json', 'confirmed_manifest.json', 'confirmed_fitted_params.json'):
    shutil.copy2(Path(cal['artifact_path']).parent / fname, copy_ws / fname)
swapped = predict_current('other_valid_same_ast_instance')

# Positive control: the already implemented expected-instance check rejects the same replacement.
action['confirmed_instance_digest'] = old_art['confirmed_instance_digest']
anchored = predict_current('anchor_original_instance_digest')

node_inventory = []
for nd in runner.load_relocated_forest_from_bundle(bundle / 'relocatable_forest_snapshots/post_stage2_forest_snapshot.json', bundle).nodes.values():
    act = nd['claim_contract']['conclusion_or_action']
    node_inventory.append({'node_id': nd['node_id'], 'has_artifact': bool(act.get('authored_artifact_path')),
                           'has_expected_instance_digest': bool(act.get('confirmed_instance_digest')),
                           'archived': bool(nd['utility_ledger'].get('archived_low_utility'))})
result = {
    'node_id': node_id,
    'node_has_expected_instance_digest_before_control': had_expected_instance,
    'same_ast': old_art['ast_skeleton_digest'] == new_art['ast_skeleton_digest'],
    'same_source_sha': old_art['model_sha256'] == new_art['model_sha256'],
    'old_params': old_art['fitted_params'], 'new_params': new_art['fitted_params'],
    'old_instance_digest': old_art['confirmed_instance_digest'], 'new_instance_digest': new_art['confirmed_instance_digest'],
    'new_calibration_confirmed': cal['confirmed_ok'], 'new_calibration_rmse': cal['confirm_rmse'],
    'new_strict_loader_verified': new_loaded is not None,
    'baseline': base, 'swapped_genuine_instance': swapped, 'anchored_positive_control': anchored,
    'node_inventory': node_inventory,
}
(OUT / 'probe_same_ast_instance_swap.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
