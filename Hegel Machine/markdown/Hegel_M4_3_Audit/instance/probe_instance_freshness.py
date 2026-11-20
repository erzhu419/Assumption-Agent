"""Bounded independent controls for 346153536; source and committed evidence stay read-only."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

# HEGEL_AUDIT_REPO accepts either the checkout root or its "Hegel Machine" directory.
ROOT = Path(os.environ.get('HEGEL_AUDIT_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m43_346153536/Hegel Machine')).expanduser().resolve()
if (ROOT / 'Hegel Machine' / 'src').is_dir():
    ROOT = ROOT / 'Hegel Machine'
OUT = Path(os.environ.get('HEGEL_AUDIT_OUT', '/workspace/scratch/3f14ad2da0ed/audit_m43_346153536')).expanduser().resolve() / 'instance'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition import m4_compositional_discovery as mc
from hegel_machine.cognition import m4_multiscale_renormalization as ms
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab

out = {}

# Real pre-V17.4 archived artifact: public flag semantics, no model/JSON mutation.
legacy = ROOT / 'markdown/Hegel_M4_1_Audit/replay/evidence/persisted_stage1_authored_workspaces/stage1_W1_haldane_shunt/authored_method.json'
legacy_j = json.loads(legacy.read_text())
legacy_calls = []
orig_ev = mc.evaluate_isolated_python_model
def ev_spy(*args, **kwargs):
    legacy_calls.append({'model': args[1], 'n_eval': len(args[2]), 'n_train': len(kwargs['isolated_prefix_rows'])})
    return orig_ev(*args, **kwargs)
mc._VERIFIED_AUTHORED_CACHE.clear()
with patch.object(mc, 'evaluate_isolated_python_model', ev_spy):
    legacy_res = mc.load_verified_authored_method(
        legacy,
        expected_ast_digest=legacy_j['ast_skeleton_digest'],
        expected_instance_digest='definitely-not-the-instance-digest',
        verify_frozen_only=True,
    )
out['legacy_loader_flags'] = {
    'source': str(legacy.relative_to(ROOT)),
    'has_host_bundle': 'host_confirmation_bundle' in legacy_j,
    'accepted_with_mismatched_expected_instance_and_verify_frozen_only': legacy_res is not None,
    'isolated_refit_calls': legacy_calls,
    'verification_mode': (legacy_res or {}).get('_verification_mode'),
    'current_R3_uses_require_external_manifest_true': True,
}

# A real current legacy consumer: scale-blind quench, no forced model failure.
forest, _ = ms.build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
targets, _ = ms.build_m4_cluster_targets(cluster_idx=0)
acqs = []
evals = []
orig_acq = ms.MultiScaleWorkspaceTaskLab.acquire_channel
orig_ms_ev = ms.evaluate_isolated_python_model
def acq_spy(self, pts, *, channel='coarse_macro', phase='pilot'):
    acqs.append({'phase': phase, 'channel': channel, 'points': deepcopy(pts)})
    return orig_acq(self, pts, channel=channel, phase=phase)
def ms_ev_spy(*args, **kwargs):
    evals.append({'model': args[1], 'eval_points': [r['features'] for r in args[2]],
                  'training_points': [r['features'] for r in kwargs['isolated_prefix_rows']]})
    return orig_ms_ev(*args, **kwargs)
with patch.object(ms.MultiScaleWorkspaceTaskLab, 'acquire_channel', acq_spy), patch.object(ms, 'evaluate_isolated_python_model', ms_ev_spy):
    result = ms.evaluate_m4_target_under_arm(
        targets[6], arm_name='HippoPPR_ScaleBlind_Walk', forest=forest,
        index=HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=False),
        stage1_gate=None, work_dir=OUT / 'scale_blind_t07',
    )
out['legacy_scale_blind_t07'] = {'result': result, 'acquisitions': acqs, 'evaluations': evals}

# Known linear model, with two observations already bought before the CodeAgent continuation.
oracle_calls = []
def linear_oracle(pt):
    oracle_calls.append(deepcopy(pt))
    return 2.5 * pt['x']
meter = CostMeter(obs_budget=64, meas_budget=400, compute_budget=10000)
lab = OpenWorkspaceTaskLab(
    task_id='audit_prepaid_linear', surface_domain='audit', feature_names=['x'],
    feature_bounds={'x': (0.1, 8.0)}, cost_per_obs=4.0,
    task_view={'goal_kind': 'open_workspace_mechanism_prediction', 'tolerance': 0.001},
    meter=meter, private_oracle_fn=linear_oracle,
)
paid = [{'x': 1.0}, {'x': 3.0}]
lab.acquire(paid, workspace=None, phase='paid_before_continuation')
before = meter.snapshot()
ep = mc.run_compositional_code_agent_episode(
    lab=lab, heldout_rows=None, tolerance=0.001,
    workspace_dir=OUT / 'prepaid_linear', role_names=['x'], initial_query_batches=[],
    atomic_base_specs=[mc.make_atomic_operator_spec('linear_1d')], confirm_points=[{'x': 5.0}],
)
out['prepaid_linear'] = {
    'before_meter': before, 'after_meter': meter.snapshot(),
    'oracle_calls': oracle_calls, 'training_rows': ep['training_prefix_rows'],
    'confirm_rows': ep['authored_artifact']['host_confirmation_bundle']['confirm_rows'],
    'confirmed_ok': ep['confirmed_ok'], 'agent_summary': ep.get('agent_summary'),
    'expected_oracle_calls': [*paid, {'x': 5.0}],
}

(OUT / 'probe_instance_freshness.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps({
    'legacy_loader_flags': out['legacy_loader_flags'],
    'scale_blind_summary': {'target': targets[6].target_id, 'acquisitions': acqs,
                           'result': result},
    'prepaid_linear': out['prepaid_linear'],
}, ensure_ascii=False, indent=2))
