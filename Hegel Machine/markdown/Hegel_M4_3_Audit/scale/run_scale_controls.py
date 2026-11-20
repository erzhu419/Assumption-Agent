"""Bounded read-only M4.3 controls using committed relocatable artifacts; no forest rebuild."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

# HEGEL_AUDIT_REPO accepts either the checkout root or its "Hegel Machine" directory.
ROOT = Path(os.environ.get('HEGEL_AUDIT_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m43_346153536/Hegel Machine')).expanduser().resolve()
if (ROOT / 'Hegel Machine' / 'src').is_dir():
    ROOT = ROOT / 'Hegel Machine'
OUT = Path(os.environ.get('HEGEL_AUDIT_OUT', '/workspace/scratch/3f14ad2da0ed/audit_m43_346153536')).expanduser().resolve() / 'scale'
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition import m4_2_unified_closure_and_pi_lifecycle as m42
from hegel_machine.cognition import m4_3_channel_bound_lifecycle_and_audit_closure as m43
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m4_multiscale_renormalization import MultiScaleWorkspaceTaskLab

spec = importlib.util.spec_from_file_location('runner_m43', ROOT / 'scripts/run_v17_5_m4_3_unified_study.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
BUNDLE = ROOT / 'docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120'
SNAPS = BUNDLE / 'relocatable_forest_snapshots'

def load_forest(name='post_stage2_forest_snapshot.json'):
    return runner.load_relocated_forest_from_bundle(SNAPS / name, BUNDLE)

def frozen_dump(f):
    return json.dumps(runner.serialize_relocatable_forest(f, BUNDLE), sort_keys=True)

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    evidence = {'commit': '3461535364fcdb4e9ae9c87ccba502b7e6b56fe8',
                'source': 'committed relocatable bundle, no rebuild, no source mutation'}
    f = load_forest()
    before = frozen_dump(f)
    fit_calls = []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'fit':
            fit_calls.append({'filename': frame.f_code.co_filename, 'line': frame.f_code.co_firstlineno})
    sys.setprofile(profile)
    try:
        stream = m42.run_interleaved_zero_fit_dispatcher_stream(f)
    finally:
        sys.setprofile(None)
    no_edges = load_forest()
    no_edges.coarse_graining_edges = []
    no_edge_stream = m42.run_interleaved_zero_fit_dispatcher_stream(no_edges)
    evidence['r3'] = {
        'read_only_full_forest': before == frozen_dump(f),
        'observed_python_fit_calls': fit_calls,
        'stream': stream,
        'deleted_edges_Q05': next(r for r in no_edge_stream['receipts'] if r['query_id'].startswith('R3_Q05')),
    }

    # Real Stage0 certificate and real Stage2 action have distinct fitted instances.
    s0 = load_forest('true_stage0_forest_snapshot.json')
    cert0 = deepcopy(next(e['empirical_closure_certificate'] for e in s0.coarse_graining_edges
                          if e['coarse_node_id'] == 'ode_meso_schur_pi_thermal'))
    swapped = load_forest()
    cnode = 'ode_meso_schur_pi_thermal'
    old_action_instance = swapped.nodes[cnode]['claim_contract']['conclusion_or_action']['confirmed_instance_digest']
    swap_result = swapped.revalidate_vertical_edges_with_certificate(
        cnode, witness_episode_id='audit_authentic_stage0_certificate_on_stage2_instance',
        closure_certificate=cert0, reason='bounded identity mismatch control')
    edge = next(e for e in swapped.coarse_graining_edges if e['coarse_node_id'] == cnode)
    evidence['authentic_certificate_cross_instance'] = {
        'stage0_certificate_instance': cert0.get('confirmed_instance_digest'),
        'stage2_action_instance': old_action_instance,
        'return_value': swap_result,
        'after_edge_instance': edge.get('confirmed_instance_digest'),
        'after_edge_certificate_instance': edge['empirical_closure_certificate'].get('confirmed_instance_digest'),
        'after_edge_box': edge['scale_map']['window_or_partition']['verified_input_box'],
        'after_cert_box': edge['empirical_closure_certificate']['verified_input_box'],
    }

    targets, _ = m43.build_m4_3_target_suite(seed=20301120)
    pi_target = targets[20]
    acquisitions = []
    fitted_candidates = []
    original_acquire = MultiScaleWorkspaceTaskLab.acquire_channel
    original_fit_candidates = m43._autonomous_fit_candidate_specs
    def spy_acquire(self, points, *, channel='coarse_macro', phase='pilot'):
        result = original_acquire(self, points, channel=channel, phase=phase)
        acquisitions.append({'task_id': self.task_id, 'phase': phase, 'channel': channel,
                             'points': deepcopy(points), 'rows': deepcopy(result)})
        return result
    def spy_fit_candidates(*args, **kwargs):
        best, count = original_fit_candidates(*args, **kwargs)
        fitted_candidates.append({'candidate_count': count,
                                  'best_tag': best.get('tag') if best else None,
                                  'best_role_order': best.get('role_order') if best else None,
                                  'best_train_rmse': best.get('train_rmse') if best else None,
                                  'model_relpath': best.get('model_relpath') if best else None})
        return best, count
    def evaluate_control(t, tag):
        cf = load_forest()
        idx = HippoRAGWormholeIndex(cf, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        start = len(acquisitions)
        fit_start = len(fitted_candidates)
        with patch.object(MultiScaleWorkspaceTaskLab, 'acquire_channel', spy_acquire), \
             patch.object(m43, '_autonomous_fit_candidate_specs', spy_fit_candidates):
            res = m43.evaluate_m4_3_target_under_arm(
                t, arm_name=tag, seed=20301120, work_dir=OUT / tag,
                forest=cf, index=idx, ordinary_history_specs=None, fallback_pool_specs=[],
                open_code_agent_fallback=False, allow_pool_fallback=False)
        contract = json.loads((OUT / tag / 'model_scope_contract.json').read_text())
        return {'result': res, 'model_scope_contract': contract, 'acquisitions': acquisitions[start:],
                'actual_fitted_candidates': fitted_candidates[fit_start:]}

    evidence['native_adiabatic_T21_control'] = evaluate_control(pi_target, 'native_adiabatic_T21')

    # Canonical coordinate names remain canonical; only the requested point distribution changes.
    swap_pilot = [
        {'u_drive': 2.45, 'T_horizon': .40}, {'u_drive': 2.90, 'T_horizon': .52},
        {'u_drive': 2.55, 'T_horizon': .67}, {'u_drive': 2.80, 'T_horizon': .85},
        {'u_drive': 2.65, 'T_horizon': 1.05}, {'u_drive': 2.95, 'T_horizon': 1.20},
    ]
    swap_confirm = [{'u_drive': 2.70, 'T_horizon': .47}]
    swap_heldout = [{'u_drive': 2.50, 'T_horizon': .42}, {'u_drive': 2.85, 'T_horizon': .55},
                    {'u_drive': 2.75, 'T_horizon': .75}]
    swap_target = replace(pi_target, target_id='audit_canonical_axis_swap',
                          feature_bounds={'u_drive': (2.4, 3.0), 'T_horizon': (.4, 1.3)},
                          pilot_points=swap_pilot, confirm_points=swap_confirm, heldout_points=swap_heldout)
    evidence['canonical_axis_role_confusion'] = {
        'gate_only': m42._find_valid_canonical_role_map_for_box(
            swap_target.feature_names, f.nodes[cnode], [*swap_pilot, *swap_confirm]),
        **evaluate_control(swap_target, 'canonical_axis_swap'),
    }

    # No program mutation: fresh_points remains 1. First declared confirm is already paid;
    # the second is fresh, physically valid, but outside the certificate.
    exhausted_first = replace(pi_target, target_id='audit_confirm_fresh_outside_box',
                             feature_bounds={'u_drive': (.4, 3.0), 'T_horizon': (.05, 5.2)},
                             confirm_points=[deepcopy(pi_target.pilot_points[0]),
                                             {'u_drive': 2.8, 'T_horizon': .4}])
    evidence['fresh_confirmation_outside_certificate'] = evaluate_control(exhausted_first, 'fresh_confirm_outside_box')

    # Native micro path from the committed benchmark, preserved matrix discrepancy.
    micro = evaluate_control(targets[22], 'native_quench_exact_claim')
    src = (OUT / 'native_quench_exact_claim/recalled_fine_2mode.py').read_text()
    ns = {}
    exec(compile(src, '<audited_existing_fine_surrogate>', 'exec'), ns)
    raw_micro_rows = [{'features': p, 'target': targets[22].oracle_fn(p)} for p in targets[22].pilot_points]
    ns['fit'](raw_micro_rows)
    micro['native_heldout_errors'] = [
        {'point': p, 'prediction': ns['predict'](p), 'truth': targets[22].oracle_fn(p),
         'abs_error': abs(ns['predict'](p) - targets[22].oracle_fn(p))}
        for p in targets[22].heldout_points]
    micro['matrix_entries'] = {'RK4_A12': m42.DEFAULT_PI_S_ODE_PARAMS['g_sf'],
                                'surrogate_A12': m42.DEFAULT_PI_S_ODE_PARAMS['g_sf'] / m42.DEFAULT_PI_S_ODE_PARAMS['tau_s']}
    evidence['fine_exact_formula_claim'] = micro
    (OUT / 'scale_controls.json').write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'r3_read_only': evidence['r3']['read_only_full_forest'], 'r3_fit_calls': len(fit_calls),
        'r3_all_routes_exact': stream['all_routes_exact'],
        'certificate_swap': evidence['authentic_certificate_cross_instance'],
        'native_T21': evidence['native_adiabatic_T21_control']['result'],
        'axis_swap': evidence['canonical_axis_role_confusion']['result'],
        'axis_swap_fitted_candidates': evidence['canonical_axis_role_confusion']['actual_fitted_candidates'],
        'fresh_outside_box': evidence['fresh_confirmation_outside_certificate']['result'],
        'fine_exact_claim': micro['result'], 'fine_native_errors': micro['native_heldout_errors']
    }, indent=2))

if __name__ == '__main__':
    main()
