"""Read-only, zero-API M4.1 counterfactuals; all new work is local to this audit."""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import textwrap

REPO = Path(os.environ.get('HEGEL_AUDIT_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m41_99b1fdfeb/Hegel Machine'))
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'src'))
import hegel_machine.cognition.m4_1_unified_discovery_and_renormalization as u
from hegel_machine.cognition.m4_compositional_discovery import apply_structural_combinator, load_verified_authored_method, make_atomic_operator_spec
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab

EVID = REPO / 'docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120'
ART = EVID / 'persisted_stage1_authored_workspaces'


def load_history():
    forest, atoms = u.build_stage0_atomic_forest()
    arts = [load_verified_authored_method(ART / ws / 'authored_method.json') for ws in (
        'stage1_W1_haldane_shunt', 'stage1_W2_osc_shunt_4p', 'stage1_W3_murray_capped')]
    assert all(arts)
    hist = [*atoms,
        ('op_inter_ring_3p', apply_structural_combinator(make_atomic_operator_spec('exp_relax_2d'), 'oscillatory_damping_modulation')),
        ('op_inter_monod_shunt', apply_structural_combinator(make_atomic_operator_spec('michaelis_1d', arity=2), 'additive_secondary_shunt', secondary_role=1)),
        *[(tag, deepcopy(a['operator_spec'])) for tag,a in zip((
            'op_composed_haldane_shunt_2d', 'op_composed_osc_shunt_4p_2d', 'op_composed_murray_capped_1d'), arts)],
    ]
    return forest, hist


def static_order_probe():
    _, hist = load_history()
    # One fixed list for all targets. Keep all 11 original entries; do not use target labels/cues.
    preferred = ['op_composed_haldane_shunt_2d', 'op_composed_osc_shunt_4p_2d',
                 'op_atomic_murray_1d', 'op_composed_murray_capped_1d']
    rank = {name: i for i, name in enumerate(preferred)}
    ordered = sorted(hist, key=lambda item: rank.get(item[0], len(rank)))
    targets, _ = u.build_unified_12_target_suite()
    rows = []
    for i in range(12):
        r = u.evaluate_unified_target_under_arm(targets[i], arm_name='Audit_FixedStaticOrder',
            forest=None, index=None, ordinary_history_specs=ordered, fallback_universe_specs=hist,
            ws_dir=OUT / 'static_order' / targets[i].target_id)
        rows.append(r)
        print('static', r['target_id'], r['evaluated_candidate_count'], r['cost']['composite_cost'], flush=True)
    official = [json.loads(line) for line in (EVID / 'per_target_receipts_336_runs.jsonl').read_text().splitlines()]
    chosen = {r['target_id'] for r in rows}
    baseline = [r for r in official if r['arm'] == 'Hegel_M4_1_UnifiedForest' and 'seed' not in r and r['target_id'] in chosen]
    return {'order': [t for t, _ in ordered], 'records': rows, 'matched_committed_hegel_records': baseline,
        'mean_composite_cost': mean(r['cost']['composite_cost'] for r in rows),
        'hegel_mean_composite_cost': mean(r['cost']['composite_cost'] for r in baseline),
        'mean_evaluated_candidates': mean(r['evaluated_candidate_count'] for r in rows),
        'all_strict_pass': all(r['strict_pass'] for r in rows)}


def prefix_loss_probe():
    forest, hist = load_history()
    # Reconstruct the actual Stage1+Stage2 active forest from the repository's own
    # registration/compression/revision code and read-only confirmed artifacts.
    # Do not rerun discovery, DEV acquisitions, or the full study.
    chain_src = inspect.getsource(u.run_m4_1_unified_continual_discovery_chain)
    register_block = chain_src[chain_src.index('    # --- Register Verified Authored'):chain_src.index('    # Register 2 misleading surface-cue lures')]
    revision_block = chain_src[chain_src.index('    # Protocol revision + Horizontal'):chain_src.index('    ledger["stage2_compression_and_empirical_dev_lures"]')]
    ctx = dict(vars(u))
    ctx.update(work_root=ART, forest_unified=forest)
    exec(textwrap.dedent(register_block), ctx)
    exec(textwrap.dedent(revision_block), ctx)
    idx = HippoRAGWormholeIndex(forest, damping_alpha=0.0, wormhole_equivalence_pooling=True, scale_aware=True)
    target = u.build_unified_12_target_suite()[0][8]
    events = []
    orig_acq, orig_fit = OpenWorkspaceTaskLab.acquire, u._autonomous_fit_candidate_specs
    def acq(self, points, *args, **kwargs):
        events.append({'event': 'acquire', 'phase': kwargs.get('phase'), 'points': deepcopy(points)})
        return orig_acq(self, points, *args, **kwargs)
    def fit(*args, **kwargs):
        events.append({'event': 'fit', 'tags': [x[0] for x in kwargs['specs_to_try']],
                       'training_points': [r['features'] for r in kwargs['training_rows']]})
        return orig_fit(*args, **kwargs)
    OpenWorkspaceTaskLab.acquire, u._autonomous_fit_candidate_specs = acq, fit
    try:
        r = u.evaluate_unified_target_under_arm(target, arm_name='Audit_Alpha0SameHistory_PrefixTrace',
            forest=forest, index=idx, ordinary_history_specs=None, fallback_universe_specs=hist,
            ws_dir=OUT / 'prefix_loss' / 'original')
        original_events = deepcopy(events)
        events.clear()
        # Same existing runner, two local counterfactual changes only:
        # retain already-paid failed confirm; do not repurchase that point as a fresh confirm.
        src = inspect.getsource(u.evaluate_unified_target_under_arm)
        src = src.replace('    # Open Compositional Fallback when prior recall',
            "    if not confirmed_ok and index is not None and 'cf0_rows' in locals():\n"
            "        frozen_prefix = [*frozen_prefix, *cf0_rows]\n\n"
            '    # Open Compositional Fallback when prior recall', 1)
        src = src.replace('lab.acquire(target.confirm_points, workspace=None, phase="fallback_confirm")',
            'lab.acquire([p for p in target.confirm_points if not any(p == r["features"] for r in frozen_prefix)], workspace=None, phase="fallback_confirm")', 1)
        namespace = dict(vars(u))
        exec(src, namespace)
        fixed = namespace['evaluate_unified_target_under_arm'](target, arm_name='Audit_KeepPaidPrefix_FreshConfirm',
            forest=forest, index=idx, ordinary_history_specs=None, fallback_universe_specs=hist,
            ws_dir=OUT / 'prefix_loss' / 'retain_paid_prefix')
        return {'original_record': r, 'original_events': original_events, 'index_edge_counts': idx.edge_counts,
                'retained_prefix_record': fixed, 'retained_prefix_events': deepcopy(events),
                'scope': 'One Alpha0 SameHistory target, active forest reconstructed with exact repository registration/compression/revision statements and committed artifacts; no production code edits; two local runner substitutions shown in probe source.'}
    finally:
        OpenWorkspaceTaskLab.acquire, u._autonomous_fit_candidate_specs = orig_acq, orig_fit


def r3_frozen_parameter_probe():
    results = {}
    for ws_name, points, oracle in (
        ('stage1_S1_monod_parent', [{'s': .5, 'b': 0.}, {'s': 1.1, 'b': 0.}, {'s': 1.9, 'b': 0.}], lambda f: 6*f['s']/(2+f['s'])),
        ('stage1_W1_haldane_shunt', [{'s': 6., 'b': .8}, {'s': 10.5, 'b': .4}], lambda f: 8.8*f['s']/(2.1+f['s']+.14*f['s']**2)+.76*f['b']),
    ):
        src = (ART / ws_name / 'authored_model.py').read_text()
        ns = {}
        exec(compile(src, str(ART / ws_name / 'authored_model.py'), 'exec'), ns)
        before = deepcopy(ns['_PARAMS'])
        unfitted = [ns['predict']({'features': p}) for p in points]
        prefix = json.loads((ART / ws_name / 'confirmed_training_observations.json').read_text())
        ns['fit'](prefix)
        after = deepcopy(ns['_PARAMS'])
        fitted = [ns['predict']({'features': p}) for p in points]
        ys = [oracle(p) for p in points]
        rmse = lambda pred: math.sqrt(mean((a-b)**2 for a,b in zip(pred,ys)))
        results[ws_name] = {'module_initial_params': before, 'after_fit_same_old_prefix_params': after,
            'predict_without_fit_rmse': rmse(unfitted), 'predict_after_old_prefix_fit_rmse': rmse(fitted),
            'training_prefix_count': len(prefix)}
    return results


if __name__ == '__main__':
    result = {'commit': '99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c',
        'static_order': static_order_probe(), 'paid_prefix_loss': prefix_loss_probe(),
        'r3_parameters': r3_frozen_parameter_probe()}
    (OUT / 'probe_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'static_order' and k != 'paid_prefix_loss'}, ensure_ascii=False))
    print('prefix original', result['paid_prefix_loss']['original_record']['cost'])
    print('prefix retained', result['paid_prefix_loss']['retained_prefix_record']['cost'])
