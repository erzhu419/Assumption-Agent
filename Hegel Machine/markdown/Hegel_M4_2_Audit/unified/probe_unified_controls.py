"""Bounded, post-hoc diagnostics; never selects from hidden target family or heldout errors."""
from copy import deepcopy
import json
import pickle
import sys
from pathlib import Path
from collections import Counter
from unittest.mock import patch

OUT = Path('/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified')
REPO = Path('/workspace/scratch/3f14ad2da0ed/repo_m42_fb4b23aaf/Hegel Machine')
sys.path.insert(0, str(REPO / 'src'))
from hegel_machine.cognition import m4_2_unified_closure_and_pi_lifecycle as m

with (OUT / 'shared_built.pkl').open('rb') as f:
    built = pickle.load(f)
with (OUT / 'true_stage0_snapshot.pkl').open('rb') as f:
    s0 = pickle.load(f)
targets, oos = m.build_m4_2_target_suite(20_261_120)

def summary(forest):
    return {
        'nodes': sorted(forest.nodes),
        'horizontal_edges': len(forest.horizontal_wormhole_edges),
        'coarse_edges': len(forest.coarse_graining_edges),
        'local_edges': len(forest.local_implication_edges),
    }

frozen4 = m.PhilosophicalForest()
for nid in ('stage0_monod_2d', 'stage0_murray_1d', 'hydro_micro_pore_1d', 'ode_micro_2mode_node'):
    frozen4.nodes[nid] = deepcopy(built['forest_before_stage2_maintenance'].nodes[nid])

frozen_out = {'original_four_node_snapshot': summary(frozen4), 'true_stage0_snapshot': summary(s0['forest']), 'runs': []}
for target in (targets[8], targets[16]):
    for label, base_f in [('original_four', frozen4), ('true_stage0', s0['forest'])]:
        forest = deepcopy(base_f)
        idx = m.HippoRAGWormholeIndex(forest, damping_alpha=.65, wormhole_equivalence_pooling=True, scale_aware=True)
        record = m.evaluate_m4_2_target_under_arm(
            target, arm_name=label, seed=20_261_120,
            work_dir=OUT / 'frozen_probe' / label / target.target_id,
            forest=forest, index=idx, ordinary_history_specs=None,
            fallback_pool_specs=built['ordinary_history_forward'],
        )
        record['cost'].pop('wall_clock_seconds', None)
        frozen_out['runs'].append(record)
(OUT / 'frozen_snapshot_control.json').write_text(json.dumps(frozen_out, indent=2))

# The two operators are exactly the pair for which the report claims low-pilot
# indistinguishability. Both are fitted to the SAME already purchased pilot rows.
# Selection is min pilot RMSE, never family labels or confirmation/heldout outcomes.
pair_tags = ('op_composed_monod_shunt_2d', 'op_composed_murray_shunt_2d')
spec_by_tag = dict(built['ordinary_history_forward'])
pair_rows = []
for target in targets[8:16]:
    pilot = [{'features': dict(p), 'target': target.oracle_fn(p)} for p in target.pilot_points]
    fits = []
    for tag in pair_tags:
        ws = OUT / 'pilot_pair_probe' / target.target_id / tag
        candidate, count = m._autonomous_fit_candidate_specs(
            ws, specs_to_try=[(tag, spec_by_tag[tag])], feature_names=target.feature_names,
            training_rows=pilot, tolerance=0.0,
        )
        fits.append({'tag': tag, 'train_rmse': candidate['train_rmse'], 'evaluated': count,
                     'model_relpath': candidate['model_relpath'], 'workspace': str(ws)})
    selected = min(fits, key=lambda r: r['train_rmse'])
    confirm = [{'features': dict(target.confirm_points[0]), 'target': target.oracle_fn(target.confirm_points[0])}]
    heldout = [{'features': dict(p), 'target': target.oracle_fn(p)} for p in target.heldout_points]
    cf = m.evaluate_isolated_python_model(Path(selected['workspace']), selected['model_relpath'], confirm, isolated_prefix_rows=pilot)
    ho = m.evaluate_isolated_python_model(Path(selected['workspace']), selected['model_relpath'], heldout, isolated_prefix_rows=pilot)
    pair_rows.append({'target_id': target.target_id, 'selection_rule': 'minimum pilot RMSE across both report fork operators',
                     'candidate_fits': fits, 'selected_tag': selected['tag'],
                     'selected_confirm_rmse': cf['rmse'], 'selected_heldout_rmse': ho['rmse'],
                     'tolerance': target.tolerance,
                     'strict_pass': cf['returncode'] == 0 and cf['rmse'] <= target.tolerance and ho['returncode'] == 0 and ho['rmse'] <= target.tolerance,
                     'paid_pilot_count': len(pilot), 'fresh_confirmation_count': 1,
                     'extra_discriminating_measurements': 0})
(OUT / 'posthoc_pilot_pair_control.json').write_text(json.dumps(pair_rows, indent=2))

# Direct execution trace for a stock DEV-frozen Murray-onset row. Record overlaps
# between each evaluator query batch and its training prefix; do not alter behavior.
eval_events = []
acq_events = []
orig_eval = m.evaluate_isolated_python_model
orig_acquire = m.OpenWorkspaceTaskLab.acquire

def track_eval(*args, **kwargs):
    query_rows = args[2]
    train_rows = kwargs.get('isolated_prefix_rows') or []
    result = orig_eval(*args, **kwargs)
    eval_events.append({'query_features': [r['features'] for r in query_rows],
                        'training_count': len(train_rows),
                        'query_count': len(query_rows),
                        'overlap_count': sum(any(q['features'] == t['features'] for t in train_rows) for q in query_rows),
                        'rmse': result['rmse']})
    return result

def track_acquire(self, points, **kwargs):
    result = orig_acquire(self, points, **kwargs)
    acq_events.append({'phase': kwargs.get('phase'), 'points': deepcopy(points)})
    return result

with patch.object(m, 'evaluate_isolated_python_model', track_eval), patch.object(m.OpenWorkspaceTaskLab, 'acquire', track_acquire):
    repeat_record = m.evaluate_m4_2_target_under_arm(
        targets[12], arm_name='Ordinary_AllHistory_DevFrozenOrder', seed=20_261_120,
        work_dir=OUT / 'confirmation_reuse_probe', forest=None, index=None,
        ordinary_history_specs=built['ordinary_history_dev_frozen'],
        fallback_pool_specs=built['ordinary_history_forward'],
    )
repeat_record['cost'].pop('wall_clock_seconds', None)
(OUT / 'confirmation_reuse_control.json').write_text(json.dumps({'record': repeat_record, 'acquisitions': acq_events, 'evaluations': eval_events}, indent=2))

print(json.dumps({'frozen_runs': [{k: r[k] for k in ('arm','target_id','confirmed_via_recall','strict_pass','selected_node_id','cost')} for r in frozen_out['runs']],
                  'pilot_pair_passes': sum(r['strict_pass'] for r in pair_rows),
                  'pilot_pair_total': len(pair_rows),
                  'confirm_reuse_evaluations': eval_events}, indent=2))
