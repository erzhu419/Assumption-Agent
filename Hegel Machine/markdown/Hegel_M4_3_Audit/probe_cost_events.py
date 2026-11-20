"""Read-only instrumentation of existing M4.3 targets; no source changes."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ap = argparse.ArgumentParser()
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
root = args.repo.resolve()
sys.path.insert(0, str(root / 'src'))
from hegel_machine.cognition import m4_3_channel_bound_lifecycle_and_audit_closure as m43
from hegel_machine.cognition import m4_1_unified_discovery_and_renormalization as m41
from hegel_machine.cognition import m4_compositional_discovery as comp
spec = importlib.util.spec_from_file_location('audit_runner', root / 'scripts/run_v17_5_m4_3_unified_study.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
bundle = root / 'docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120'
forest = runner.load_relocated_forest_from_bundle(bundle / 'relocatable_forest_snapshots/post_stage2_forest_snapshot.json', bundle)
targets, _ = m43.build_m4_3_target_suite(seed=20301120)
events = []
orig_auto = m43._autonomous_fit_candidate_specs
orig_isolated = comp.evaluate_isolated_python_model
orig_fine = m43._eval_pi_s_fine_scalar

def traced_auto(*a, **kw):
    result = orig_auto(*a, **kw)
    events.append({'kind':'autonomous_candidates_returned', 'count':result[1],
                   'best_tag':None if result[0] is None else result[0]['tag'],
                   'training_count':len(kw['training_rows'])})
    return result

def traced_eval(*a, **kw):
    caller = sys._getframe(1)
    events.append({'kind':'isolated_evaluation', 'caller':caller.f_code.co_name,
                   'line':caller.f_lineno, 'model':a[1], 'eval_count':len(a[2]),
                   'training_count':len(kw['isolated_prefix_rows'])})
    return orig_isolated(*a, **kw)

def traced_fine(*a, **kw):
    caller = sys._getframe(1)
    events.append({'kind':'fine_RK4_call', 'caller':caller.f_code.co_name,
                   'line':caller.f_lineno, 'u':a[0], 'T':a[1]})
    return orig_fine(*a, **kw)

m43._autonomous_fit_candidate_specs = traced_auto
m43.evaluate_isolated_python_model = traced_eval
m41.evaluate_isolated_python_model = traced_eval
comp.evaluate_isolated_python_model = traced_eval
m43._eval_pi_s_fine_scalar = traced_fine
records = []
with TemporaryDirectory(prefix='hegel-m43-cost-audit-') as td:
    for ix in (0, 20, 22, 23):
        events.clear()
        t = targets[ix]
        result = m43.evaluate_m4_3_target_under_arm(t, arm_name='Hegel_Unified_Continual_Forest',
            seed=20301120, work_dir=Path(td)/t.target_id,
            forest=forest, index=m43.HippoRAGWormholeIndex(forest, damping_alpha=.65,
                wormhole_equivalence_pooling=True, scale_aware=True),
            ordinary_history_specs=None, fallback_pool_specs=[])
        records.append({'target_id':t.target_id, 'receipt':result, 'events':list(events),
                        'event_counts':dict(Counter(x['kind'] for x in events))})
args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
for r in records:
    print(r['target_id'], r['receipt']['strict_pass'], r['receipt']['evaluated_candidate_count'], r['event_counts'])
