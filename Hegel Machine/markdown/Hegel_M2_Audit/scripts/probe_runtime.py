"""Small counterexamples and controls on fixed 9115 source, without modifying it."""

from _common import HEGEL_REPO, AUDIT_OUTPUT_DIR
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab, make_m2_open_workspace_suite, _evaluate_workspace_submission
import hegel_machine.cognition.open_workspace_wormhole as w

root = AUDIT_OUTPUT_DIR / 'runtime'
root.mkdir(parents=True, exist_ok=True)
results = {}

# Exact deepcopy closure test: two copies retain identical oracle/RNG, and do not
# produce the same first noisy observation even at the same point.
suite = make_m2_open_workspace_suite(5026101)
fac, ev = suite['targets'][0][2]
base = fac()
copy_a, copy_b = deepcopy(base), deepcopy(base)
pt = {k: (lo + hi) / 2 for k, (lo, hi) in base.feature_bounds.items()}
ya = copy_a.acquire([pt])[0].target
yb = copy_b.acquire([pt])[0].target
results['deepcopy_oracle_state'] = {
    'identical_callable': copy_a._oracle is copy_b._oracle,
    'same_point_first_observation_a': ya,
    'same_point_first_observation_b': yb,
    'equal': ya == yb,
}

# New lab with same public API but law outside current finite grammar. Capture
# actual failed confirmation, then show it is still returned as completed.
tol = 0.01
lab = OpenWorkspaceTaskLab(
    task_id='audit_sinusoid', surface_domain='audit_sinusoid',
    feature_names=('u',), feature_bounds={'u': (1.0, 10.0)},
    cost_per_obs=5.0,
    task_view={'goal_kind':'open_workspace_mechanism_prediction','target_metric':'rmse_original_scale','tolerance':tol},
    meter=CostMeter(obs_budget=24, meas_budget=120, compute_budget=800),
    private_oracle_fn=lambda p: 5.0 + math.sin(p['u']),
)
validation_calls = []
old_validate = w.validate_workspace_model_on_rows
def capture_validation(*args, **kwargs):
    value = old_validate(*args, **kwargs)
    validation_calls.append(value)
    return value
w.validate_workspace_model_on_rows = capture_validation
ws = root / 'outside_grammar'
start = time.perf_counter()
sub = w.solve_open_workspace_direct(lab, ws)
real_seconds = time.perf_counter() - start
w.validate_workspace_model_on_rows = old_validate
eval_rows = [Observation(features={'u': 1 + i * 9 / 30}, target=5 + math.sin(1 + i * 9 / 30), history=()) for i in range(31)]
external = _evaluate_workspace_submission(ws, sub.get('model_path'), lab.observations, eval_rows, tol)
results['failed_confirmation_returns_completed'] = {
    'submission': sub, 'confirmation_calls': validation_calls,
    'external': external, 'costs': lab.meter.snapshot(),
    'accounted_python_seconds': lab.python_execution_seconds,
    'observed_entire_solver_wall_seconds': real_seconds,
}

# One formation shared by coordinate permutation and declarative-step controls.
h = w.OpenWorkspaceHegelResearcher()
formation = h.form_and_revise_in_workspaces('audit_5026101', suite, root / 'formation')
results['formation'] = {'source_count':len(formation),'nodes':len(h.forest.nodes),'edges':len(h.forest.horizontal_wormhole_edges),'revision_count':len(h.revision_events)}

rows = []
for variant in ['original','feature_order_reversed','delete_confirm_and_calibrate_steps']:
    agent = deepcopy(h)
    lab = fac()
    if variant == 'feature_order_reversed':
        lab.feature_names = tuple(reversed(lab.feature_names))
        lab.task_view['feature_names'] = list(lab.feature_names)
    if variant == 'delete_confirm_and_calibrate_steps':
        for node in agent.forest.nodes.values():
            node['program_steps'] = [s for s in node.get('program_steps',[]) if s.get('op') == 'synthesize_scoped_python_schema']
    ws = root / variant
    sub = agent.solve_target('audit_target', lab, ws)
    result = ev(ws, sub, lab.observations)
    rows.append({'variant':variant,'sub':sub,'external':result,'costs':lab.meter.snapshot(),'alignment':agent.alignment_checks})
results['feature_order_and_program_steps'] = rows

# Evidence receipt contract accepts old target and node version, absent error,
# and NaN error. This is current utility-level API behavior, not an attack on runs.
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use
cc={'claim_id':'same_law','permission_tier':'local_solver_guidance','evidence_status':'SUPPORTED_IN_SCOPE','preconditions':[]}
tv={'goal_kind':'open_workspace_mechanism_prediction','tolerance':0.01,'target_id':'new_target','node_version':2}
receipts=[
    {'claim_id':'same_law','passed':True,'target_id':'old_target','node_version':1,'observed_error':0.0},
    {'claim_id':'same_law','passed':True},
    {'claim_id':'same_law','passed':True,'observed_error':float('nan')},
]
results['receipt_binding']=[authorize_abstraction_use(requested_tier='local_solver_guidance',task_view=tv,target_features={},claim_contract=cc,evidence_receipt=r) for r in receipts]

output = root.parent / 'runtime_probe_result.json'
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'saved':str(output),
    'deepcopy_oracle_state':results['deepcopy_oracle_state'],
    'outside_grammar':{'status':sub.get('status'), 'confirmation_rmse':[v.get('rmse') for v in validation_calls], 'external':external},
    'feature_order_and_program_steps':[{k:x[k] for k in ('variant','costs','external')} for x in rows],
    'receipt_binding':results['receipt_binding'],
},indent=2))
