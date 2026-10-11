"""Prepare fixed confirmation discrimination cohorts; no experiment or source freeze."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts/instrument_whole_visit_20261003/successor_regression'
ANCHOR = ROOT / 'artifacts/shared_condition_research_20261003/successor_regression'
output = BASE / 'successor_regression'
protocol = deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
protocol.update(
    title='Independently confirm an actual response revision against its frozen incumbent',
    baseline_version='0.170.0', candidate_version='0.171.0', quality_anchor_version='0.164.0',
    hypothesis_test='A confirmation planner that retains the actual frozen pre-revision response may acquire more discriminating fresh evidence than confirmation based only on the revised response remaining cut uncertainty.',
    primary_endpoint='All-task and last-seven-stage MSE, solved<=1e-6, false completion, all observation/measurement/compute and CPU, task/chain wins-ties-losses, every new failure/recovery, actual old/new response confirmation forecasts and paid paired outcomes.',
    evaluation='All24 new long learning chains finish before336 new terminal scores. Reuse24 old chains/336 scores from actual0.170 and0.164. Preserve every root/task/seed/limit; no terminal-driven source changes or repeat learning.',
    baseline_runtime_reference='artifacts/instrument_whole_visit_20261003/runtime/src',
    candidate_variant='Build on0.170 complete instrument revisit cost learning. Add confirmation_baseline from the actual pre-research response. A pair plan acquires the already funded one independent fresh batch; ordinary cut research retains its original allocation.',
    development_selection='Known root07 paid p2/p7 revisions are development material only. The long regression includes all twelve previously evaluated roots.',
    prior_protocol=str((PRIOR / 'protocol.json').relative_to(ROOT)),
    candidate_source_status='Prepared only; awaiting stable source, tests and root GO.',
    execution={'maximum_workers':12, 'status':'prepared_not_started'},
    learning_barrier_outputs=[str(output.relative_to(ROOT))],
    source_accounting='Both new arms start empty. prior_confirmation strips the new baseline only in confirmation calls and restores the prior unresolved-cut/good-fit call gate. All complete-visit costs, information/method feedback and draft knowledge remain enabled.',
    development_causal_correction='Existing paid p2/p7 old/new predictions coincide on every old fresh point: zero gain was correctly recorded. The new mechanism changes discriminating confirmation selection, not a fictitious reference-alias bug.',
)
runtime = str((BASE / 'runtime/src').relative_to(ROOT))
protocol['arms'] = {
    'anchor_164': {'runtime':'artifacts/shared_condition_research_20261003/runtime/src', 'knowledge':'empty', 'result_source':'reused0.164 long cohort'},
    'old_cumulative': {'runtime':'artifacts/instrument_whole_visit_20261003/runtime/src', 'knowledge':'empty', 'result_source':'reused0.170 long cohort'},
    'new_cumulative': {'runtime':runtime, 'knowledge':'empty', 'result_source':'new run'},
    'prior_confirmation': {'runtime':runtime, 'knowledge':'empty', 'result_source':'new run',
        'ablation_interface':'For calls carrying confirmation_baseline, drop that kwarg and call the original planner only if selected.condition_uncertainty is unresolved with original_target_loss<=tolerance. Preserve all ordinary planner calls and all0.170 cost learning.'},
}
output.mkdir(parents=True, exist_ok=True)
(output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
for arm, source in (('anchor_164', ANCHOR), ('old_cumulative', PRIOR)):
    for root in protocol['roots']:
        target = output / 'runs' / arm / root
        target.mkdir(parents=True, exist_ok=True)
        for name in ('checkpoint.json.gz','evaluation.json'):
            path = target / name
            if not path.exists():
                path.symlink_to(source / 'runs/new_cumulative' / root / name)
development = {
    'status':'prepared_not_started',
    'fixture':'artifacts/instrument_whole_visit_20261003/paid_confirmation_root07.json.gz',
    'positions':[2,7], 'arms':['prior_confirmation','paired_confirmation'],
    'caps_per_arm':{'observation_cost':24,'measurement_cost':24,'compute_cost':30096},
    'cap_basis':'Actual fresh batch length and saved confirmation_compute_quote; not an invented intermediate historical balance.',
    'reconstruction':'Each arm reconstructs uncertainty from identical paid batches, actual revised program and the saved pre-visit research recipe. Reconstruction/planning/predictions consume the same new cap. Full intermediate condition_state and compute ledger were not saved.',
    'laboratory':'Restore only paid acquisitions before confirmation; saved old fresh outcome is excluded from planning.',
    'scope':'Two known paid-revision planning comparisons, not complete historical session recovery or independent unseen-task testing.',
    'terminal_scores':0,'external_model_calls':0,
}
(BASE / 'development_protocol.json').write_text(json.dumps(development,indent=2)+'\n')
print({'prepared_new_learning':24,'prepared_terminal_scores':336,'reused_chains':24,
       'reused_scores':336,'development_new_runs_executed':0})
