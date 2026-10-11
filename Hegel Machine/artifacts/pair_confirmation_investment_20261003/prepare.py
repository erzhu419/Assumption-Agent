"""Prepare fixed pair-investment cohorts; do not freeze source or run research."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts/coherent_confirmation_20261003/successor_regression'
ANCHOR = ROOT / 'artifacts/shared_condition_research_20261003/successor_regression'
output = BASE / 'successor_regression'
protocol = deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
protocol.update(
    title='Value and cost of purchasing a discriminating frozen-response confirmation',
    baseline_version='0.172.0', candidate_version='0.173.0', quality_anchor_version='0.164.0',
    hypothesis_test='Use the coherent old/new forecast and paid experience to decide whether the next pair-confirmation batch is worth purchasing. The control replaces only condition_confirmation._acquire_pair_confirmation with the0.172 single acquire; retain all other production code.',
    primary_endpoint='All-task and last-seven-stage MSE, solved<=1e-6, false completion, all observation/measurement/compute and CPU, task/chain wins-ties-losses, every new failure/recovery, actual pair purchase/decline decisions, predicted contrast and paid fresh discrimination.',
    evaluation='All24 new long learning chains finish before336 new terminal scores. Reuse24 old chains/336 scores from actual0.172 and0.164. Preserve every root/task/seed/limit; no terminal-driven source changes or repeat learning.',
    baseline_runtime_reference='artifacts/coherent_confirmation_20261003/runtime/src',
    candidate_variant='Pair-only procurement helper. Both arms retain the0.172 coherent pair planner, all ordinary research, method/draft knowledge, shared resource pool and complete instrument revisit costs.',
    development_selection='Reuse the existing root07 p2/p7 and root12 p9 paid prefixes. Do not read another large checkpoint. The formal cohort includes all twelve previously evaluated roots.',
    prior_protocol=str((PRIOR / 'protocol.json').relative_to(ROOT)),
    candidate_source_status='Prepared only; awaiting stable source, tests and root GO.',
    candidate_freeze='Only after final production tests and explicit root GO.',
    execution={'maximum_workers':12,'status':'prepared_not_started'},
    learning_barrier_outputs=[str(output.relative_to(ROOT))],
    source_accounting='Both new arms start empty. Study-only prior_investment restores only0.172 pair single-batch procurement. It must not replace the coherent planner, drop confirmation_baseline, delete experience or change ordinary research.',
    development_causal_correction='0.172 changed forecast consistency and actual local discrimination but all168 terminal task qualities tied0.171 with extra observation and compute costs. Test whether pair purchase value can reduce wasted investment without surrendering useful confirmation.',
)
runtime = str((BASE / 'runtime/src').relative_to(ROOT))
protocol['arms'] = {
    'anchor_164': {'runtime':'artifacts/shared_condition_research_20261003/runtime/src','knowledge':'empty','result_source':'reused0.164 long cohort'},
    'old_cumulative': {'runtime':'artifacts/coherent_confirmation_20261003/runtime/src','knowledge':'empty','result_source':'reused0.172 long cohort'},
    'new_cumulative': {'runtime':runtime,'knowledge':'empty','result_source':'new run'},
    'prior_investment': {'runtime':runtime,'knowledge':'empty','result_source':'new run',
        'ablation_interface':'Replace condition_confirmation._acquire_pair_confirmation with acquire(channels, confirm_continuation_program, probe=plan[probe]); all other production code and feedback stay enabled.'},
}
output.mkdir(parents=True,exist_ok=True)
(output / 'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for arm, source in (('anchor_164',ANCHOR),('old_cumulative',PRIOR)):
    for root in protocol['roots']:
        target = output / 'runs' / arm / root
        target.mkdir(parents=True,exist_ok=True)
        for name in ('checkpoint.json.gz','evaluation.json'):
            path = target / name
            if not path.exists():
                path.symlink_to(source / 'runs/new_cumulative' / root / name)
development = {
    'status':'prepared_not_started',
    'fixture':'artifacts/coherent_confirmation_20261003/paid_confirmation_cases.json.gz',
    'cases':[{'root':'root07','position':2},{'root':'root07','position':7},{'root':'root12','position':9}],
    'arms':['prior_investment','pair_investment'],
    'caps_per_arm':{'observation_cost':96,'measurement_cost':96,'compute_cost':90000},
    'cap_basis':'Root fixed a new equal96/96/90000 allowance before any development result to permit additional batches. Original24/24/30096 quotes remain metadata, not an exact recovered historical balance. Formal cohort budgets stay unchanged.',
    'reconstruction':'Reuse the previous paid-state rebuilding procedure, charge its cost equally and separately. Both arms use the same coherent pair planner and its original15000 planning cap. Precise intermediate historical state and compute balance were not saved.',
    'laboratory':'Restore only preconfirmation paid acquisitions. Saved fresh outcomes must not enter planning, valuation or the purchase decision.',
    'runner_basis':'artifacts/coherent_confirmation_20261003/run_development.py; integrate the new decision between plan and acquisition after the interface is reviewed, not the previous planner ablation.',
    'endpoints':'Purchase/decline, paid information used, coherent forecast, actual rows and prediction difference when purchased, remaining budget and complete compute/CPU. A skipped confirmation is not a demonstrated repair.',
    'scope':'Three known paid-revision decisions; no terminal evaluation, full historical session recovery or unseen-task claim.',
    'terminal_scores':0,'external_model_calls':0,
}
(BASE / 'development_protocol.json').write_text(json.dumps(development,indent=2)+'\n')
print({'prepared_new_learning':24,'prepared_terminal_scores':336,
       'reused_chains':24,'reused_scores':336,'executed_new_runs':0})
