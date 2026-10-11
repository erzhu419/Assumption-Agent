"""Prepare the fixed cohort only; source freeze and all new runs await root GO."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts/independent_confirmation_20261003/successor_regression'
ANCHOR = ROOT / 'artifacts/shared_condition_research_20261003/successor_regression'
output = BASE / 'successor_regression'
protocol = deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
protocol.update(
    title='Coherent sampling expectation for independent frozen-response confirmation',
    baseline_version='0.171.0', candidate_version='0.172.0', quality_anchor_version='0.164.0',
    hypothesis_test='Predict old/new response contrast under the same reading-residual distribution as the independent confirmation. Only pair scoring changes; ordinary research and existing zero-gain, unresolved and stopping semantics remain enabled. The study-only control delegates only non-None confirmation_baseline calls to the frozen0.171 pair planner; all other calls use current source.',
    primary_endpoint='All-task and last-seven-stage MSE, solved<=1e-6, false completion, all observation/measurement/compute and CPU, task/chain wins-ties-losses, every new failure/recovery, actual old/new confirmation forecasts and paid paired outcomes.',
    evaluation='All 24 new long learning chains finish before 336 new terminal scores. Reuse 24 old chains/336 scores from actual 0.171 and 0.164. Preserve every root/task/seed/limit; no terminal-driven source changes or repeat learning.',
    baseline_runtime_reference='artifacts/independent_confirmation_20261003/runtime/src',
    candidate_variant='Only the confirmation pair reading-residual distribution and response expectation change. Both arms retain actual frozen old/new pair confirmation, ordinary research, method feedback, draft knowledge and complete instrument revisit cost learning.',
    development_selection='Existing root07 paid p2/p7 revisions and frozen0.171 root12 p9 paid revision are development material. The formal regression retains all twelve previously evaluated roots.',
    prior_protocol=str((PRIOR / 'protocol.json').relative_to(ROOT)),
    candidate_source_status='Prepared only; awaiting stable source, tests and root GO.',
    execution={'maximum_workers':12, 'status':'prepared_not_started'},
    learning_barrier_outputs=[str(output.relative_to(ROOT))],
    source_accounting='Both new arms start empty. prior_sampling disables only the new pair-sampling expectation mechanism by calling the frozen0.171 pair planner only when confirmation_baseline is non-None; it must not disable the 0.171 frozen old/new pair. All other knowledge and resource accounting remain enabled.',
    development_causal_correction='0.171 expected contrast can be dominated by deterministic replay of a paid counterexample while its interval exposure predicts a likely fresh miss. A miss alone does not prove model mismatch. The present design must make these forecasts describe the actual acquisition distribution.',
)
runtime = str((BASE / 'runtime/src').relative_to(ROOT))
protocol['arms'] = {
    'anchor_164': {'runtime':'artifacts/shared_condition_research_20261003/runtime/src', 'knowledge':'empty', 'result_source':'reused0.164 long cohort'},
    'old_cumulative': {'runtime':'artifacts/independent_confirmation_20261003/runtime/src', 'knowledge':'empty', 'result_source':'reused0.171 long cohort'},
    'new_cumulative': {'runtime':runtime, 'knowledge':'empty', 'result_source':'new run'},
    'prior_sampling': {'runtime':runtime, 'knowledge':'empty', 'result_source':'new run',
        'ablation_interface':'Study-only importlib load of frozen0.171 observed_condition_experiments; delegate only calls with non-None confirmation_baseline to its planner. All ordinary calls and all remaining code use the candidate runtime; preserve pair acquisition, gain and stopping.'},
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
    'fixture':'artifacts/coherent_confirmation_20261003/paid_confirmation_cases.json.gz',
    'cases':[{'root':'root07','position':2},{'root':'root07','position':7},{'root':'root12','position':9}], 'arms':['prior_sampling','coherent_confirmation'],
    'caps_per_arm':{'observation_cost':24,'measurement_cost':24,'compute_cost':30096},
    'cap_basis':'Actual saved fresh batch length and confirmation_compute_quote; not a recovered historical remainder.',
    'reconstruction':'Both arms rebuild missing uncertainty from identical paid batches, actual revised program and saved pre-visit recipe. Each arm pays reconstruction, planning and predictions inside the same cap. Both retain the 0.171 frozen old/new pair with the same 15000 planning cap.',
    'laboratory':'Restore paid acquisitions only through the preconfirmation prefix; saved fresh outcomes are excluded from planning.',
    'root12_extraction':'One authorized read of the frozen0.171 final checkpoint extracts last visit.previous, candidate, paid training prefix and frozen pair. Intermediate condition state and compute balance are not stored; exact historical stage recovery cannot be claimed.',
    'scope':'Known paid-revision planning comparisons, not unseen-task testing or full historical session restoration.',
    'ablation_interface':'For non-None confirmation_baseline, call the frozen0.171 pair planner; all ordinary calls use current source. Preserve baseline, fixed fresh acquisition, gain and stop. Await root GO before running.',
    'terminal_scores':0, 'external_model_calls':0,
}
(BASE / 'development_protocol.json').write_text(json.dumps(development,indent=2)+'\n')
print({'prepared_new_learning':24,'prepared_terminal_scores':336,
       'reused_chains':24,'reused_scores':336,'executed_new_runs':0})
