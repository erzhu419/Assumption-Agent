"""Prepare one fixed long-cohort cost-learning contrast; no learning or terminal calls."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts/downstream_research_value_20261003/successor_regression'
ANCHOR = ROOT / 'artifacts/shared_condition_research_20261003/successor_regression'
output = BASE / 'successor_regression'
protocol = deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
protocol.update(
    title='Learn the complete realized instrument revisit cost while retaining phase gains and information misses',
    baseline_version='0.168.0', candidate_version='0.170.0', quality_anchor_version='0.164.0',
    cohort='successor_regression', scope='previously evaluated 12-root 14-stage regression',
    hypothesis_test='A complete paid revisit cost sample should improve later investment decisions compared with recording only its first scored phase; phase gain, trials, information misses and all other knowledge remain unchanged.',
    primary_endpoint='All-task and last-seven-stage MSE, solved<=1e-6, false completion, complete observation/measurement/compute and CPU, task/chain wins-ties-losses, every new failure and recovery, actual natural visit order and complete-versus-first-phase costs.',
    evaluation='All 24 new long learning chains finish before 336 new terminal task-arm scores. Reuse 24 old chains and 336 terminal scores from 0.168 and 0.164. Keep all roots, seeds, tasks and resources; no terminal-based source changes or repeat training.',
    baseline_runtime_reference='artifacts/downstream_research_value_20261003/runtime/src',
    candidate_variant='Current 0.169 plus complete instrument revisit cost learning only. The 0.169 draft-counterexample consumer remains active in both new arms; 0.168 is the actual reused baseline, not a relabeled 0.169 run.',
    development_selection='The root07 visit17 prefix is a known paid development case. All twelve old regression roots remain in the separate frozen long cohort.',
    prior_protocol=str((PRIOR / 'protocol.json').relative_to(ROOT)),
    learning_barrier_outputs=[str(output.relative_to(ROOT))],
    candidate_source_status='Prepared only; waiting for stable production tests and root GO.',
    execution={'maximum_workers':12, 'status':'prepared_not_started'},
    source_accounting='Both new arms begin empty and retain identical phase gain, trials, condition information and draft knowledge. first_phase_cost changes only the cost sample passed into record_instrument_revisit_cost. Both historical references are reused.',
    development_causal_correction='Prior no-cycle-discount development identified a scheduling effect, not an endpoint gain. This experiment changes full-visit cost learning, preserving the original cost estimator and all information evidence.',
)
protocol.pop('preterminal_abort', None)
runtime = str((BASE / 'runtime/src').relative_to(ROOT))
protocol['arms'] = {
    'anchor_164': {'runtime':'artifacts/shared_condition_research_20261003/runtime/src', 'knowledge':'empty', 'result_source':'reused 0.164 long cohort'},
    'old_cumulative': {'runtime':'artifacts/downstream_research_value_20261003/runtime/src', 'knowledge':'empty', 'result_source':'reused 0.168 long cohort'},
    'new_cumulative': {'runtime':runtime, 'knowledge':'empty', 'result_source':'new run'},
    'first_phase_cost': {'runtime':runtime, 'knowledge':'empty', 'result_source':'new run',
        'ablation_interface':'Capture each policy first scored record_condition_followup costs; pass those costs to the original record_instrument_revisit_cost helper. Keep sampling_context and screening_compute_cost unchanged. No deleted experience or disabled cycle discount.'},
}
protocol['retained_state'] = 'One direct checkpoint per chain, overwritten after each step; complete paid trajectories remain inside. Small step receipts and research evidence are retained.'
output.mkdir(parents=True, exist_ok=True)
(output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
for arm, source in (('anchor_164', ANCHOR), ('old_cumulative', PRIOR)):
    for root in protocol['roots']:
        target = output / 'runs' / arm / root
        target.mkdir(parents=True, exist_ok=True)
        for name in ('checkpoint.json.gz', 'evaluation.json'):
            path = target / name
            if not path.exists():
                path.symlink_to(source / 'runs/new_cumulative' / root / name)
development = {
    'status':'prepared_not_started', 'candidate_runtime':runtime,
    'source':'artifacts/downstream_research_value_20261003/successor_regression/runs/new_cumulative/root07/visit17.json.gz',
    'shared_prefix':{'initial_tasks':14, 'actual_revisit_positions':[5,5,5]},
    'remaining_resources':{'observation_cost':1732, 'measurement_cost':1852, 'compute_cost':2117462},
    'new_continuations':1, 'terminal_scores':0,
    'comparator':'Reuse the existing natural 0.168 root07 suffix from the exact shared paid prefix.',
    'stopping':'Normal UnifiedResearchSession queue and stopping; no manually inserted/excluded purchase or budget change.',
    'separation':'Development A ends without terminal evaluation. Formal B requires a separate root GO after production is stable.',
    'external_model_calls':0,
}
(BASE / 'development_protocol.json').write_text(json.dumps(development, indent=2) + '\n')
print({'prepared_new_learning':24, 'prepared_terminal_scores':336,
       'reused_chains':24, 'reused_scores':336, 'executed_learning':0})
