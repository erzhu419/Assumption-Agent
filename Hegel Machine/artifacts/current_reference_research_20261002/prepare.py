"""Fix 0.158/0.159 paired cohorts before learning; reuse old regression receipts."""
from copy import deepcopy
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/research_confirmation_withdrawal_20261001'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def reuse(cohort, old_cohort, arm, old_arm, roots):
    for root in roots:
        dest = BASE / cohort / 'runs' / arm / root
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('checkpoint.json.gz', 'evaluation.json'):
            (dest / name).symlink_to(OLD / old_cohort / 'runs' / old_arm / root / name)

source = json.loads((OLD / 'fresh_replication/protocol.json').read_text())
outputs = [str((BASE / name).relative_to(ROOT)) for name in ('seen_regression', 'fresh_replication')]
for cohort in ('seen_regression', 'fresh_replication'):
    p = deepcopy(source)
    p.update(title='Reprice remaining response research against the currently adopted explanation',
        cohort=cohort, candidate_version='0.159.0', baseline_version='0.158.0',
        candidate_source_status='awaiting root GO and source freeze; no candidate learning started',
        learning_barrier_outputs=outputs,
        scope='previous fresh instances reused as seen regression' if cohort == 'seen_regression' else 'new instances of the same unchanged task factories')
    p.pop('withdrawal_ablation', None)
    p['arms'] = {
        'baseline_158': {'runtime':'artifacts/research_confirmation_withdrawal_20261001/runtime/src', 'mode':'natural', 'result_source':'reused previous fresh candidate' if cohort == 'seen_regression' else 'new run'},
        'candidate': {'runtime':'artifacts/current_reference_research_20261002/runtime/src', 'mode':'natural', 'result_source':'new run'}}
    if cohort == 'fresh_replication':
        for group in p['roots'].values():
            for task in group['tasks']:
                task['apparatus_seed'] += 1000
                task['seed'] += 1000
    save(BASE / cohort / 'protocol.json', p)
    if cohort == 'seen_regression':
        reuse(cohort, 'fresh_replication', 'baseline_158', 'candidate', p['roots'])
p = json.loads((OLD / 'instrument_regression/protocol.json').read_text())
p.update(title='Current-reference research and condition-boundary confirmation instrument regression',
    baseline_version='0.158.0', candidate_version='0.159.0',
    candidate_freeze='awaiting source freeze and explicit inclusion; prepared only',
    evaluation='Prepared regression cohort. If activated, all mixed and instrument learning completes before new terminal evaluations.',
    candidate_variant='Current adopted explanation governs remaining research; precise source scope fixed before candidate learning.',
    development_selection='All twelve previously evaluated instrument roots and seven stages retained; not fresh-instance evidence.')
p['arms'] = {'old_cumulative': {'runtime':'artifacts/research_confirmation_withdrawal_20261001/runtime/src','knowledge':'empty','result_source':'reused previous new_cumulative'},
    'new_cumulative': {'runtime':'artifacts/current_reference_research_20261002/runtime/src','knowledge':'empty','result_source':'new run'}}
p['baseline_runtime_reference'] = p['arms']['old_cumulative']['runtime']
p['execution'] = {'maximum_workers':16, 'status':'prepared only; awaiting inclusion decision'}
save(BASE / 'instrument_regression/protocol.json', p)
reuse('instrument_regression','instrument_regression','old_cumulative','new_cumulative',p['roots'])
