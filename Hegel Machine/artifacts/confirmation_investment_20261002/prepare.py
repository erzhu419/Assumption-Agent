"""Fix 0.159/0.160 confirmation-investment cohorts before new learning."""
from copy import deepcopy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/current_reference_research_20261002'
COHORTS=('mixed_regression','instrument_seen','instrument_fresh')
BASELINE='artifacts/current_reference_research_20261002/runtime/src'
CANDIDATE='artifacts/confirmation_investment_20261002/runtime/src'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def reuse(cohort,old_cohort,arm,old_arm,roots):
    for root in roots:
        dest=BASE/cohort/'runs'/arm/root
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('checkpoint.json.gz','evaluation.json'):
            (dest/name).symlink_to(OLD/old_cohort/'runs'/old_arm/root/name)

instrument=json.loads((OLD/'instrument_regression/protocol.json').read_text())
for cohort in COHORTS:
    mixed=cohort=='mixed_regression'
    p=json.loads((OLD/'fresh_replication/protocol.json').read_text()) if mixed else deepcopy(instrument)
    p.update(title='Choose confirmation investment from obtainable discrimination, within fixed total budgets',
        cohort=cohort,baseline_version='0.159.0',candidate_version='0.160.0',
        fixed_before_learning=True,offline=True,external_model_calls=0,
        candidate_source_status='awaiting root GO; no candidate learning started',
        candidate_freeze='Candidate source will be frozen once before candidate learning.',
        learning_barrier_outputs=[str((BASE/name).relative_to(ROOT)) for name in COHORTS],
        evaluation='All three cohorts and arms finish learning before any new terminal evaluation. Reused historical baseline evaluations are identified separately; all original costs, budgets, thresholds and terminal rules retained. No result-based exclusions or tuning.')
    if mixed:
        p['scope']='Previous fresh mixed instances reused as regression; not new generalization evidence.'
        p['arms']={'baseline_159':{'runtime':BASELINE,'mode':'natural','result_source':'reused previous fresh candidate'},
            'candidate':{'runtime':CANDIDATE,'mode':'natural','result_source':'new run'}}
    else:
        p.update(primary_endpoint='Instrument independent response MSE, solved<=1e-6, false completion, all charged observation/measurement/compute and process CPU. Measure actual boundary discrimination from funded confirmations.',
            hypothesis_test='Choose the amount and sequence of confirmation from attainable distinction among current condition hypotheses, under unchanged total research budgets. Independent target response, complete expenditure and unresolved failures decide success.',
            candidate_variant='Confirmation investment responds to reachable condition-boundary information, rather than a fixed 24-row confirmation.',
            baseline_runtime_reference=BASELINE,
            development_selection='All twelve roots and all seven stages retained. Seen and previously unused same-factory instrument instances reported separately.',
            prior_protocol=str((OLD/'instrument_regression/protocol.json').relative_to(ROOT)),
            execution={'maximum_workers':16,'status':'baseline preparation; candidate awaiting GO'})
        p['scope']='previously evaluated instrument regression' if cohort=='instrument_seen' else 'previously unused apparatus seeds of the same unchanged instrument task factory'
        p['arms']={'old_cumulative':{'runtime':BASELINE,'knowledge':'empty','result_source':'reused previous instrument new_cumulative' if cohort=='instrument_seen' else 'new run'},
            'new_cumulative':{'runtime':CANDIDATE,'knowledge':'empty','result_source':'new run'}}
        if cohort=='instrument_fresh':
            for index,group in enumerate(p['roots'].values()):
                apparatus=9934101+index
                for task in group['tasks']:
                    shift=apparatus-task['apparatus_seed']
                    task['apparatus_seed']=apparatus
                    task['seed']+=shift
    save(BASE/cohort/'protocol.json',p)
    if cohort=='mixed_regression':
        reuse(cohort,'fresh_replication','baseline_159','candidate',p['roots'])
    elif cohort=='instrument_seen':
        reuse(cohort,'instrument_regression','old_cumulative','new_cumulative',p['roots'])
