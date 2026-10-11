"""Prepare the fixed 0.163 repair against two reused quality references."""
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'artifacts/response_risk_investment_20261002'
COHORTS={'seen_regression':'seen_regression','successor_regression':'successor_regression'}
for cohort,previous_group in COHORTS.items():
    prior=PREVIOUS/previous_group
    p=copy.deepcopy(json.loads((prior/'protocol.json').read_text()))
    p.update(title='Background-specific research experience and joint planning, measurement and confirmation investment',
        baseline_version='0.162.0',candidate_version='0.163.0',quality_anchor_version='0.160.0',
        development_selection='Both cohorts are previously evaluated regressions; all twelve roots and every protocol stage retained.',
        cohort=cohort,scope='previously evaluated short-sequence regression' if cohort=='seen_regression' else 'previously evaluated 14-stage repeated-mechanism regression',
        hypothesis_test='Condition follow-up experience should transfer only across applicable public experiment backgrounds; planning, acquisition and confirmation should jointly preserve useful research while reducing wasted continuation.',
        primary_endpoint='Recovery of seen root04 stage3 quality; all-task and successor-stage independent MSE, solved<=1e-6, false completion, charged observation/measurement/compute, process CPU, and paired chain outcomes versus both 0.162 and the 0.160 quality anchor. The deletion arm tests retained follow-up knowledge contribution.',
        evaluation='All 36 new chains finish learning before all 420 new terminal scores. Both cohorts are previously evaluated regressions. Reused 48 chains and 504 prior terminal values are explicit; no result-based exclusions or tuning.',
        baseline_runtime_reference='artifacts/response_risk_investment_20261002/runtime/src',
        candidate_variant='Scope follow-up experience to applicable experiment backgrounds and account for joint planning, acquisition and confirmation investment.',
        prior_protocol=str((prior/'protocol.json').relative_to(ROOT)),
        learning_barrier_outputs=[str((BASE/name).relative_to(ROOT)) for name in COHORTS],
        candidate_source_status='Awaiting root GO; no new learning started.',
        execution={'maximum_workers':16,'status':'protocol prepared; waiting for candidate freeze and GO'})
    p['arms']={
        'anchor_160':{'runtime':'artifacts/confirmation_investment_20261002/runtime/src','knowledge':'empty','result_source':'reused 0.160 '+previous_group},
        'old_cumulative':{'runtime':'artifacts/response_risk_investment_20261002/runtime/src','knowledge':'empty','result_source':'reused 0.162 '+previous_group},
        'new_cumulative':{'runtime':'artifacts/background_research_investment_20261003/runtime/src','knowledge':'empty','result_source':'new run'}}
    if cohort=='successor_regression':
        p['arms']['no_marginal_history']={'runtime':'artifacts/background_research_investment_20261003/runtime/src','knowledge':'empty','result_source':'new run','remove_action_experience_context':':condition_followup_research:'}
    p['source_accounting']='All new arms begin empty. The deletion arm removes only :condition_followup_research: action_experience contexts before each task/revisit; every other knowledge item and current-task paid state remains. 0.160 and 0.162 results are reused via references.'
    directory=BASE/cohort
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    for arm,source_arm in (('anchor_160','anchor_160'),('old_cumulative','new_cumulative')):
        for root in p['roots']:
            target=directory/'runs'/arm/root
            target.mkdir(parents=True,exist_ok=True)
            for name in ('checkpoint.json.gz','evaluation.json'):
                path=target/name
                if not path.exists():path.symlink_to(prior/'runs'/source_arm/root/name)
print({'new_learning':36,'new_terminal_scores':420,'reused_chains':48,'reused_terminal_scores':504,'compared_chains':84,'compared_terminal_scores':924})
