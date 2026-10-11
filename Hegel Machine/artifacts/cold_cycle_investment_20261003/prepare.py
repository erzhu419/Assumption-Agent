"""Prepare the fixed 0.166 repair against two reused quality references."""
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'artifacts/joint_mismatch_investment_20261003'
ANCHOR=ROOT/'artifacts/shared_condition_research_20261003'
COHORTS={'seen_regression':'seen_regression','successor_regression':'successor_regression'}
for cohort,previous_group in COHORTS.items():
    prior=PREVIOUS/previous_group
    p=copy.deepcopy(json.loads((prior/'protocol.json').read_text()))
    p.update(title='Predict initial complete experiment cost from executable work and current paid evidence',
        baseline_version='0.165.0',candidate_version='0.166.0',quality_anchor_version='0.164.0',
        development_selection='Both cohorts are previously evaluated regressions; all twelve roots and every protocol stage retained.',
        cohort=cohort,scope='previously evaluated short-sequence regression' if cohort=='seen_regression' else 'previously evaluated 14-stage repeated-mechanism regression',
        hypothesis_test='Initial cycle cost forecasts should reflect expected executable work while retaining complete hard reserves and independent confirmation, avoiding both missed worthwhile research and wasteful repeated purchases.',
        primary_endpoint='Recovery of long root04 position1 and avoiding excessive purchases on short root01 position1; retain every 0.165 newly failing case and every existing stage. All-task and successor-stage independent MSE, solved<=1e-6, false completion, complete observation/measurement/compute, CPU, paired task/chain outcomes, and predicted versus realized complete cycle cost against 0.165 diagnosis baseline and 0.164 quality reference. The deletion arm tests retained follow-up knowledge contribution.',
        evaluation='All 36 new chains finish learning before all 420 new terminal scores. Both cohorts are previously evaluated regressions. Reused 48 chains and 504 prior terminal values are explicit; no result-based exclusions or tuning.',
        baseline_runtime_reference='artifacts/joint_mismatch_investment_20261003/runtime/src',
        candidate_variant='Estimate initial complete cycle execution cost independently of the full reserve; retain current evidence and applicable paid cycle feedback.',
        prior_protocol=str((prior/'protocol.json').relative_to(ROOT)),
        learning_barrier_outputs=[str((BASE/name).relative_to(ROOT)) for name in COHORTS],
        candidate_source_status='Awaiting root GO; no new learning started.',
        execution={'maximum_workers':16,'status':'protocol prepared; waiting for candidate freeze and GO'})
    p['arms']={
        'anchor_164':{'runtime':'artifacts/shared_condition_research_20261003/runtime/src','knowledge':'empty','result_source':'reused 0.164 '+previous_group},
        'old_cumulative':{'runtime':'artifacts/joint_mismatch_investment_20261003/runtime/src','knowledge':'empty','result_source':'reused 0.165 '+previous_group},
        'new_cumulative':{'runtime':'artifacts/cold_cycle_investment_20261003/runtime/src','knowledge':'empty','result_source':'new run'}}
    if cohort=='successor_regression':
        p['arms']['no_marginal_history']={'runtime':'artifacts/cold_cycle_investment_20261003/runtime/src','knowledge':'empty','result_source':'new run','remove_action_experience_context':':condition_followup_research:'}
    p['source_accounting']='All new arms begin empty. The deletion arm removes only :condition_followup_research: action_experience contexts before each task/revisit; every other knowledge item and current-task paid state remains. 0.164 and 0.165 results are reused via references.'
    directory=BASE/cohort
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    for arm,source_arm in (('anchor_164','new_cumulative'),('old_cumulative','new_cumulative')):
        source=(ANCHOR/previous_group) if arm=='anchor_164' else prior
        for root in p['roots']:
            target=directory/'runs'/arm/root
            target.mkdir(parents=True,exist_ok=True)
            for name in ('checkpoint.json.gz','evaluation.json'):
                path=target/name
                if not path.exists():path.symlink_to(source/'runs'/source_arm/root/name)
print({'new_learning':36,'new_terminal_scores':420,'reused_chains':48,'reused_terminal_scores':504,'compared_chains':84,'compared_terminal_scores':924})
