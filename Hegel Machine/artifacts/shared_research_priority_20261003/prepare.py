"""Prepare the fixed 0.167 repair against two reused quality references."""
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
    p.update(title='Compare retained method and instrument research on the same available resource pool',
        baseline_version='0.165.0',candidate_version='0.167.0',quality_anchor_version='0.164.0',
        development_selection='Both cohorts are previously evaluated regressions; all twelve roots and every protocol stage retained.',
        cohort=cohort,scope='previously evaluated short-sequence regression' if cohort=='seen_regression' else 'previously evaluated 14-stage repeated-mechanism regression',
        hypothesis_test='Retained method and instrument opportunities should be repriced against the same current shared pool and maximum resource share before competing; genuine complete reserves and independent confirmation remain unchanged.',
        primary_endpoint='All-task and successor-stage independent MSE, solved<=1e-6, false completion, full observation/measurement/compute and CPU, paired task/chain outcomes, all new failures and recoveries, actual method/instrument choices and the current-pool resource shares used to price them. Retain every existing task and failed case. Compare 0.167 to reused 0.165 diagnosis baseline and 0.164 quality reference, plus keep/delete follow-up knowledge contrast.',
        evaluation='All 36 new chains finish learning before all 420 new terminal scores. Both cohorts are previously evaluated regressions. Reused 48 chains and 504 prior terminal values are explicit; no result-based exclusions or tuning.',
        baseline_runtime_reference='artifacts/joint_mismatch_investment_20261003/runtime/src',
        candidate_variant='Reprice method and instrument proposals using the same current remaining pool and maximum resource fraction; begin from restored 0.165 and do not reintroduce rejected 0.166 cold pricing.',
        prior_protocol=str((prior/'protocol.json').relative_to(ROOT)),
        learning_barrier_outputs=[str((BASE/name).relative_to(ROOT)) for name in COHORTS],
        candidate_source_status='Awaiting root GO; no new learning started.',
        rejected_result_reference='artifacts/cold_cycle_investment_20261003/compact_results.json; supplementary negative history only, not rerun or pooled',
        execution={'maximum_workers':16,'status':'protocol prepared; waiting for candidate freeze and GO'})
    p['arms']={
        'anchor_164':{'runtime':'artifacts/shared_condition_research_20261003/runtime/src','knowledge':'empty','result_source':'reused 0.164 '+previous_group},
        'old_cumulative':{'runtime':'artifacts/joint_mismatch_investment_20261003/runtime/src','knowledge':'empty','result_source':'reused 0.165 '+previous_group},
        'new_cumulative':{'runtime':'artifacts/shared_research_priority_20261003/runtime/src','knowledge':'empty','result_source':'new run'}}
    if cohort=='successor_regression':
        p['arms']['no_marginal_history']={'runtime':'artifacts/shared_research_priority_20261003/runtime/src','knowledge':'empty','result_source':'new run','remove_action_experience_context':':condition_followup_research:'}
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
