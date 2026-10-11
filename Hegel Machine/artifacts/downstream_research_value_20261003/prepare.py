"""Prepare the fixed 0.168 method-feedback reopening study against two reused quality references."""
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'artifacts/shared_research_priority_20261003'
ANCHOR=ROOT/'artifacts/shared_condition_research_20261003'
COHORTS={'seen_regression':'seen_regression','successor_regression':'successor_regression'}
for cohort,previous_group in COHORTS.items():
    prior=PREVIOUS/previous_group
    p=copy.deepcopy(json.loads((prior/'protocol.json').read_text()))
    p.update(title='Reopen existing paid method offers after relevant measured method feedback',
        baseline_version='0.167.0',candidate_version='0.168.0',quality_anchor_version='0.164.0',
        development_selection='Both cohorts are previously evaluated regressions; all twelve roots and every protocol stage retained.',
        cohort=cohort,scope='previously evaluated short-sequence regression' if cohort=='seen_regression' else 'previously evaluated 14-stage repeated-mechanism regression',
        hypothesis_test='Relevant measured method feedback should invalidate an already-paid unsuccessful method quotation and reopen it when the existing draft can now repair the unresolved consumer. No new upstream research reward or extra draft is required.',
        primary_endpoint='All-task and successor-stage independent MSE, solved<=1e-6, false completion, full observation/measurement/compute and CPU, paired task/chain outcomes, all new failures and recoveries, actual method feedback, previously paid quote reopening and fresh consumer repair gains/costs. Retain every existing task and failed case. Compare 0.168 to reused 0.167 diagnosis baseline and 0.164 quality reference, plus a targeted ablation of the new method-feedback reopening hook.',
        evaluation='All 36 new chains finish learning before all 420 new terminal scores. Both cohorts are previously evaluated regressions. Reused 48 chains and 504 prior terminal values are explicit; no result-based exclusions or tuning.',
        baseline_runtime_reference='artifacts/shared_research_priority_20261003/runtime/src',
        candidate_variant='Build on 0.167 shared-pool pricing; relevant measured method feedback reopens an old paid method quote via UnifiedResearchSession._refresh_method_feedback. Do not add upstream research rewards. Rejected 0.166 cold pricing remains excluded.',
        prior_protocol=str((prior/'protocol.json').relative_to(ROOT)),
        learning_barrier_outputs=[str((BASE/name).relative_to(ROOT)) for name in COHORTS],
        candidate_source_status='Awaiting root GO; no new learning started.',
        rejected_result_reference='artifacts/cold_cycle_investment_20261003/compact_results.json; supplementary negative history only, not rerun or pooled',
        execution={'maximum_workers':16,'status':'protocol prepared; waiting for candidate freeze and GO'})
    p['arms']={
        'anchor_164':{'runtime':'artifacts/shared_condition_research_20261003/runtime/src','knowledge':'empty','result_source':'reused 0.164 '+previous_group},
        'old_cumulative':{'runtime':'artifacts/shared_research_priority_20261003/runtime/src','knowledge':'empty','result_source':'reused 0.167 '+previous_group},
        'new_cumulative':{'runtime':'artifacts/downstream_research_value_20261003/runtime/src','knowledge':'empty','result_source':'new run'}}
    if cohort=='successor_regression':
        p['arms']['no_downstream_feedback']={'runtime':'artifacts/downstream_research_value_20261003/runtime/src','knowledge':'empty','result_source':'new run','ablation_interface':'Override UnifiedResearchSession._refresh_method_feedback as a no-op; preserve normal method and condition experience, original screening and resource quotation'}
    p['source_accounting']='All new arms begin empty. The ablation overrides only UnifiedResearchSession._refresh_method_feedback as a no-op. Normal method and condition experience, original screening, resource quotation and paid evidence remain unchanged. 0.164 and 0.167 results are reused via references.'
    p['development_causal_correction']='Paid root11 p9 analysis established that the old draft0002 together with method experience already acquired by the keep arm can repair p9. Extra upstream p2 research or draft18 is not necessary; this candidate tests feedback reopening of an existing paid method opportunity, not an upstream research bonus.'
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
