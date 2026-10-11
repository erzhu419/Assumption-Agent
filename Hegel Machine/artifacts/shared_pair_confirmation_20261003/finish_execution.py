"""Final small conclusions from saved summaries only; no learning or scoring."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def save(name,value):(BASE/name).write_text(json.dumps(value,indent=2)+'\n')
c=read(BASE/'compact_results.json')['successor_regression']; d=read(BASE/'paired_details.json')['successor_regression']
q=read(BASE/'quick_terminal_results.json'); full=q['new_cumulative']; prior=q['prior_pair_queue']; nofee=q['no_pair_cost_feedback']
fee={'matched_roots':0,'visit_sequence_changes':[], 'aligned_price_events':0,'changed_expected_cost_quotes':0,'changed_resource_shares':0,'changed_relative_gains':0}
for root in sorted({r['root'] for r in d['resource_use']}):
    visits={a:[v for v in d['revisit_outcomes'] if v['arm']==a and v['root']==root] for a in ('new_cumulative','no_pair_cost_feedback')}
    route=lambda v:[(x['position'],x['basis'],x['source_task_position'],x['costs']) for x in v]
    if route(visits['new_cumulative'])==route(visits['no_pair_cost_feedback']):fee['matched_roots']+=1
    else:fee['visit_sequence_changes'].append(root)
    quotes={a:[e for e in d['events'] if e['arm']==a and e['root']==root and e['action']=='price_pair_confirmation'] for a in visits}
    for a,b in zip(quotes['new_cumulative'],quotes['no_pair_cost_feedback']):
        if a['task_position']!=b['task_position']:continue
        fee['aligned_price_events']+=1
        fee['changed_expected_cost_quotes']+=a['expected_resource_costs']!=b['expected_resource_costs']
        fee['changed_resource_shares']+=a['investment_decision']['resource_share']!=b['investment_decision']['resource_share']
        fee['changed_relative_gains']+=a['relative_gain']!=b['relative_gain']
visits={a:[v for v in d['revisit_outcomes'] if v['arm']==a and v['root']=='root10'] for a in ('new_cumulative','prior_pair_queue')}
actual=next(v for v in d['shared_pair_visits'] if v['arm']=='new_cumulative' and v['root']=='root10')
actions=[{k:v for k,v in a.items() if k not in ('confirmation_pair','program','baseline_program','condition_uncertainty','support')} for a in actual['paid_pair_actions'] if a['action']!='plan_condition_confirmation']
diagnosis={'root':'root10','first_shared_path_difference':{'full':{'visit_ordinal':2,'position':10,'basis':'pending_pair_confirmation'},'prior':{'visit_ordinal':2,'position':4,'basis':'new_partial_method_distinguishing_experiment'}},'independent_pair':{'position':actual['position'],'costs':actual['costs'],'paid_actions':actions},'visit_sequences':{a:[{k:v[k] for k in ('position','visit_ordinal','basis','costs','resource_limits','before_validation_loss','after_validation_loss')} for v in vs] for a,vs in visits.items()},'position4_first_visit_same_costs':next(v for v in visits['new_cumulative'] if v['position']==4)['costs']==next(v for v in visits['prior_pair_queue'] if v['position']==4)['costs'],'finding':'The new position10 confirmation detects a real response counterexample; all-paid old/new losses are tied so it does not restore the old explanation. The subsequent shared path gives position4 two method visits versus three in the control, and spends its last visit on position10. Position4 loses a later revision opportunity; no isolated counterfactual separates knowledge propagation from shared scheduling.','position4_terminal':{'full':next(r['mse'] for r in full['rows'] if r['root']=='root10' and r['position']==4),'prior':next(r['mse'] for r in prior['rows'] if r['root']=='root10' and r['position']==4)}}
save('root10_pair_queue_diagnosis.json',diagnosis)
summary={'complete':True,'version':'0.174.0','decision':'global pair queue candidate rejected','production_reference':'0.173.0','tests_passed':78,'new_learning_chains':36,'new_terminal_task_arm_scores':504,'external_model_calls':0,'arms':{a:{k:v for k,v in r.items() if k not in ('rows','changed_scores_vs_prior')} for a,r in q.items()},'comparisons':c['comparisons'],'shared_pair_confirmation':c['shared_pair_confirmation'],'cost_feedback_contribution':fee,'control_reproduction':c['ablation_baseline_reproduction'],'full_vs_noqueue_percent':{'mse':100*(full['mse']/prior['mse']-1),'last7_mse':100*(full['last7_mse']/prior['last7_mse']-1),'compute':100*(full['costs']['compute_cost']/prior['costs']['compute_cost']-1),'cpu':100*(full['all_cpu_seconds']/prior['all_cpu_seconds']-1)},'stale_bug_cohort_cases':0,'stale_bug_coverage':'Final report plus each revisit.previous covers each task report before and after every executed visit; no pending paid-row mismatch was present.','correction':read(BASE/'correction/stopped_before_terminal.json'),'limitations':['The new global queue is not accepted: all-task and later-stage quality and total resources worsen.','Cost samples changed some forecasts but delivered no independent selection, quality or resource benefit in this cohort.','The correction is retained as frozen source and tests plus partial learning only, with no terminal evaluation.']}
save('execution_summary.json',summary)
print(json.dumps({'fee_contribution':fee,'relative':summary['full_vs_noqueue_percent'],'root10_pair_actual_cost':actual['costs']},indent=2))
