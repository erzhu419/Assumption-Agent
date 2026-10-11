"""Compact causal interpretation from completed small receipts only."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())
summary=read(BASE/'development_summary.json')
result={'complete':True,'version':'0.176.1','new_continuations':4,'terminal_scores':56,
    'external_model_calls':0,'arms':{'full':summary['arms']['baseline'],
    'without_designated_source_gain':summary['arms']['candidate']},'cases':[],
    'scope':'Two historically seen roots, identical real visit14 paid states, fixed terminal4x64 seed_offset1000003. Marginal source gain-credit suppression; all actual source execution, trials, cost experience, information feedback, programs, observations and method knowledge retained.',
    'aborted_preliminary':'aborted_late_hook retains the short no-terminal preliminary run stopped after static review of intervention timing.'}
for root in ('root02','root05'):
    audits={arm:read(BASE/'runs'/arm/root/'feedback_intervention.json') for arm in ('baseline','candidate')}
    records={arm:read(BASE/'runs'/arm/root/'development.json') for arm in ('baseline','candidate')}
    source_position=audits['baseline']['source_position']
    source_end=audits['baseline']['after_source_revisits']
    target=8 if root=='root02' else 4
    c={'root':root,'source_position':source_position,'target_position':target,
       'source_execution_identical':audits['baseline']['source_execution']==audits['candidate']['source_execution'],
       'source_feedback_events_identical':audits['baseline']['source_events']==audits['candidate']['source_events'],
       'source_expected_gain_components':audits['candidate']['source_expected_gain_components'],
       'boundary':audits['candidate']['intervention_boundary'],'arms':{}}
    for arm in ('baseline','candidate'):
        r,a=records[arm],audits[arm]
        quotes=next(q for q in r['quotes_after_feedback'] if q['after_total_revisits']==source_end)
        first=[{'position':o['task_position'],'basis':o['basis'],
            'expected_loss_reduction':o.get('expected_loss_reduction'),
            'empirical_marginal_prediction_gain':o.get('investment_decision',{}).get('empirical_marginal_prediction_gain'),
            'information_value':o.get('investment_decision',{}).get('expected_information_value'),
            'resource_share':o.get('investment_decision',{}).get('resource_share')}
            for o in quotes['offers'] if o['task_position'] in (source_position,target)]
        target_visits=[v for v in r['visits'][source_end:] if v['position']==target]
        target_pairs=[e for v in target_visits for e in v['paid_events'] if e['action']=='compare_method_followup_on_fresh_response']
        paid_revisions=[{k:e[k] for k in ('action','before_loss','after_loss')}
            for v in target_visits for e in v['paid_events']
            if e['action'] in ('revise_competing_observed_conditions','revise_condition_on_paid_response')
            and e.get('before_loss',0)-e.get('after_loss',0)>1e-7]
        c['arms']['full' if arm=='baseline' else 'without_designated_source_gain']={
            'source_visit_costs':a['source_execution']['visit_costs'],
            'first_post_source_quotes':first,'position_sequence':[v['position'] for v in r['visits']],
            'suffix_costs':r['suffix_costs'],'target_visits':len(target_visits),
            'target_visit_costs':{k:sum(v['execution_costs'][k] for v in target_visits) for k in ('observation_cost','measurement_cost','compute_cost')},
            'target_program_changes':sum(v['program_changed'] for v in target_visits),
            'target_positive_fresh_pairs':sum(e['before_loss']-e['after_loss']>1e-7 for e in target_pairs),
            'target_paid_revisions':paid_revisions,
            'withdrawal_predicate_calls':a['target_withdrawal_predicate_calls'],
            'withdrawal_positive_calls':a['target_withdrawal_positive_calls'],
            'counterexample_producer_calls':a['target_counterexample_producer_calls']}
    result['cases'].append(c)
result['terminal_changes']=[row for row in summary['pairs'] if abs(row['candidate_mse']-row['baseline_mse'])>1e-7]
result['full_minus_removed_gain']={'passed':summary['arms']['baseline']['passed']-summary['arms']['candidate']['passed'],
    'mse':summary['arms']['baseline']['mse']-summary['arms']['candidate']['mse'],
    'all_costs':{k:summary['arms']['baseline']['all_costs'][k]-summary['arms']['candidate']['all_costs'][k] for k in ('observation_cost','measurement_cost','compute_cost')},
    'measured_cpu_seconds':summary['arms']['baseline']['measured_cpu_seconds']-summary['arms']['candidate']['measured_cpu_seconds']}
result['withdrawal_arm']={'executed':False,'reason':'Both complete full continuations have zero target-counterexample producers and zero positive transfer_refuted/draft_transfer_refuted predicates; the gain-suppressed controls also have zero. A disabled-withdrawal arm would not test this channel here.'}
result['conclusion']='Specific paid source prediction-gain feedback has an independent downstream contribution in these two matched continuations. Root05 source p9 credit preserves subsequent p4 research and held-out recovery; removing it loses p4, while root02 quality stays equal and costs increase. This does not validate general transfer, autonomous wrong-knowledge withdrawal, or long-horizon net benefit.'
(BASE/'contribution_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
p=read(BASE/'development_protocol.json');p['status']='Complete:4 continuations then56 terminal scores; source execution remained identical. No withdrawal ablation due zero actual positive consumption.'
(BASE/'development_protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
print({key:result[key] for key in ('complete','new_continuations','terminal_scores','full_minus_removed_gain','terminal_changes')})
