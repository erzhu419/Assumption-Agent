"""Describe selected companions separately from restored and actual purchases."""
import json
import os
from pathlib import Path
BASE=Path(os.environ.get('HEGEL_PORTFOLIO_OUTPUT',str(Path(__file__).resolve().parent)))

def read(path):
    return json.loads(path.read_text())

protocol=read(BASE/'development_protocol.json')
summary=read(BASE/'development_summary.json')
result={'complete':True,'candidate_version':protocol['candidate_version'],
    'scope':protocol['scope'],'arms':{},'terminal_pairs':summary['pairs']}
for arm in protocol['arms']:
    roots={}
    restored=[]
    selected_companions=0
    for root in protocol['roots']:
        directory=BASE/'runs'/arm/root
        execution=read(directory/'companion_execution.json')['actual_visits']
        trace=read(directory/'development.json')
        pairs=[event for event in trace['scheduler_events']
            if event['action']=='choose_research_portfolio' and event.get('companion') is not None]
        selected_companions+=len(pairs)
        root_restored=[]
        for index,(actual,paid) in enumerate(zip(execution,trace['visits'])):
            decision=actual['selected'].get('instrument_decision') or actual['selected'].get('investment_decision') or {}
            if (not actual['executed'] or decision.get('local_buy') is not False
                    or decision.get('reason')!='selected_portfolio_companion' or decision.get('buy') is not True):
                continue
            comparisons=[{key:event.get(key) for key in ('action','before_loss','after_loss','attempted_after_loss','probe_id')}
                for event in paid['paid_events'] if event['action'] in (
                    'compare_method_followup_on_fresh_response','compare_successor_method_on_fresh_response')]
            revisions=[{key:event.get(key) for key in ('action','before_loss','after_loss')}
                for event in paid['paid_events'] if event['action'] in (
                    'revise_competing_observed_conditions','revise_condition_on_paid_response')
                and event.get('before_loss',0)-event.get('after_loss',0)>1e-7]
            item={'root':root,'visit_index':index,'position':paid['position'],
                'decision':decision,'allocation':actual['allocation_limits'],
                'actual_costs':actual['actual_visit_costs'],'program_changed':paid['program_changed'],
                'fresh_comparisons':comparisons,'paid_revisions':revisions}
            root_restored.append(item)
            restored.append(item)
        roots[root]={'selected_companion_decisions':len(pairs),
            'actual_restored_companion_purchases':root_restored,
            'actual_task_sequence':[visit['position'] for visit in trace['visits'] if visit['executed']]}
    result['arms'][arm]={'metrics':summary['arms'][arm],'roots':roots,
        'selected_companion_decisions':selected_companions,
        'actual_restored_companion_purchase_count':len(restored),
        'restored_purchase_costs':{key:sum(item['actual_costs'][key] for item in restored)
            for key in ('observation_cost','measurement_cost','compute_cost')},
        'restored_purchase_program_changes':sum(item['program_changed'] for item in restored),
        'restored_purchase_positive_fresh_pairs':sum(event['before_loss']-event['after_loss']>1e-7
            for item in restored for event in item['fresh_comparisons'])}
a,c=summary['arms']['baseline'],summary['arms']['candidate']
result['candidate_minus_baseline']={'passed':c['passed']-a['passed'],'mse':c['mse']-a['mse'],
    'all_costs':{key:c['all_costs'][key]-a['all_costs'][key] for key in c['all_costs']},
    'measured_cpu_seconds':c['measured_cpu_seconds']-a['measured_cpu_seconds']}
result['task_wtl']={'win':sum(p['candidate_mse']<p['baseline_mse']-1e-7 for p in summary['pairs']),
    'tie':sum(abs(p['candidate_mse']-p['baseline_mse'])<=1e-7 for p in summary['pairs']),
    'loss':sum(p['candidate_mse']>p['baseline_mse']+1e-7 for p in summary['pairs'])}
result['new_failures']=[p for p in summary['pairs'] if p['baseline_mse']<=protocol['pass_mse']<p['candidate_mse']]
result['recoveries']=[p for p in summary['pairs'] if p['candidate_mse']<=protocol['pass_mse']<p['baseline_mse']]
(BASE/'compact_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print({arm:{'passed':data['metrics']['passed'],'mse':data['metrics']['mse'],
    'restored_actual':data['actual_restored_companion_purchase_count']} for arm,data in result['arms'].items()})
