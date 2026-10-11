"""Only small completed receipts: actual returns, downstream quotes and purchases."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

result={'complete':True,'scope':'Bundled method response attribution, executable AST applicability and future execution-cost correction; no independent factor attribution.', 'arms':{}}
for arm in ('baseline','candidate'):
    roots={}
    for root in ('root10','root11'):
        directory=BASE/'runs'/arm/root
        trace=read(directory/'development.json')
        feedback=read(directory/'method_feedback.json')
        returns=[]
        response_records=[]
        for visit in feedback['outcomes_before_downstream_quotation']:
            outcome=visit['outcome']
            for response in outcome.get('response_confirmations',[]):
                returns.append({'after_revisits':visit['after_revisits'],
                    'position':visit['position'],**response})
            for context,actions in visit.get('response_experience',{}).items():
                record=actions['successor_partial_method_response']
                response_records.append({'after_revisits':visit['after_revisits'],
                    'source_position':visit['position'],'context':context,
                    'applicability':record['knowledge_applicability'],
                    'trials':record['trials'],'expected_gain':record['expected_gain']})

        def response_source(offer,after_revisits):
            opportunity=offer.get('method_opportunity',{})
            matches=[record for record in response_records
                if record['after_revisits']<=after_revisits
                and record['applicability']==opportunity.get('method_applicability')
                and record['trials']==opportunity.get('observed_response_trials')
                and record['expected_gain']==opportunity.get('observed_response_gain')]
            if not matches:
                return None
            record=max(matches,key=lambda item:item['after_revisits'])
            return {key:value for key,value in record.items() if key!='applicability'}
        quotes=[]
        for queue in trace['quotes_after_feedback']:
            for offer in queue['offers']:
                if offer['basis']!='new_partial_method_distinguishing_experiment':
                    continue
                record=offer.get('method_feedback',{})
                opportunity=offer.get('method_opportunity',{})
                if not record.get('trials',0):
                    continue
                quotes.append({'after_revisits':queue['after_total_revisits'],
                    'position':offer['task_position'],'source_position':offer['source_task_position'],
                    'draft_ids':opportunity.get('draft_ids'),
                    'observed_response_trials':opportunity.get('observed_response_trials'),
                    'legacy_observed_visit_trials':opportunity.get('observed_visit_trials'),
                    'method_feedback':record,'expected_loss_reduction':offer['expected_loss_reduction'],
                    'expected_resource_costs':offer['expected_resource_costs'],
                    'method_applicability':opportunity.get('method_applicability'),
                    'consumed_response_source':response_source(offer,queue['after_total_revisits'])})
        purchases=[]
        for index,visit in enumerate(trace['visits']):
            offer=visit['selected_quote']
            if offer['basis']!='new_partial_method_distinguishing_experiment':
                continue
            comparisons=[{key:e[key] for key in ('probe_id','before_loss','after_loss','attempted_after_loss','attempted_source_draft_ids') if key in e}
                for e in visit['paid_events'] if e['action']=='compare_successor_method_on_fresh_response']
            purchases.append({'index':index,'position':visit['position'],
                'source_position':offer['source_task_position'],
                'draft_ids':offer.get('method_opportunity',{}).get('draft_ids'),
                'method_feedback':offer.get('method_feedback'),
                'expected_resource_costs':offer.get('expected_resource_costs'),
                'execution_costs':visit['execution_costs'],'screening_compute_cost':offer.get('screening_compute_cost'),
                'program_changed':visit['program_changed'],'fresh_comparisons':comparisons,
                'consumed_response_source':response_source(offer,index)})
        quoted_sources={(q['position'],tuple(q['draft_ids'] or []),
            q['consumed_response_source']['context'],q['consumed_response_source']['trials'])
            for q in quotes if q['consumed_response_source'] is not None}
        roots[root]={'actual_response_confirmations':returns,
            'positive_response_confirmations':sum(r['gain']>1e-7 for r in returns),
            'negative_response_confirmations':sum(r['gain']<-1e-7 for r in returns),
            'unique_target_draft_feedback_quotes':len(quoted_sources),
            'purchases_consuming_recorded_response_feedback':sum(
                p['consumed_response_source'] is not None for p in purchases),
            'cross_task_purchases_consuming_recorded_response_feedback':sum(
                p['consumed_response_source'] is not None
                and p['position']!=p['consumed_response_source']['source_position'] for p in purchases),
            'experienced_quotes_in_recorded_queues':quotes,
            'actual_method_purchases':purchases,
            'feedback_events':[event for event in trace['scheduler_events']
                if event['action'] in ('refresh_method_feedback','reconsider_cached_method')]}
    result['arms'][arm]=roots
(BASE/'method_consumption_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print({arm:{root:{'response_confirmations':len(data['actual_response_confirmations']),
    'positive':data['positive_response_confirmations'],'negative':data['negative_response_confirmations'],
    'method_purchases':len(data['actual_method_purchases'])} for root,data in roots.items()}
    for arm,roots in result['arms'].items()})
