"""Compact three-arm outcomes, complete costs and actual knowledge consumption."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
BASE=Path(__file__).resolve().parent
PREFIX=':condition_followup_research:'
def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return json.load(stream)
    return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def counts(pairs):
    result=dict(win=0,tie=0,loss=0)
    for new,old in pairs:
        result['tie' if math.isclose(new,old,rel_tol=1e-9,abs_tol=1e-12) else 'win' if new<old else 'loss']+=1
    return result
def trim(event):
    return {k:v for k,v in event.items() if k not in ('program','baseline_program','reading_mean_program','competing_hypotheses','alternatives')}
def cycle_cost_pairs(events):
    """Match committed session prices to their next independent fresh settlement."""
    grouped={}
    for event in events:
        if 'position' in event and 'action_index' in event:
            grouped.setdefault((event['arm'],event['root'],event['position']),[]).append(event)
    pairs=[]
    for (arm,root,position),actions in grouped.items():
        actions.sort(key=lambda event:event['action_index'])
        visit=0
        for index,event in enumerate(actions):
            if (event['action']!='choose_condition_research_investment'
                    or event.get('investment_scope')!='session_remaining'
                    or event.get('expected_cycle_costs') is None):
                continue
            visit+=1
            settlement=None
            for following in actions[index+1:]:
                if (following['action']=='choose_condition_research_investment'
                        and following.get('investment_scope')=='session_remaining'):
                    break
                if following['action'] in ('compare_method_followup_on_fresh_response',
                                            'retain_unconfirmed_method_followup'):
                    settlement=following
                    break
            actual=None
            if settlement is not None:
                actual=dict(settlement['stage_costs'])
                actual['compute_cost']=max(0,actual['compute_cost']-event.get('compute_spent_before_decision',0))
            pairs.append({'arm':arm,'root':root,'position':position,'cycle_at_position':visit,
                'decision_action_index':event['action_index'],
                'expected_cycle_costs':event['expected_cycle_costs'],
                'actual_execution_costs':actual,
                'settled_stage_costs':dict(settlement['stage_costs']) if settlement is not None else None,
                'screening_compute_cost':event.get('compute_spent_before_decision',0),
                'fresh_pair_scored':settlement is not None and settlement['before_loss'] is not None,
                'marginal_prediction_trials':event.get('marginal_prediction_trials',0),
                'decision_reason':event.get('reason'),
                'terminal_mse':event['terminal_mse'],
                'before_loss':settlement.get('before_loss') if settlement else None,
                'after_loss':settlement.get('after_loss') if settlement else None})
    return pairs

def cycle_forecast_summary(pairs,arms):
    summaries={}
    for arm in arms:
        rows=[row for row in pairs if row['arm']==arm]
        complete=[row for row in rows if row['fresh_pair_scored']]
        expected={key:sum(row['expected_cycle_costs'][key] for row in complete)
            for key in ('observation_cost','measurement_cost','compute_cost')}
        actual={key:sum(row['actual_execution_costs'][key] for row in complete) for key in expected}
        errors=[row['expected_cycle_costs']['compute_cost']-row['actual_execution_costs']['compute_cost'] for row in complete]
        summaries[arm]={'forecast_evidence':'recorded' if rows else 'not_recorded',
            'executed_priced_cycles':len(rows),'complete_fresh_pair_cycles':len(complete),
            'without_fresh_pair_cycles':len(rows)-len(complete),'summed_expected_costs':expected,'summed_actual_costs':actual,
            'mean_absolute_compute_error':sum(abs(error) for error in errors)/len(errors) if errors else None,
            'mean_compute_overprediction':sum(errors)/len(errors) if errors else None,
            'compute_overpredicted_cycles':sum(error>1e-12 for error in errors),
            'compute_underpredicted_cycles':sum(error< -1e-12 for error in errors),
            'whole_visit_cost_pairs':sum(row.get('whole_visit_costs') is not None for row in complete),
            'summed_whole_visit_costs':{key:sum(row['whole_visit_costs'][key] for row in complete
                if row.get('whole_visit_costs') is not None) for key in expected},
            'summed_later_work_costs':{key:sum(row['later_work_costs'][key] for row in complete
                if row.get('later_work_costs') is not None) for key in expected}}
    return summaries

def main():
    result={'external_model_calls':0,'new_learning_chains':36,'new_terminal_task_arm_scores':420,
        'reused_learning_chains':48,'compared_chains':84,'compared_terminal_task_arm_scores':924}
    details={}
    prior=BASE.parent/'joint_mismatch_investment_20261003'
    prior_compact=read(prior/'compact_results.json')
    prior_details=read(prior/'paired_details.json')
    for cohort in ('seen_regression','successor_regression'):
        summary=read(BASE/cohort/'summary.json')
        arms=list(summary['totals'])
        rows=summary['rows']
        totals=summary['totals']
        for total in totals.values():
            total['all_cpu_seconds']=total['learning_cpu_seconds']+total['terminal_cpu_seconds']
        roots=sorted({row['root'] for row in rows})
        indexed={arm:{(row['root'],row['position']):row for row in rows if row['arm']==arm} for arm in arms}
        comparisons={}
        task_pairs=[]
        for reference in (arm for arm in arms if arm!='new_cumulative'):
            selected=indexed['new_cumulative']
            baseline=indexed[reference]
            values=[(row['mse'],baseline[key]['mse']) for key,row in selected.items()]
            chain_pairs=[(sum(r['mse'] for k,r in selected.items() if k[0]==root),
                sum(r['mse'] for k,r in baseline.items() if k[0]==root)) for root in roots]
            entry={'task_counts':counts(values),'chain_counts':counts(chain_pairs),
                'mse_change_percent':100*(sum(a for a,b in values)/sum(b for a,b in values)-1),
                'cost_delta':{key:totals['new_cumulative']['all_costs'][key]-totals[reference]['all_costs'][key] for key in totals[reference]['all_costs']},
                'newly_failed':[{'root':k[0],'position':k[1],'candidate_mse':r['mse'],'reference_mse':baseline[k]['mse']} for k,r in selected.items() if r['mse']>1e-6 and baseline[k]['mse']<=1e-6],
                'newly_solved':[{'root':k[0],'position':k[1],'candidate_mse':r['mse'],'reference_mse':baseline[k]['mse']} for k,r in selected.items() if r['mse']<=1e-6 and baseline[k]['mse']>1e-6]}
            if cohort=='successor_regression':
                for label,predicate in (('first_round',lambda p:p<7),('successors',lambda p:p>=7)):
                    pairs=[(r['mse'],baseline[k]['mse']) for k,r in selected.items() if predicate(k[1])]
                    entry[label]={'candidate_mse':sum(a for a,b in pairs)/len(pairs),
                        'reference_mse':sum(b for a,b in pairs)/len(pairs),'task_counts':counts(pairs),
                        'solved':{arm:sum(r['mse']<=1e-6 for k,r in indexed[arm].items() if predicate(k[1])) for arm in ('new_cumulative',reference)}}
            comparisons[reference]=entry
            task_pairs.extend({'reference':reference,'root':k[0],'position':k[1],
                'candidate_mse':r['mse'],'reference_mse':baseline[k]['mse'],'delta':r['mse']-baseline[k]['mse']} for k,r in selected.items())
        mechanism={arm:Counter() for arm in arms}
        events=[]
        resource_use=[]
        revisit_outcomes=[]
        protocol=read(BASE/cohort/'protocol.json')
        for arm in arms:
            tally=mechanism[arm]
            if arm in ('anchor_164','old_cumulative'):
                source_arm='old_cumulative' if arm=='anchor_164' else 'new_cumulative'
                tally.update(prior_compact[cohort]['mechanism'][source_arm])
                for target,key in ((events,'events'),(resource_use,'resource_use'),(revisit_outcomes,'revisit_outcomes')):
                    target.extend({**row,'arm':arm} for row in prior_details[cohort][key] if row['arm']==source_arm)
                continue
            tally.update({'retain_method_confirmation_counterexample':0,
                'retained_joint_response_mismatch':0,'reopened_joint_response_mismatch':0})
            for root in roots:
                state=read(BASE/cohort/'runs'/arm/root/'checkpoint.json.gz')
                unused={key:protocol['limits']['chain_'+key.removesuffix('_cost')]-value for key,value in state['costs'].items()}
                resource_use.append({'arm':arm,'root':root,'learning_costs':state['costs'],
                    'remaining_pool':unused,'revisits':len(state['revisits']),
                    'screening_compute_cost':state.get('screening_compute_cost',0),
                    'scheduler_events':dict(Counter(event['action'] for event in state['scheduler_events']))})
                for key,value in unused.items():tally['remaining_'+key]+=value
                tally['all_scheduler_screening_compute_cost']+=state.get('screening_compute_cost',0)
                for event in state['scheduler_events']:
                    if event['action'] in ('screen_instrument_condition_revisit','price_instrument_condition_revisit'):
                        tally[event['action']]+=1
                        if event['action']=='screen_instrument_condition_revisit':
                            tally['instrument_screening_compute_cost']+=event.get('compute_cost',0)
                        events.append({'arm':arm,'root':root,'scope':'scheduler',**trim(event)})
                tally['cached_condition_revisit_quotes']+=len(state.get('condition_revisit_quotes',[]))
                for visit_index,visit in enumerate(state['revisits']):
                    following=next((item for item in state['revisits'][visit_index+1:] if item['task_position']==visit['task_position']),None)
                    after=following['previous'] if following else state['results'][visit['task_position']]['research']
                    before=visit['previous']
                    tail=after['actions'][len(before['actions']):]
                    experiment=visit.get('candidate',{}).get('instrument_condition_experiment',{})
                    cycle_quote_index=next((len(before['actions'])+index for index,event in enumerate(tail)
                        if event['action']=='choose_condition_research_investment'
                        and event.get('investment_scope')=='session_remaining'),None)
                    entry={'arm':arm,'root':root,'position':visit['task_position'],'basis':visit['basis'],
                        'before_status':before['status'],'after_status':after['status'],
                        'decision_action_index':cycle_quote_index,
                        'cycle_cost_forecast':experiment.get('cycle_cost_forecast'),
                        'cycle_screening_compute_cost':experiment.get('screening_compute_cost'),
                        'before_validation_loss':before.get('validation_loss'),'after_validation_loss':after.get('validation_loss'),
                        'before_condition_state':(before.get('condition_uncertainty') or {}).get('status'),
                        'after_condition_state':(after.get('condition_uncertainty') or {}).get('status'),
                        'before_prediction_ambiguity':(before.get('condition_uncertainty') or {}).get('expected_prediction_ambiguity'),
                        'after_prediction_ambiguity':(after.get('condition_uncertainty') or {}).get('expected_prediction_ambiguity'),
                        'costs':visit['visit_costs'],'resource_limits':visit['resource_limits'],
                        'actions':dict(Counter(item['action'] for item in tail)),
                        'terminal_mse':indexed[arm][root,visit['task_position']]['mse']}
                    revisit_outcomes.append(entry)
                    tally['revisit_'+visit['basis']]+=1
                    for key,value in visit['visit_costs'].items():tally['revisit_'+key]+=value
                    tally['revisit_new_completion']+=before['status']!='completed' and after['status']=='completed'
                removals=state['study'].get('marginal_history_removals',[])
                tally['removal_events']+=len(removals)
                tally['removed_contexts']+=sum(event['removed_contexts'] for event in removals)
                for context,records in state['knowledge'].get('action_experience',{}).items():
                    if PREFIX not in context:
                        continue
                    record=records.get('condition_followup_research',{})
                    tally['retained_marginal_contexts']+=1
                    tally['retained_measured_contexts']+=bool(record.get('trials',0) or record.get('information_batches',0))
                    tally['retained_prediction_trials']+=record.get('trials',0)
                    tally['retained_information_batches']+=record.get('information_batches',0)
                    for field in ('information_event_sum','predicted_information_event_sum','information_gain_sum'):
                        if field in record:tally['retained_'+field]+=record[field]
                for item in state['results']:
                    position=item['task_position']
                    score=indexed[arm][root,position]
                    for index,action in enumerate(item['research']['actions']):
                        name=action['action']
                        if name=='retain_method_confirmation_counterexample':
                            tally['retain_method_confirmation_counterexample']+=1
                            tally['retained_joint_response_mismatch']+=bool(action.get('joint_response_mismatch'))
                        if name=='choose_method_followup_research' and action.get('reason')=='new_revision_refuted_with_joint_response_mismatch':
                            tally['reopened_joint_response_mismatch']+=1
                        if name=='stop_method_response_reconstruction':
                            tally['reconstruction_stop_'+action.get('reason','unspecified')]+=1
                        if (action.get('sampling_context') or {}).get('basis')=='paid_local_control_residuals':
                            tally['paid_background_plans']+=1
                            tally['paid_background_rows']+=action.get('local_reading_rows',0)
                        if name=='allocate_condition_confirmation':
                            tally['allocations']+=1
                            tally['acquired_rows']+=action['acquired_rows']
                            tally['batches']+=len(action['outcomes'])
                            for outcome in action['outcomes']:
                                for field in ('informative','interval_narrowed','new_counterexample','changed_program'):
                                    tally[field]+=int(outcome[field])
                        if name in ('choose_condition_research_investment','allocate_condition_confirmation'):
                            decisions=[action] if name=='choose_condition_research_investment' else action.get('decisions',[])
                            for decision in decisions:
                                tally['decisions']+=1
                                tally['purchased_decisions']+=bool(decision.get('buy'))
                                tally['prior_prediction_trials_consumed']+=bool(decision.get('marginal_prediction_trials',0))
                                if 'information_yield' in decision:
                                    tally['historical_information_discount_applied']+=decision['information_yield']<1.
                                if 'compute_spent_before_decision' in decision:
                                    prefix=decision['compute_spent_before_decision']
                                    tally['decisions_with_paid_compute_prefix']+=prefix>0
                                    tally['maximum_paid_compute_prefix']=max(tally['maximum_paid_compute_prefix'],prefix)
                                    tally['maximum_estimated_remaining_compute_cost']=max(
                                        tally['maximum_estimated_remaining_compute_cost'],decision['estimated_remaining_compute_cost'])
                                if 'response_risk_known' in decision:
                                    tally['decisions_with_known_response_risk']+=decision['response_risk_known']
                                    tally['decisions_with_unknown_response_risk']+=not decision['response_risk_known']
                        if name in ('compare_method_followup_on_fresh_response','retain_unconfirmed_method_followup'):
                            tally[name]+=1
                            for key,value in action['stage_costs'].items():tally['followup_'+key]+=value
                            if action['before_loss'] is not None:
                                gain=action['before_loss']-action['after_loss']
                                tally['positive_followup_gain' if gain>1e-12 else 'negative_followup_gain' if gain< -1e-12 else 'zero_followup_gain']+=1
                        if ('marginal_context' in action or 'sampling_context' in action or name in ('compare_method_followup_on_fresh_response','retain_unconfirmed_method_followup','stop_condition_followup_research','retain_method_confirmation_counterexample','choose_method_followup_research','stop_method_response_reconstruction')):
                            events.append({'arm':arm,'root':root,'position':position,'action_index':index,
                                'terminal_mse':score['mse'],'final_status':item['research']['status'],**trim(action)})
        cycle_pairs=cycle_cost_pairs(events)
        quotes={(row['arm'],row['root'],row['position'],row.get('decision_action_index')):row
            for row in revisit_outcomes if row.get('decision_action_index') is not None}
        for row in cycle_pairs:
            quote=quotes.get((row['arm'],row['root'],row['position'],row['decision_action_index']),{})
            row['cycle_cost_forecast']=quote.get('cycle_cost_forecast')
            row['whole_visit_costs']=quote.get('costs')
            if quote.get('cycle_screening_compute_cost') is not None:
                row['screening_compute_cost']=quote['cycle_screening_compute_cost']
                if row['settled_stage_costs'] is not None:
                    row['actual_execution_costs']={**row['settled_stage_costs'],
                        'compute_cost':max(0,row['settled_stage_costs']['compute_cost']-row['screening_compute_cost'])}
            row['later_work_costs']=({key:row['whole_visit_costs'][key]-row['actual_execution_costs'][key]
                for key in ('observation_cost','measurement_cost','compute_cost')} if row['whole_visit_costs'] is not None
                and row['actual_execution_costs'] is not None else None)
        result[cohort]={'cycle_cost_forecast':cycle_forecast_summary(cycle_pairs,arms),
            'reference_sources':'reused 0.164 quality reference and 0.165 diagnosis baseline; both cohorts previously evaluated',
            'totals':totals,'comparisons':comparisons,'mechanism':{arm:dict(tally) for arm,tally in mechanism.items()}}
        details[cohort]={'cycle_cost_pairs':cycle_pairs,'task_pairs':task_pairs,'events':events,'resource_use':resource_use,'revisit_outcomes':revisit_outcomes,
            'focus_scores':{arm:[{'root':k[0],'position':k[1],'mse':r['mse'],'status':r['status']} for k,r in indexed[arm].items() if k in {('root04',1),('root01',1),('root09',1),('root11',9)}] for arm in arms},
            'largest_deteriorations':sorted(task_pairs,key=lambda x:x['delta'],reverse=True)[:6]}
    save(BASE/'compact_results.json',result)
    save(BASE/'paired_details.json',details)
    print(json.dumps({key:value['comparisons'] for key,value in result.items() if isinstance(value,dict) and 'comparisons' in value},indent=2))
if __name__=='__main__':main()
