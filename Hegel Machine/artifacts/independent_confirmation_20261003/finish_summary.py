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
    if event.get('action') in ('plan_condition_confirmation','validate_continued_program',
            'compare_method_followup_on_fresh_response','compare_successor_method_on_fresh_response'):
        return {k:v for k,v in event.items() if k not in ('reading_mean_program','competing_hypotheses','alternatives')}
    return {k:v for k,v in event.items() if k not in ('program','baseline_program','reading_mean_program','competing_hypotheses','alternatives')}
def main():
    result={'external_model_calls':0,'new_learning_chains':24,'new_terminal_task_arm_scores':336,
        'reused_learning_chains':24,'compared_chains':48,'compared_terminal_task_arm_scores':672}
    details={}
    prior=BASE.parent/'instrument_whole_visit_20261003'
    prior_compact=read(prior/'compact_results.json')
    prior_details=read(prior/'paired_details.json')
    for cohort in ('successor_regression',):
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
        confirmation_rows=[]
        protocol=read(BASE/cohort/'protocol.json')
        for arm in arms:
            tally=mechanism[arm]
            if arm in ('anchor_164','old_cumulative'):
                source_arm='anchor_164' if arm=='anchor_164' else 'new_cumulative'
                tally.update(prior_compact[cohort]['mechanism'][source_arm])
                for target,key in ((events,'events'),(resource_use,'resource_use'),(revisit_outcomes,'revisit_outcomes')):
                    target.extend({**row,'arm':arm} for row in prior_details[cohort][key] if row['arm']==source_arm)
                continue
            tally.update({'retain_method_confirmation_counterexample':0,
                'retained_joint_response_mismatch':0,'reopened_joint_response_mismatch':0,
                'refresh_method_feedback':0,'reconsider_cached_method':0,
                'planned_confirmation_pairs':0,'recorded_confirmation_pair_forecast_sum':0,
                'feedback_scheduled_quotes':0,'feedback_nonpositive_quotes':0,
                'feedback_refresh_compute_cost':0})
            for root in roots:
                state=read(BASE/cohort/'runs'/arm/root/'research_evidence.json.gz')
                confirmation_rows.extend({'arm':arm,'root':root,**entry}
                    for entry in state.get('posthoc_confirmation_diagnostics',[]))
                unused={key:protocol['limits']['chain_'+key.removesuffix('_cost')]-value for key,value in state['costs'].items()}
                resource_use.append({'arm':arm,'root':root,'learning_costs':state['costs'],
                    'remaining_pool':unused,'revisits':len(state['revisits']),
                    'screening_compute_cost':state.get('screening_compute_cost',0),
                    'scheduler_events':dict(Counter(event['action'] for event in state['scheduler_events']))})
                for key,value in unused.items():tally['remaining_'+key]+=value
                tally['all_scheduler_screening_compute_cost']+=state.get('screening_compute_cost',0)
                for event in state['scheduler_events']:
                    if event['action'] in ('refresh_method_feedback','reconsider_cached_method'):
                        tally[event['action']]+=1
                        tally['feedback_scheduled_quotes']+=bool(event.get('scheduled'))
                        tally['feedback_nonpositive_quotes']+=bool(event.get('withdrawn'))
                        tally['feedback_refresh_compute_cost']+=event.get('compute_cost',0)
                        events.append({'arm':arm,'root':root,'scope':'scheduler',**trim(event)})
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
                    first_phase=next((event for event in tail
                        if event['action']=='compare_method_followup_on_fresh_response'
                        and event.get('before_loss') is not None and event.get('after_loss') is not None),None)
                    cost_quote=visit.get('instrument_cost_quote',{})
                    first_phase_costs=(dict(first_phase['stage_costs']) if first_phase else None)
                    if first_phase_costs is not None:
                        first_phase_costs['compute_cost']-=cost_quote.get('screening_compute_cost',0)
                    entry={'arm':arm,'root':root,'position':visit['task_position'],'basis':visit['basis'],
                        'visit_ordinal':visit_index+1,'source_task_position':visit['source_task_position'],
                        'method_draft_id':visit.get('method_draft_id'),'method_draft_ids':visit.get('method_draft_ids',[]),
                        'before_status':before['status'],'after_status':after['status'],
                        'before_validation_loss':before.get('validation_loss'),'after_validation_loss':after.get('validation_loss'),
                        'before_condition_state':(before.get('condition_uncertainty') or {}).get('status'),
                        'after_condition_state':(after.get('condition_uncertainty') or {}).get('status'),
                        'before_prediction_ambiguity':(before.get('condition_uncertainty') or {}).get('expected_prediction_ambiguity'),
                        'after_prediction_ambiguity':(after.get('condition_uncertainty') or {}).get('expected_prediction_ambiguity'),
                        'costs':visit['visit_costs'],'resource_limits':visit['resource_limits'],
                        'instrument_cost_quote':cost_quote,
                        'first_scored_phase_execution_costs':first_phase_costs,
                        'after_first_phase_execution_costs':({key:visit['visit_costs'][key]-value
                            for key,value in first_phase_costs.items()} if first_phase_costs else None),
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
                    for label in ('instrument_revisits','instrument_cycles'):
                        samples=record.get(label,[])
                        tally[label+'_cost_samples']+=len(samples)
                        for sample in samples:
                            for key in ('observation_cost','measurement_cost','compute_cost'):
                                tally[label+'_sample_'+key]+=sample[key]
                for item in state['results']:
                    position=item['task_position']
                    score=indexed[arm][root,position]
                    for index,action in enumerate(item['research']['actions']):
                        name=action['action']
                        if name=='plan_condition_confirmation' and action.get('confirmation_pair'):
                            tally['planned_confirmation_pairs']+=1
                            tally['recorded_confirmation_pair_forecast_sum']+=action.get('expected_disagreement',0.)
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
                        if ('marginal_context' in action or 'sampling_context' in action or name in ('plan_condition_confirmation','validate_continued_program','compare_method_followup_on_fresh_response','retain_unconfirmed_method_followup','stop_condition_followup_research','retain_method_confirmation_counterexample','choose_method_followup_research','stop_method_response_reconstruction')):
                            events.append({'arm':arm,'root':root,'position':position,'action_index':index,
                                'terminal_mse':score['mse'],'final_status':item['research']['status'],**trim(action)})
        reproduction=None
        if 'prior_confirmation' in indexed:
            ablated=indexed['prior_confirmation']
            old=indexed['old_cumulative']
            task_differences=[]
            for key,row in ablated.items():
                changed={field:{'ablation':row.get(field),'baseline':old[key].get(field)}
                    for field in ('mse','trajectories','learning_costs','costs','status')
                    if row.get(field)!=old[key].get(field)}
                if changed:task_differences.append({'root':key[0],'position':key[1],'changed':changed})
            indexed_resources={arm:{r['root']:r for r in resource_use if r['arm']==arm}
                for arm in ('prior_confirmation','old_cumulative')}
            root_differences=[]
            for root in roots:
                new=indexed_resources['prior_confirmation'][root]
                old_use=indexed_resources['old_cumulative'][root]
                changed={field:{'ablation':new.get(field),'baseline':old_use.get(field)}
                    for field in ('learning_costs','screening_compute_cost','revisits')
                    if new.get(field)!=old_use.get(field)}
                if changed:root_differences.append({'root':root,'changed':changed})
            reproduction={'reference':'reused 0.170 new_cumulative',
                'scope':'Exact task MSE, trajectory scores, learning and terminal resource dictionaries and status; exact root learning ledger, screening fee and revisit count. CPU is measured, not expected to reproduce exactly.',
                'task_count':len(ablated),'tasks_exactly_reproduced':len(ablated)-len(task_differences),
                'roots_exactly_reproduced':len(roots)-len(root_differences),
                'task_differences':task_differences,'root_differences':root_differences,
                'all_resource_delta':{key:totals['prior_confirmation']['all_costs'][key]-totals['old_cumulative']['all_costs'][key]
                    for key in totals['old_cumulative']['all_costs']}}
        cost_learning={}
        for arm in ('new_cumulative','prior_confirmation'):
            visits=[visit for visit in revisit_outcomes if visit['arm']==arm
                and visit['basis']=='unresolved_instrument_condition']
            paired=[visit for visit in visits if visit.get('first_scored_phase_execution_costs') is not None]
            quoted=[visit for visit in visits if visit.get('instrument_cost_quote',{}).get('expected_resource_costs')]
            cost_learning[arm]={
                'instrument_visits':len(visits), 'visits_with_scored_phase':len(paired),
                'quoted_visits':len(quoted),
                'first_scored_phase_execution_costs':{key:sum(v['first_scored_phase_execution_costs'][key] for v in paired)
                    for key in ('observation_cost','measurement_cost','compute_cost')},
                'after_first_phase_execution_costs':{key:sum(v['after_first_phase_execution_costs'][key] for v in paired)
                    for key in ('observation_cost','measurement_cost','compute_cost')},
                'expected_execution_costs':{key:sum(v['instrument_cost_quote']['expected_resource_costs'][key] for v in quoted)
                    for key in ('observation_cost','measurement_cost','compute_cost')},
                'actual_execution_costs_of_quoted_visits':{key:sum(v['costs'][key] for v in quoted)
                    for key in ('observation_cost','measurement_cost','compute_cost')},
                'scope':'Execution only; original paid screening is subtracted once from the first phase; scheduler screening remains separately counted in whole-chain resources.',
            }
        confirmation_stats={}
        for arm in ('new_cumulative','prior_confirmation'):
            subset=[entry for entry in confirmation_rows if entry['arm']==arm]
            confirmation_stats[arm]={}
            for label,group in (('all_recorded_fresh_phases',subset),
                    ('explicit_pair_plans',[entry for entry in subset if entry['pair_plan']])):
                scored=[entry for entry in group if 'discriminating_rows' in entry]
                confirmation_stats[arm][label]={
                    'count':len(group), 'acquired_batches':sum(entry['acquired_rows']>0 for entry in group),
                    'programs_differ':sum(entry.get('programs_differ',False) for entry in scored),
                    'discriminating_batches':sum(entry['discriminating_rows']>0 for entry in scored),
                    'discriminating_rows':sum(entry['discriminating_rows'] for entry in scored),
                    'zero_discrimination_batches':sum(entry['discriminating_rows']==0 for entry in scored),
                    'positive_gain_batches':sum(entry['gain_positive'] for entry in scored),
                    'negative_gain_batches':sum(entry['gain_negative'] for entry in scored),
                    'zero_gain_batches':sum(not entry['gain_positive'] and not entry['gain_negative'] for entry in scored),
                    'common_mismatch_batches':sum(entry['common_mismatch'] for entry in scored),
                    'diagnostic_predictions':sum(entry['diagnostic_predictions'] for entry in group),
                    'expected_disagreement_sum':sum(entry.get('expected_disagreement') or 0. for entry in group),
                    'duration_disagreement_sum':sum(entry.get('duration_disagreement') or 0. for entry in group),
                    'boundary_disagreement_sum':sum(entry.get('boundary_disagreement') or 0. for entry in group),
                }
        confirmation_stats['reused_0170']={
            'explicit_pair_plans':0,
            'pointwise_prediction_disagreement':'Not recomputed from old large checkpoints; the contemporaneous same-source prior-confirmation arm supplies measured control diagnostics, and its task/ledger reproduction is reported separately.'}
        result[cohort]={'reference_sources':'reused 0.164 quality reference and 0.170 diagnosis baseline; both cohorts previously evaluated',
            'totals':totals,'comparisons':comparisons,'instrument_cost_learning':cost_learning,
            'confirmation_diagnostics':confirmation_stats,'mechanism':{arm:dict(tally) for arm,tally in mechanism.items()},
            **({'ablation_baseline_reproduction':reproduction} if reproduction is not None else {})}
        details[cohort]={'task_pairs':task_pairs,'events':events,'resource_use':resource_use,'revisit_outcomes':revisit_outcomes,
            'posthoc_confirmation_diagnostics':confirmation_rows,
            'focus_scores':{arm:[{'root':k[0],'position':k[1],'mse':r['mse'],'status':r['status']} for k,r in indexed[arm].items() if k in {('root07',5),('root04',1),('root01',1),('root09',3),('root06',4),('root10',4),('root10',9),('root11',9),('root08',9)}] for arm in arms},
            'largest_deteriorations':sorted(task_pairs,key=lambda x:x['delta'],reverse=True)[:6]}
    save(BASE/'compact_results.json',result)
    save(BASE/'paired_details.json',details)
    print(json.dumps({key:value['comparisons'] for key,value in result.items() if isinstance(value,dict) and 'comparisons' in value},indent=2))
if __name__=='__main__':main()
