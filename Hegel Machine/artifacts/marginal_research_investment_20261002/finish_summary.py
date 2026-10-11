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
def main():
    result={'external_model_calls':0,'new_learning_chains':48,'new_terminal_task_arm_scores':588,
        'reused_learning_chains':12,'compared_chains':60,'compared_terminal_task_arm_scores':672}
    details={}
    for cohort in ('seen_regression','fresh_successors'):
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
                'cost_delta':{key:totals['new_cumulative']['all_costs'][key]-totals[reference]['all_costs'][key] for key in totals[reference]['all_costs']}}
            if cohort=='fresh_successors':
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
        for arm in arms:
            tally=mechanism[arm]
            for root in roots:
                state=read(BASE/cohort/'runs'/arm/root/'checkpoint.json.gz')
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
                for item in state['results']:
                    position=item['task_position']
                    score=indexed[arm][root,position]
                    for index,action in enumerate(item['research']['actions']):
                        name=action['action']
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
                                tally['prior_information_experience_consumed']+=decision.get('information_yield',1.)<1.
                        if name in ('compare_method_followup_on_fresh_response','retain_unconfirmed_method_followup'):
                            tally[name]+=1
                            for key,value in action['stage_costs'].items():tally['followup_'+key]+=value
                            if action['before_loss'] is not None:
                                gain=action['before_loss']-action['after_loss']
                                tally['positive_followup_gain' if gain>1e-12 else 'negative_followup_gain' if gain< -1e-12 else 'zero_followup_gain']+=1
                        if ('marginal_context' in action or name in ('compare_method_followup_on_fresh_response','retain_unconfirmed_method_followup','stop_condition_followup_research')):
                            events.append({'arm':arm,'root':root,'position':position,'action_index':index,
                                'terminal_mse':score['mse'],'final_status':item['research']['status'],**trim(action)})
        result[cohort]={'baseline_source':'reused previous 0.160 candidate' if cohort=='seen_regression' else 'new run',
            'totals':totals,'comparisons':comparisons,'mechanism':{arm:dict(tally) for arm,tally in mechanism.items()}}
        details[cohort]={'task_pairs':task_pairs,'events':events,'largest_deteriorations':sorted(task_pairs,key=lambda x:x['delta'],reverse=True)[:6]}
    save(BASE/'compact_results.json',result)
    save(BASE/'paired_details.json',details)
    print(json.dumps({key:value['comparisons'] for key,value in result.items() if isinstance(value,dict) and 'comparisons' in value},indent=2))
if __name__=='__main__':main()
