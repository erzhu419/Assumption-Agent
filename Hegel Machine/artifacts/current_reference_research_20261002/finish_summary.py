"""Compact paired results with explicit reused data and complete charged costs."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
BASE = Path(__file__).resolve().parent

def read(path):
    if path.suffix == '.gz':
        with gzip.open(path,'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def change(new,old):
    return 100*(new/old-1) if old else None

def counts(pairs,atol=0.):
    return dict(Counter('tie' if math.isclose(new,old,rel_tol=1e-9,abs_tol=atol) else 'win' if new<old else 'loss' for new,old in pairs))

def compact_event(event):
    return {k:v for k,v in event.items() if k not in ('program','reading_mean_program','competing_hypotheses','alternatives')}

def main():
    compact = {'external_model_calls':0,'new_learning_chains':48,'new_terminal_tasks':552,
        'compared_chains_including_reuse':72,'compared_terminal_tasks_including_reuse':792,
        'scope':'Seen mixed and instrument regressions are separate from fresh same-factory instances. Costs include screening, revisits and terminals. Process CPU is reported as observed, not attributed across different worker scheduling.'}
    details = {}
    for cohort in ('seen_regression','fresh_replication','instrument_regression'):
        summary=read(BASE/cohort/'summary.json')
        instrument=cohort=='instrument_regression'
        newarm,oldarm=('new_cumulative','old_cumulative') if instrument else ('candidate','baseline_158')
        if instrument:
            for total in summary['totals'].values():
                total['all_cpu_seconds']=total['learning_cpu_seconds']+total['terminal_cpu_seconds']
        rows=summary['rows' if instrument else 'tasks']
        new,old=summary['totals'][newarm],summary['totals'][oldarm]
        result={'baseline_source':'new run' if cohort=='fresh_replication' else 'reused preceding 0.158 full result',
            'totals':summary['totals'],'families':{},
            'cost_change_percent':{k:change(new['all_costs'][k],old['all_costs'][k]) for k in new['all_costs']},
            'cost_delta':{k:new['all_costs'][k]-old['all_costs'][k] for k in new['all_costs']}}
        task_pairs=[]
        roots=sorted({r['root'] for r in rows})
        for family in (('instrument',) if instrument else ('noise','condition')):
            group={arm:{(r['root'],r['position']):r for r in rows if r['arm']==arm and (instrument or r['family']==family)} for arm in (newarm,oldarm)}
            pairs=[(r['mse'],group[oldarm][k]['mse']) for k,r in group[newarm].items()]
            chains=[]
            for root in roots:
                means=[sum(r['mse'] for r in group[arm].values() if r['root']==root)/sum(r['root']==root for r in group[arm].values()) for arm in (newarm,oldarm)]
                chains.append(means)
            family_result={'mse_change_percent':change(sum(a for a,b in pairs),sum(b for a,b in pairs)),
                'task_counts':counts(pairs,0. if family=='noise' else 1e-12),
                'chain_counts':counts(chains,0. if family=='noise' else 1e-12),
                'chains':{root:{'candidate':pair[0],'baseline':pair[1]} for root,pair in zip(roots,chains)}}
            if not instrument:
                family_result['predictive_mse_change_percent']=change(new['families'][family]['mean_predictive_mse'],old['families'][family]['mean_predictive_mse'])
            result['families'][family]=family_result
            task_pairs.extend({'root':k[0],'position':k[1],'family':family,'baseline_mse':group[oldarm][k]['mse'],'candidate_mse':r['mse'],'delta':r['mse']-group[oldarm][k]['mse']} for k,r in group[newarm].items())
        event_counts={arm:Counter() for arm in (oldarm,newarm)}
        quote_counts={arm:Counter() for arm in (oldarm,newarm)}
        costs=[]
        for root in roots:
            states={arm:read(BASE/cohort/'runs'/arm/root/'checkpoint.json.gz') for arm in (oldarm,newarm)}
            records=[]
            for arm,state in states.items():
                for event in state.get('scheduler_events',[]):
                    if event['action']=='reprice_response_question':
                        quote_counts[arm]['reestimates']+=1
                        quote_counts[arm]['worth_comparing' if event['worth_comparing'] else 'not_worth_comparing']+=1
                        quote_counts[arm]['compute_cost']+=event['compute_cost']
                        records.append({'arm':arm,'location':'scheduler',**compact_event(event)})
                for item in state['results']:
                    for index,action in enumerate(item['research']['actions']):
                        event_counts[arm][action['action']]+=1
                        if any(word in action['action'] for word in ('repric','refresh_response','current_response','plan_condition_confirmation')):
                            records.append({'arm':arm,'position':item['task_position'],'action_index':index,**compact_event(action)})
            costs.append({'root':root,'learning_cost_delta':{k:states[newarm]['costs'][k]-states[oldarm]['costs'][k] for k in new['all_costs']},
                'revisits':{arm:len(states[arm]['revisits']) for arm in states},
                'process_cpu_seconds':{arm:states[arm]['study']['cpu_seconds'] for arm in states},
                'changed_program_positions':[i for i,(a,b) in enumerate(zip(states[newarm]['results'],states[oldarm]['results'])) if a['research']['program']!=b['research']['program']],
                'events':records})
        result['learning_action_counts']={arm:dict(c) for arm,c in event_counts.items()}
        result['current_reference_quote_counts']={arm:dict(c) for arm,c in quote_counts.items()}
        details[cohort]={'chains':costs,'task_pairs':task_pairs,'largest_deteriorations':sorted(task_pairs,key=lambda r:r['delta'],reverse=True)[:6]}
        compact[cohort]=result
    save(BASE/'compact_results.json',compact)
    save(BASE/'paired_details.json',details)
    print(json.dumps({k:{x:v[x] for x in ('cost_delta','families')} for k,v in compact.items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':
    main()
