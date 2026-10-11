"""One paid prefix, queue-only cycle-cost ablation, one actual revisit per arm."""
import gzip,json,sys,time
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
STUDY=ROOT/'artifacts/downstream_research_value_20261003'
sys.path.insert(0,str(ROOT/'scripts'))
from run_observed_condition_decoding_study import laboratory,load_runtime
protocol=json.loads((STUDY/'successor_regression/protocol.json').read_text())
module=load_runtime(protocol,'new_cumulative')
from hegel_machine.cognition import instrument_condition_revisits as icr
from hegel_machine.cognition import condition_research_returns as returns
source=STUDY/'successor_regression/runs/new_cumulative/root07/visit17.json.gz'
with gzip.open(source,'rt') as stream: original=json.load(stream)
assert len(original['results'])==14 and [v['task_position'] for v in original['revisits']]==[5,5,5]
real_estimate=icr.instrument_cycle_cost_estimate
real_record=returns.record_condition_research
real_calibration=returns._current_calibration
context={}
def program_difference(a,b):
    numbers=[];other=[]
    def walk(a,b,path=''):
        if a==b:return
        if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
            for key in a:walk(a[key],b[key],path+'/'+key)
        elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
            numbers.append({'path':path,'before':a,'after':b,'absolute_delta':abs(a-b)})
        else:other.append(path)
    walk(a,b)
    return {'exactly_equal':a==b,'numeric_changes':len(numbers),'largest_numeric_changes':sorted(numbers,key=lambda v:v['absolute_delta'],reverse=True)[:4],'max_absolute_delta':max((x['absolute_delta'] for x in numbers),default=0.),'nonnumeric_changes':other[:6]}
def question_view(q):
    if q is None:return None
    return {k:v for k,v in q.items() if k!='program'}
def traced_calibration(question,previous,experiments,plan=None):
    answer=real_calibration(question,previous,experiments,plan)
    context['calibration'].append({'phase':context['phase'],'question':question_view(question),'program_vs_last_question':program_difference((previous.get('last_question') or {}).get('program'),(question or {}).get('program')),'sampling_context':(plan or {}).get('sampling_context'),'retained_trial_count':len(previous.get('sampling_trials',[])),'experiment_count':len(experiments),'experiment_changed_program_count':sum(bool((x.get('research_return') or {}).get('changed_program')) for x in experiments),'calibration':deepcopy(answer)})
    return answer
returns._current_calibration=traced_calibration
def traced_record(previous,before,after,plan,**kwargs):
    answer=real_record(previous,before,after,plan,**kwargs)
    first,last=returns._question(before),returns._question(after)
    context['recording'].append({'phase':context['phase'],'before_question':question_view(first),'after_question':question_view(last),'same_question':returns._same_question(first,last),'program_difference':program_difference((first or {}).get('program'),(last or {}).get('program')),'explicit_changed_program':kwargs.get('changed_program',False),'last_outcome':deepcopy(answer['last_outcome']),'current_question_returns':deepcopy(answer['current_question_returns'])})
    return answer
returns.record_condition_research=traced_record
def quote_view(offer):
    return {key:deepcopy(offer.get(key)) for key in ('task_position','basis','relative_gain','minimum_resources','expected_resource_costs','investment_decision')}
result={'source':str(source.relative_to(ROOT)),'runtime':protocol['arms']['new_cumulative']['runtime'],'scope':'One shared paid prefix, two queue cost readings, at most one real follow-up visit per arm; no terminal or further study.','intervention':'Only instrument_condition_revisits.instrument_cycle_cost_estimate returns the unchanged hard reserve instead of reading instrument_cycles. All information, predictive gains, acquisition plans and complete action_experience remain intact.','arms':{}}
for arm in ('original','no_cycle_cost_discount'):
    context={'phase':'queue','calibration':[],'recording':[]}
    icr.instrument_cycle_cost_estimate=real_estimate if arm=='original' else lambda record,plan,resources:dict(resources)
    session=module.UnifiedResearchSession(state=original,lab_factory=laboratory)
    before_costs=session._costs();before_experience=deepcopy(session.library.action_experience)
    initial_screen=next(x['screen'] for x in session.state['condition_revisit_quotes'] if x['task_position']==5)
    question=returns._question(initial_screen['condition_uncertainty'])
    prefix=initial_screen['marginal_context'].rsplit(':knowledge:',1)[0]+':knowledge:'
    records=[(key,value['condition_followup_research']) for key,value in session.library.action_experience.items() if key.startswith(prefix) and 'condition_followup_research' in value and value['condition_followup_research'].get('knowledge_applicability')==initial_screen['marginal_applicability']]
    session._offer_response_questions()
    queue=sorted(session.state['pending_revisits'],key=lambda x:x['relative_gain'],reverse=True)
    offer=queue[0]
    item={'before_remaining':session._remaining(),'queue_costs':{k:session._costs()[k]-before_costs[k] for k in before_costs},'experience_exactly_preserved_by_quote':session.library.action_experience==before_experience,'queue':[quote_view(x) for x in queue],'target_current_question':question_view(question),'target_plan':{k:deepcopy(v) for k,v in initial_screen['plan'].items() if k!='probe'},'target_plan_probe_id':initial_screen['plan']['probe']['probe_id'],'target_record_contexts':[key for key,value in records],'target_record_totals':[{k:v for k,v in value.items() if isinstance(v,(int,float))} for key,value in records],'target_instrument_cycles':[deepcopy(value.get('instrument_cycles',[])) for key,value in records]}
    context['phase']='paid_revisit'
    position=offer['task_position'];old_actions=len(session.state['results'][position]['research']['actions']);session.state['pending_revisits'].remove(offer)
    started=time.process_time();success=session._perform_revisit(offer);elapsed=time.process_time()-started
    report=session.state['results'][position]['research'];visit=session.state['revisits'][-1];tail=report['actions'][old_actions:]
    wanted={'choose_condition_research_investment','compare_method_followup_on_fresh_response','stop_condition_followup_research','revise_condition_on_paid_response','validate_continued_program'}
    event_view=[]
    for event in tail:
        if event['action'] not in wanted:continue
        event_view.append({k:deepcopy(v) for k,v in event.items() if k in ('action','buy','reason','expected_marginal_value','resource_share','expected_information_fraction','information_yield','marginal_prediction_trials','reopened','current_question_trials','current_question_information_batches','current_question_predicted_information_batches','expected_cycle_costs','probe_id','before_loss','after_loss','stage_costs','changed_program','loss','accepted','current_fit_confirmed','observation_rows')})
    item.update({'selected_position':position,'executed':success,'paid_visit_costs':visit['visit_costs'],'paid_cpu_seconds':elapsed,'whole_incremental_costs':{k:session._costs()[k]-before_costs[k] for k in before_costs},'paid_events':event_view,'final_status':report['status'],'condition_state':question_view(returns._question(report.get('condition_uncertainty'))),'new_history_records':context['recording'],'calibration_calls':context['calibration']})
    result['arms'][arm]=item
    (BASE/'cost_scope_development.json').write_text(json.dumps(result,indent=2)+'\n')
    print({'arm':arm,'selected':position,'queue':[(x['task_position'],x['relative_gain']) for x in queue[:3]],'costs':visit['visit_costs'],'recorded_outcomes':len(context['recording'])},flush=True)
result['complete']=True
(BASE/'cost_scope_development.json').write_text(json.dumps(result,indent=2)+'\n')
