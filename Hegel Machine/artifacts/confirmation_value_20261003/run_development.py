"""Local normal-condition continuation from three recorded paid reports."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gzip,json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
FIXTURE=ROOT/'artifacts/coherent_confirmation_20261003/paid_confirmation_cases.json.gz'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);parser.add_argument('--label',required=True);args=parser.parse_args()
    runtime=Path(args.runtime);runtime=runtime if runtime.is_absolute() else ROOT/runtime
    sys.path.insert(0,str(runtime));sys.path.insert(0,str(ROOT/'scripts'))
    from hegel_machine.cognition.unified_research import ResearchConfig,UnifiedResearchAgent
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition import instrument_condition_revisits as instruments
    from run_observed_condition_decoding_study import laboratory
    fixture=json.loads(gzip.decompress(FIXTURE.read_bytes()))
    output={'complete':False,'runtime':str(runtime.relative_to(ROOT)),'label':args.label,'fixture':str(FIXTURE.relative_to(ROOT)),
        'scope':'Both arms rescreen and reprice the actual historic report, charging the new screen inside its original available allowance. Known paid local reports replayed through the normal instrument continuation. Both arms use the recorded cumulative limits and identical local candidate; missing session-wide knowledge starts empty. This is not an exact shared-session replay.',
        'terminal_scores':0,'external_model_calls':0,'cases':[]}
    for case in fixture['cases']:
        previous=deepcopy(case['visit_previous_report'])
        for batch in previous['batches']:
            history=[]
            for row in batch['observations']:
                row.setdefault('history',deepcopy(history));history.append({**row['features'],'__target__':row['target']})
        limits=case['resource_limits'];config=replace(ResearchConfig(**case['config']),seed=case['task']['seed'],
            observation_budget=limits['observation_cost'],compute_budget=limits['compute_cost'])
        start=time.process_time()
        available={key:limits[key]-previous['costs'][key] for key in limits}
        screen_budget=SearchBudget(min(case['config']['compute_budget']//8,available['compute_cost']))
        captured=[]
        with laboratory(case['task']) as lab:
            agent=UnifiedResearchAgent(config,measurement_budget=limits['measurement_cost'])
            screen=instruments.screen_instrument_condition_revisit(agent,lab,previous,screen_budget)
            quoted_remaining={**available,'compute_cost':available['compute_cost']-screen_budget.spent}
            original_consider=instruments.consider_condition_research
            def capture(*args,**kwargs):
                value=original_consider(*args,**kwargs);captured.append(deepcopy(value));return value
            instruments.consider_condition_research=capture
            try: offer=instruments.instrument_condition_offer(screen,quoted_remaining,experience=agent.library.action_experience)
            finally: instruments.consider_condition_research=original_consider
            if offer is not None:
                agent.config=replace(config,compute_budget=limits['compute_cost']-screen_budget.spent)
                report=agent.run(lab,previous=previous,continuation_candidate=offer['candidate'])
            else:
                report={**deepcopy(previous),'visit_costs':{'observation_cost':0,'measurement_cost':0,'compute_cost':0,'queries':0}}
        total_costs={**report['visit_costs'],'compute_cost':report['visit_costs']['compute_cost']+screen_budget.spent}
        cpu=time.process_time()-start
        tail=report['actions'][len(previous['actions']):]
        revisions=[{k:v for k,v in a.items() if not isinstance(v,(dict,list))} for a in tail if a['action'] in ('revise_competing_observed_conditions','revise_paid_observed_condition')]
        confirmations=[{k:v for k,v in a.items() if not isinstance(v,(dict,list))} for a in tail if a['action'] in ('validate_continued_program','compare_method_followup_on_fresh_response','compare_successor_method_on_fresh_response')]
        decisions=[a for a in tail if a['action']=='choose_condition_research_investment']
        state=previous.get('condition_uncertainty') or {}
        entry={'root':case['root'],'position':case['task_position'],'limits':limits,'starting_costs':previous['costs'],
            'costs':total_costs,'execution_costs':report['visit_costs'],
            'screening_compute_cost':screen_budget.spent,'quoted_remaining':quoted_remaining,'purchased':offer is not None,
            'screen_revision_value':(screen or {}).get('plan',{}).get('revision_value'),
            'screening_decisions':captured,'cpu_seconds':cpu,'status':report['status'],'final_validation_loss':report.get('validation_loss'),
            'initial_program':previous['program'],'final_program':report['program'],'program_changed':previous['program']!=report['program'],
            'starting_condition':{'status':state.get('status'),'original_target_loss':state.get('original_target_loss'),'conditions':[{'threshold_intervals':c.get('threshold_intervals'),'endpoint_count':len(c.get('endpoint_programs',[])),'conditional_loss':c.get('conditional_loss'),'expected_prediction_ambiguity':c.get('expected_prediction_ambiguity')} for c in state.get('conditions',[])]},
            'paid_revisions':revisions,'fresh_confirmations':confirmations,'investment_decisions':decisions,
            'new_batches':report['batches'][len(previous['batches']):],
            'actions':[{k:v for k,v in a.items() if k not in ('program','baseline_program','condition_uncertainty','competing_hypotheses','alternatives','reading_mean_program')} for a in tail]}
        output['cases'].append(entry)
        (BASE/f'development_{args.label}.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:entry[k] for k in ('root','position','costs','program_changed','final_validation_loss','paid_revisions')}) ,flush=True)
    output['complete']=True;(BASE/f'development_{args.label}.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
