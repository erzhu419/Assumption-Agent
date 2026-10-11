"""Paid ordinary-source competition; no instrument or synthetic gain injection."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]

def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,separators=(',',':')).encode()
    path.write_bytes(gzip.compress(data,compresslevel=1) if path.suffix=='.gz' else data+b'\n')

def runtime(protocol,arm):
    path=(ROOT/protocol['arms'][arm]['runtime']).resolve()
    # Each arm is a separate import universe. The shell may already expose
    # the live source tree, so remove a previously loaded package before
    # loading the frozen baseline.
    for name in list(sys.modules):
        if name == 'hegel_machine' or name.startswith('hegel_machine.'):
            del sys.modules[name]
    sys.path.insert(0,str(path))
    from hegel_machine.cognition import unified_session
    assert Path(unified_session.__file__).is_relative_to(path)
    return unified_session

def factory(item):
    from hegel_machine.cognition.environments import make_suite
    from hegel_machine.cognition.measurement_lab import MeasurementBatch
    from hegel_machine.cognition.models import Probe
    from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
    if item['kind']=='unknown':
        return UnknownTaskLab(item['task_index'],item['seed'])
    class SuiteLaboratory:
        channels=()
        def __init__(self):
            self.environment=make_suite(item['suite_seed'],partition=item['partition'])[item['task_index']]
            self.spec=replace(self.environment.spec,task_id=item['task_id'])
            self.acquisitions=[];self.observation_cost=0;self.measurement_cost=0
        def spec_for(self,channels=()):
            assert not channels
            return self.spec
        @property
        def costs(self):
            return {'observation_cost':self.observation_cost,'measurement_cost':0,'queries':len(self.acquisitions)}
        def acquire(self,probe,channels=()):
            assert not channels
            rows=self.environment.observe(probe)
            self.observation_cost+=len(rows)
            self.acquisitions.append({'probe_id':probe.probe_id,'sequence':list(probe.sequence),'channels':[]})
            return MeasurementBatch(rows,len(rows),0,())
        def restore(self,acquisitions):
            self.__init__()
            for row in acquisitions:self.acquire(Probe(row['probe_id'],tuple(row['sequence'])))
        def evaluation_rows(self,channels=(),*,seed=1009,length=12,trajectories=4):
            assert not channels
            other=deepcopy(self.environment)
            other._seed=seed  # Keep the already selected mechanism; change only evaluator draws.
            other._evaluation_trajectory_length=length
            other._evaluation_trajectories=trajectories
            return other.evaluation_rows()
        def close(self):pass
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
    return SuiteLaboratory()

def prepare(protocol,case):
    module=runtime(protocol,'baseline_0180')
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.knowledge_returns import transfer_condition
    from hegel_machine.cognition.models import Observation
    conf=module.ResearchConfig(**protocol['config'])
    definition=protocol['cases'][case]
    session=module.UnifiedResearchSession(conf,tasks=definition['tasks'],measurement_budget=768,lab_factory=factory)
    cpu=time.process_time()
    for position,item in enumerate(definition['tasks']):
        limited=position in definition['partial_positions']
        with factory(item) as lab:
            config=replace(conf,seed=item['seed'],compute_budget=protocol['partial_initial_compute'] if limited else conf.compute_budget)
            report=UnifiedResearchAgent(config,library=session.library,measurement_budget=768).run(lab)
        session.state['results'].append({'task_position':position,'available_concepts':[], 'research':report})
    session.state['status']='paused'
    state=session.to_dict()
    state['study']={'prepare_cpu_seconds':time.process_time()-cpu,'case':case,'terminal_rows':0}
    conditions=[]
    for pos in definition['partial_positions']:
        report=state['results'][pos]['research']
        rows=[Observation(x['features'],x['target'],tuple(x['history'])) for b in report['batches'] for x in b['observations']]
        with factory(definition['tasks'][pos]) as lab: condition=transfer_condition(lab.spec,rows)
        conditions.append({'position':pos,'condition':condition,'paid_rows':len(rows),'status':report['status']})
    receipt={'complete':True,'source_costs':state['costs'],'concept_count':len(session.library.concepts),'partial_conditions':conditions,'conditions_equal':all(x['condition']==conditions[0]['condition'] for x in conditions),'terminal_rows':0,'note':'Actual partial targets followed by normal paid source learning; no research_return or gain injected.'}
    save(BASE/'prefixes'/case/'checkpoint.json.gz',state);save(BASE/'prefixes'/case/'receipt.json',receipt)
    print({'phase':'prepare','case':case,'concepts':receipt['concept_count'],'conditions_equal':receipt['conditions_equal']},flush=True)

def learn(protocol,arm,case):
    module=runtime(protocol,arm)
    import hegel_machine.cognition.unified_revisits as revisit
    from hegel_machine.cognition.knowledge_returns import transfer_condition,return_priority
    state=read(BASE/'prefixes'/case/'checkpoint.json.gz')
    screens=[];original=revisit._successor_candidates
    def observe(agent,spec,rows,budget,relevant):
        condition=transfer_condition(spec,rows)
        before=budget.spent
        priorities=[{'source_id':c.concept_id,'priority':return_priority(c.reuse_history,condition),'matching_returns':sum(e.get('kind')=='research_return' and e.get('condition')==condition for e in c.reuse_history)} for c in agent.library.concepts if c.concept_id in relevant]
        result=original(agent,spec,rows,budget,relevant)
        candidates=result[0] if isinstance(result,tuple) else result
        screens.append({'task_id':spec.task_id,'condition':condition,'source_priorities':priorities,'candidate_sources':[x.memory_id for x in candidates],'candidate_count':len(candidates),'compute_cost':budget.spent-before})
        return result
    revisit._successor_candidates=observe
    session=module.UnifiedResearchSession(state=state,lab_factory=factory)
    cpu=time.process_time()
    # Reprice identical paid evidence under each runtime; no cached baseline offer is injected.
    ids=[c.concept_id for c in session.library.concepts]
    session._offer_new_knowledge(protocol['cases'][case]['source_positions'][-1],ids)
    result=session.run()
    receipt={'complete':result['status']=='completed','arm':arm,'case':case,'costs':result['costs'],'prefix_costs':state['costs'],'prefix_cpu_seconds':state['study']['prepare_cpu_seconds'],'cpu_seconds':time.process_time()-cpu,'screens':screens,'scheduler_events':result['scheduler_events'],'visits':[{'position':v['task_position'],'source_position':v['source_task_position'],'candidate':v['candidate'],'visit_costs':v['visit_costs']} for v in result['revisits']],'returns':[{'source_id':c.concept_id,'records':[e for e in c.reuse_history if e.get('kind')=='research_return']} for c in session.library.concepts],'actions':[{ 'position':r['task_position'],'actions':[e for e in r['research']['actions'] if e['action'] in ('compare_continuation_candidates','validate_continued_program','record_transfer_counterexample','knowledge_return','withdraw_target_transfer')]} for r in result['results']]}
    save(BASE/'runs'/arm/case/'checkpoint.json.gz',result)
    save(BASE/'runs'/arm/case/'receipt.json',receipt)
    print({'phase':'learn','arm':arm,'case':case,'complete':receipt['complete'],'visits':len(result['revisits'])},flush=True)

def evaluate(protocol,arm,case):
    for a in protocol['arms']:
        for c in protocol['cases']:assert read(BASE/'runs'/a/c/'receipt.json')['complete']
    module=runtime(protocol,arm)
    state=read(BASE/'runs'/arm/case/'checkpoint.json.gz')
    session=module.UnifiedResearchSession(state=state,lab_factory=factory)
    cpu=time.process_time();result=session.evaluate(**protocol['terminal'])
    save(BASE/'runs'/arm/case/'evaluation.json',{'scores':result['evaluation'],'cpu_seconds':time.process_time()-cpu})

def main():
    protocol=read(BASE/'protocol.json')
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','learn','evaluate'),required=True);parser.add_argument('--case',choices=tuple(protocol['cases']));parser.add_argument('--arm',choices=tuple(protocol['arms']));args=parser.parse_args()
    if args.case:
        if args.phase=='prepare':prepare(protocol,args.case)
        else:globals()[args.phase](protocol,args.arm,args.case)
        return
    jobs=[('baseline_0180',case) for case in protocol['cases']] if args.phase=='prepare' else [(arm,case) for arm in protocol['arms'] for case in protocol['cases']]
    for arm,case in jobs:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--phase',args.phase,'--case',case,'--arm',arm],cwd=ROOT,check=True)

if __name__=='__main__':main()
