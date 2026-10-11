"""Development contribution contrast from the same naturally paid two-task prefix."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import json,time
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.memory import KnowledgeLibrary
base={'environment':'objects','apparatus_seed':9013}
phases=[('total',{}),('event',{}),('total_event',{}),('total_event',{'relation_radius':.45}),('total_event',{}),('total_event',{'relation_radius':.45})]
tasks=[{**base,'readout':readout,'seed':10078+10007*i,**opts} for i,(readout,opts) in enumerate(phases)]
cfg=ResearchConfig(observation_budget=384,compute_budget=360000,minimum_improvement=1e-7,max_rounds=6,active_probes=True,seed=71)
path=Path('artifacts/composed_local_cycle_20261004/development_delete_a.json');assert not path.exists()
start=time.monotonic()
s=UnifiedResearchSession(cfg,tasks=tasks,measurement_budget=4608)
prefix=s.run(max_tasks=2)
ids=prefix['results'][0]['research']['concepts_added']
runs={}
for arm in ['full','delete_a']:
 state=deepcopy(prefix)
 if arm=='delete_a':state['knowledge']=KnowledgeLibrary.from_dict(state['knowledge']).without(ids).to_dict()
 session=UnifiedResearchSession(state=state)
 def checkpoint(report):
  r=report['results'][-1]['research'];print(json.dumps({'arm':arm,'task':len(report['results'])-1,'status':r['status'],'costs':r['costs'],'confirmation':r['confirmation_loss'],'concepts_added':r['concepts_added']}),flush=True)
 session.run(on_checkpoint=checkpoint);runs[arm]=session
out={'purpose':'development_actual_a_source_deletion_in_composed_radius_revision','protocol':{'tasks':tasks,'config':asdict(cfg),'measurement_budget':4608,'evaluation':{'length':64,'trajectories':4}},'deleted_source_ids':ids,'results':{},'external_model_calls':0}
for arm,session in runs.items():
 report=session.evaluate(length=64,trajectories=4)
 out['results'][arm]={'costs':report['costs'],'tasks':[]}
 for i,(r,e) in enumerate(zip(report['results'],report['evaluation'])):
  q=r['research'];out['results'][arm]['tasks'].append({'position':i,'status':q['status'],'costs':q['costs'],'confirmation_loss':q['confirmation_loss'],'evaluation':e,'program':q['program'],'concepts_added':q['concepts_added'],'transfers':[{k:x.get(k) for k in ['source_ids','status','accepted','validation_loss','validation_rows']} for x in q.get('transfer',[])]})
keys=['observation_cost','measurement_cost','compute_cost']
out['full_minus_delete_a_cost']={k:out['results']['full']['costs'][k]-out['results']['delete_a']['costs'][k] for k in keys}
out['elapsed_seconds']=time.monotonic()-start
path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(path),'elapsed':out['elapsed_seconds'],'full_minus_delete_a_cost':out['full_minus_delete_a_cost'],'losses':{a:[r['evaluation']['loss'] for r in s['tasks']] for a,s in out['results'].items()}}),flush=True)
