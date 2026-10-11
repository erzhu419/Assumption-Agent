from pathlib import Path
import json, time
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.meta_prior_research import _sources
base={'environment':'objects','apparatus_seed':9013}
tasks=[{**base,'readout':readout,'seed':10078+10007*i,**opts} for i,(readout,opts) in enumerate([
 ('total',{}),('event',{}),('total_event',{}),('total_event',{'relation_radius':.45}),('total_event',{})])]
cfg=ResearchConfig(observation_budget=384,compute_budget=360000,minimum_improvement=1e-7,max_rounds=6,active_probes=True,seed=71)
session=UnifiedResearchSession(cfg,tasks=tasks,measurement_budget=4608)
start=time.monotonic()
def checkpoint(report):
 if report['results']:
  r=report['results'][-1]['research']; print(json.dumps({'task':len(report['results'])-1,'status':r['status'],'costs':r['costs'],'kind':r['selected_kind'],'confirmation':r['confirmation_loss'],'calls':list(_sources(Program.from_dict(r['program']))) if r['program'] else [],'concepts_added':r['concepts_added']}),flush=True)
session.run(on_checkpoint=checkpoint)
report=session.evaluate(length=64,trajectories=4)
path=Path('artifacts/composed_local_cycle_20261004/development_baseline.json')
assert not path.exists();path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'elapsed':time.monotonic()-start,'evaluation':report['evaluation'],'costs':report['costs']}),flush=True)
