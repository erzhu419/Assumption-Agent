from pathlib import Path
import sys,json,traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime/scripts'))
from run_composed_local_cycle import protocol,run_cycle
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
orig=UnifiedResearchAgent._support
seed=int(sys.argv[1]) if len(sys.argv)>1 else 24007
case=next(c for c in json.loads((ROOT/'protocol.json').read_text())['cases'] if c['apparatus_seed']==seed)
def diagnose(self,program,training,validation,spec,budget):
 required=program.required_features(); mismatch=[]
 for part,rows in [('training',training),('validation',validation)]:
  for i,row in enumerate(rows):
   absent=required-set(row.features)
   if absent:mismatch.append({'part':part,'index':i,'absent':sorted(absent),'features':sorted(row.features),'history':len(row.history)})
 if mismatch:
  frames=[]
  import inspect
  for frame in inspect.stack():
   if frame.function in ['run','_perform_revisit']:
    f=frame.frame.f_locals;frames.append({'fn':frame.function,'position':f.get('position'),'selected_channels':f.get('selected_channels'),'confirmation_channels':f.get('confirmation_channels'),'resumed':f.get('resumed_incumbent') is not None,'selected_kind':f.get('selected',{}).get('kind'),'actions_tail':[{k:v for k,v in a.items() if k in ['action','reason','status','selected_kind','channels','purpose','accepted','loss']} for a in f.get('actions',[])[-8:]]})
  info={'case':case,'spec':spec.feature_names,'program':program.to_dict(),'mismatches':mismatch,'frames':frames}
  (ROOT/f'development_mismatch_{seed}.json').write_text(json.dumps(info,indent=2)+'\n')
  print(json.dumps({'case':case,'mismatch_count':len(mismatch),'mismatch':mismatch[:2],'frames':frames}),flush=True)
 return orig(self,program,training,validation,spec,budget)
UnifiedResearchAgent._support=diagnose
try:
 r=run_cycle(protocol(**case));print(json.dumps({'case':case,'completed':True,'cycle':r['cycle_passed']}))
except Exception as e:
 print(json.dumps({'case':case,'error':repr(e)}));traceback.print_exc()
