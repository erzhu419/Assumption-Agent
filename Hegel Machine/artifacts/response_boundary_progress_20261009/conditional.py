"""Counterfactual response consequences from the same 103-response public prefix."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from hegel_machine.cognition.world_response_constraints import query_response_family
p=ROOT/'artifacts/response_nested_support_20261009/nested_support'
s=json.loads((p/'agent/shared/reference_recalibration.json').read_text())
a=json.loads((p/'analysis.json').read_text())
c=a[0]['support_analysis']['conditional_simplification']
f={int(k):v for k,v in c['family_research']['features'].items()};est=c['assumption']['supports'][0]
rs=s['response_settings'][:103]
base=json.loads((Path(__file__).parent/'diagnosis.json').read_text())['rows'][1]
settings=[q['setting'] for q in base['queries'] if q['status']=='SAT_WITNESS']
rows=[]
for crystal,setting in [(1,14.8),(1,16.8),(2,19.),(2,21.),(3,30.4),(3,32.4)]:
 row={'experiment':{'crystal':crystal,'frequency':setting},'outcomes':[]}
 for activated in (False,True):
  cond={**row['experiment'],'activated':activated}
  base_out=query_response_family(f,rs,est,[cond],compute_limit=100000)
  status_map={}
  if base_out['status']=='SAT_WITNESS':
   for target in settings:
    out=query_response_family(f,rs,est,[cond,{'crystal':5,'frequency':float(target),'activated':True}],compute_limit=100000)
    status_map.setdefault(out['status'],[]).append(target)
  kept=status_map.get('SAT_WITNESS',[])
  row['outcomes'].append({'activated':activated,'status':base_out['status'],'sat_min':min(kept) if kept else None,'sat_max':max(kept) if kept else None,'sat_count':len(kept),'exclusions':len(status_map.get('UNSAT_VERIFIED',[])),'unknown':len(status_map.get('UNKNOWN',[]))})
 rows.append(row); print(row,flush=True)
(Path(__file__).parent/'conditional.json').write_text(json.dumps({'response_count':103,'baseline_sat_points':len(settings),'rows':rows},indent=2)+'\n')
