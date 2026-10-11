"""Read-only current solver queries on recorded public response prefixes."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from hegel_machine.cognition.world_response_constraints import query_response_family
p=ROOT/'artifacts/response_nested_support_20261009/nested_support'
s=json.loads((p/'agent/shared/reference_recalibration.json').read_text())
a=json.loads((p/'analysis.json').read_text())
c=a[0]['support_analysis']['conditional_simplification']
f={int(k):v for k,v in c['family_research']['features'].items()}
est=c['assumption']['supports'][0]
results=[]
for count in (97,103,104,106):
 rs=s['response_settings'][:count]
 queries=[]
 # All public target-grid settings in the region spanning the recorded panel;
 # coarse whole-grid queries check that this focused interval is not arbitrary.
 settings=sorted(set(range(0,10001,100))|set(range(4400,5201)))
 for setting in settings:
  out=query_response_family(f,rs,est,[{'crystal':5,'frequency':float(setting),'activated':True}],compute_limit=100000)
  queries.append({'setting':setting,'status':out['status'],'compute_cost':out['compute_cost']})
 sat=[q['setting'] for q in queries if q['status']=='SAT_WITNESS']
 row={'responses':count,'statuses':dict(Counter(q['status'] for q in queries)),'sat_min':min(sat) if sat else None,'sat_max':max(sat) if sat else None,'sat_count':len(sat),'queries':queries}
 results.append(row)
 print({k:v for k,v in row.items() if k!='queries'},flush=True)
(Path(__file__).parent/'diagnosis.json').write_text(json.dumps({'family':est,'source':str(p),'scope':'Hypothetical target activation queried from public history only; no new physical observations','rows':results},indent=2)+'\n')
