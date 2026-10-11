"""Apply the proposed branch-consequence ordering to saved public prefixes."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'initial_target_consequence_diagnosis.json'
d=json.loads(p.read_text()); rows=[]
for r in d['cases']:
 experiments=r['experiments'];key=lambda e:(e['number'],e['prediction'])
 current=next(e for e in experiments if all(e[k]==r['selected'][k] for k in ('number','prediction')))
 common=set.intersection(*[{(e['number'],e['prediction']) for e in c['positive_settings']} for c in r['target_support_by_class']])
 guaranteed=lambda e:min(b['target_signature_reduction'] for b in e['branches']) if all(b['remaining_classes'] for b in e['branches']) else 0
 possible=lambda e:max(b['target_signature_reduction'] for b in e['branches']) if all(b['remaining_classes'] for b in e['branches']) else 0
 if common:
  selected=current if key(current) in common else next(e for e in experiments if key(e) in common)
  role='conditional_member_target_trial'
 else:
  best=max(guaranteed(e) for e in experiments)
  value=guaranteed if best else possible
  best=max(value(e) for e in experiments)
  selected=current if value(current)==best else next(e for e in experiments if value(e)==best)
  role='guaranteed_nonlocal_member_test' if max(guaranteed(e) for e in experiments) else ('conditional_nonlocal_member_test' if best else 'incumbent')
 row={'case':r['case'],'before':r['selected'],'after':{'number':selected['number'],'prediction':selected['prediction']},'changed':key(current)!=key(selected),'role':role,'common_positive_settings':[{'number':n,'prediction':f} for n,f in sorted(common)],'before_value':{'guaranteed':guaranteed(current),'possible':possible(current)},'after_value':{'guaranteed':guaranteed(selected),'possible':possible(selected)}}
 rows.append(row)
 print(row['case'],row['before'],'=>',row['after'],row['role'],'CHANGED' if row['changed'] else 'same')
out={'scope':'No new task execution. Proposed target-first then guaranteed projected-signature gain, possible gain only when every guarantee is zero; keep incumbent ties. Current finite members only; no calibrated probability or truth data.', 'cases':rows,'changed':sum(r['changed'] for r in rows),'common_positive_cases':sum(bool(r['common_positive_settings']) for r in rows),'reproduced_initial_decisions':d['reproduced']}
(HERE/'proposed_rule_prefix_changes.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k not in ('scope','cases')})
