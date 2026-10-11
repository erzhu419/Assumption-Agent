"""Read only paid first-decision prefixes; no task execution or fixture truth."""
import json
import sys
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tests')]
from scripts.run_response_forecast_classes import load_runtime
load_runtime(ROOT/'artifacts/response_trajectory_choice_20261011/runtime/before')
from hegel_machine.cognition.world_response_transfer import reuse_response_knowledge
from hegel_machine.cognition.world_experiment_choice import forecast_equivalence_classes
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def key(e): return {'number':e['number'],'prediction':e['prediction']}
def positive_targets(ids, experiments):
    return sorted({e['number'] for e in experiments if ids and all(e['predictions'][i] for i in ids)})
def signatures(ids, experiments):
    return {tuple(e['predictions'][i] for e in experiments) for i in ids}
def analysis(experiments, reps, e):
    targets=[x for x in experiments if 'restore_frequency' not in x and x['id']!=e['id']]
    others=[x for x in targets if x['number']!=e['number']]
    before_sig=len(signatures(reps,targets))
    before_other=len(signatures(reps,others))
    before_pos=positive_targets(reps, targets)
    by_target={n:[x for x in targets if x['number']==n] for n in sorted({x['number'] for x in targets})}
    branches=[]
    for outcome in (False,True):
        ids=[i for i in reps if e['predictions'][i]==outcome]
        branches.append({'outcome':outcome,'remaining_classes':len(ids),
            'target_signatures':len(signatures(ids, targets)),
            'target_signature_reduction':before_sig-len(signatures(ids,targets)) if ids else None,
            'other_crystal_signatures':len(signatures(ids,others)),
            'other_crystal_signature_reduction':before_other-len(signatures(ids,others)) if ids else None,
            'common_positive_targets_except_tested_point':positive_targets(ids,targets),
            'new_common_positive_targets_except_tested_point':sorted(set(positive_targets(ids,targets))-set(before_pos)),
            'remaining_target_signatures':{str(n):len(signatures(ids,exps)) for n,exps in by_target.items()},
            'confirmed_by_measured_positive':e['number'] if outcome and 'restore_frequency' not in e else None})
    nonempty=[b for b in branches if b['remaining_classes']]
    for branch in nonempty:
        supported=set(branch['common_positive_targets_except_tested_point'])
        if branch['confirmed_by_measured_positive'] is not None:
            supported.add(branch['confirmed_by_measured_positive'])
        branch['targets_with_observed_or_common_positive']=sorted(supported)
    return {**key(e),'is_reference':'restore_frequency' in e,
        'before_target_signatures_except_tested_point':before_sig,
        'before_other_crystal_signatures':before_other,
        'before_common_positive_targets_except_tested_point':before_pos,
        'branches':branches,
        'worst_targets_with_observed_or_common_positive':min(len(b['targets_with_observed_or_common_positive']) for b in nonempty),
        'mean_targets_with_observed_or_common_positive_by_forecast_class':sum(b['remaining_classes']*len(b['targets_with_observed_or_common_positive']) for b in nonempty)/len(reps),
        'worst_target_signature_reduction':min((b['target_signature_reduction'] for b in nonempty),default=0),
        'mean_target_signature_reduction_by_forecast_class':sum(b['remaining_classes']*b['target_signature_reduction'] for b in nonempty)/len(reps),
        'mean_new_common_positive_targets_by_forecast_class':sum(b['remaining_classes']*len(b['new_common_positive_targets_except_tested_point']) for b in nonempty)/len(reps)}
rows=[]
for case in read(ROOT/'artifacts/response_forecast_classes_20261011/protocol.json')['cases']:
    trace=read(ROOT/'artifacts/response_forecast_classes_20261011/after'/case/'frozen_numeric/research_trace.json')
    question=next(q for q in trace['questions'] if q.get('answer',{}).get('frozen_transfer'))
    state=deepcopy(trace['state']); state['completed']=[]
    state['response_settings']=[r for r in state['response_settings'] if max(r.get('evidence',[0]))<question['opened_before_observation']]
    knowledge=state['frozen_response_knowledge']
    result=reuse_response_knowledge(state,[{'measurements':knowledge['measurements']}],'shared/consolidated_method.py')
    chosen=result['selection']['selected_experiment']; archived=trace['comparisons'][0]['selected_experiment']
    assert chosen==archived,case
    classes=forecast_equivalence_classes(result['experiments']); reps=[g[0] for g in classes]
    experiments=[analysis(result['experiments'],reps,e) for e in result['experiments']]
    rows.append({'case':case,'reproduced':True,'opened_before_observation':question['opened_before_observation'],
        'prefix_response_count':len(state['response_settings']),'model_count':len(result['hypotheses']),
        'forecast_class_count':len(classes),'selected':key(chosen),'experiments':experiments,
        'target_support_by_class':[{'class_size':len(c),'representative':c[0],
            'positive_settings':[key(e) for e in result['experiments'] if e['predictions'][c[0]] and 'restore_frequency' not in e]}
            for c in classes]})
output={'reproduced':len(rows),'scope':'Saved initial paid prefixes; fixed runtime from trajectory before, identical forecast-classes policy; no task execution or truth lookup.',
 'weighting_note':'Class-weighted means describe a finite proposed set; they are not calibrated outcome probabilities. Both branch outcomes are reported.',
 'cases':rows}
(HERE/'initial_target_consequence_diagnosis.json').write_text(json.dumps(output,indent=2)+'\n')
for row in rows:
 print(row['case'],'chosen',row['selected'],'classes',row['forecast_class_count'])
 for e in row['experiments']:
  if e['is_reference'] and not any(b['remaining_classes'] and b['new_common_positive_targets_except_tested_point'] for b in e['branches']):continue
  print(' ',key(e),'signature gain',round(e['mean_target_signature_reduction_by_forecast_class'],4),'positive gain',round(e['mean_new_common_positive_targets_by_forecast_class'],4),'branches',[(b['outcome'],b['remaining_classes'],b['target_signature_reduction'],b['common_positive_targets_except_tested_point']) for b in e['branches']])
