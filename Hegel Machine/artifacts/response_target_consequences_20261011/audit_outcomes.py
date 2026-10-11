"""Summarize frozen results and inspect saved-model consistency, without a world run."""
import json
import sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
read=lambda p:json.loads(p.read_text())
s=read(HERE/'summary.json');diag=[d for d in read(HERE/'diagnostic_brief.json') if d['phase']=='after']
bycase={d['case']:d for d in diag}
groups={}
for pair in s['paired']:
 changed=[d for d in bycase[pair['case']]['selections'] if d['changed']]
 group='nonlocal_changed' if any(d['rule'] in ('min_nonlocal_reduction','max_possible_reduction') for d in changed) else 'common_positive_changed' if changed else 'no_choice_changed'
 groups.setdefault(group,[]).append(pair)
summary_groups={name:{'cases':[p['case'] for p in rows], 'action_delta':sum(p['actions_difference'] for p in rows),'compute_delta':sum(p['compute_difference'] for p in rows),'C5_confirmation_delta':sum(int(p['confirmations_after']['5'])-int(p['confirmations_before']['5']) for p in rows)} for name,rows in groups.items()}
failed=[]
for d in diag:
 brief=read(HERE/'after'/d['case']/'frozen_numeric/brief.json')
 for selection in d['selections']:
  if selection['rule']=='common_positive' and not selection['physically_confirmed']:
   point=next(p for p in brief['point_confirmations'] if p['question_id']==selection['question_id']+'_physical')
   failed.append({'case':d['case'],'number':point['number'],'setting':point['prediction'],'status':point['question_status']})
case='cubic_shifted_2__gain_higher__adapted_law'
trace=read(HERE/'before'/case/'frozen_numeric/research_trace.json');state=trace['state'];models=state['frozen_response_knowledge']['models']
features={r['crystal']:{k:v['value'] for k,v in r['measurements'].items()} for r in state['research_table']['references']+state['research_table']['targets']}
forecasts={f'joint_response_{i}':(m,{n:Program.from_dict(m['program']).predict(Observation(values,0.)) for n,values in features.items()}) for i,m in enumerate(models)}
def predicts(identifier,number,setting):
 m,centers=forecasts[identifier];return m['lower']<=setting-centers[number]<=m['upper']
first=trace['comparisons'][0]['selected_experiment']
assert {i:predicts(i,5,2167.) for i in forecasts}==first['predictions'], 'Saved hypothesis identifiers do not align with the archive'
def remaining(cut):
 paid=[r for r in state['response_settings'] if min(r['evidence'])<=cut]
 return [i for i in forecasts if all(predicts(i,r['crystal'],r['frequency'])==r['activated'] for r in paid)]
before=remaining(36);after=remaining(37);supporters=[i for i in forecasts if predicts(i,5,2167.)]
interface={'case':case,'requested_setting':2167.,'positive_paid_observation':37,'positive_setting':2169.,
 'initial_supporters_of_requested_setting':supporters,'remaining_after_observation36':before,'remaining_after_observation37':after,
 'remaining_supporters_of_requested_setting':sorted(set(after)&set(supporters)),
 'reconstruction':'Exact saved fixed programs and intervals checked against paid responses through each cut; identifier-to-request forecast mapping asserted against the saved first comparison. This is not an independently saved intermediate answer snapshot or a new world execution.',
 'later_paid_responses':[r for r in state['response_settings'] if r['crystal']==5 and min(r['evidence'])>=37],
 'later_actions':[r for r in read(HERE/'before'/case/'frozen_numeric/action_trace.json') if r['index'] in (37,38,39,49,50,51)],
 'current_lines':{'answer_updates_remaining':'src/hegel_machine/cognition/world_reference_recalibration.py:352','saves_feedback':'src/hegel_machine/cognition/world_reference_recalibration.py:373','continues_requested_tuning':'src/hegel_machine/cognition/world_reference_recalibration.py:720','evaluates_each_observation':'src/hegel_machine/cognition/world_reference_recalibration.py:723','existing_explicit_supersession':'src/hegel_machine/cognition/world_reference_response.py:613'},
 'historical_entry':'artifacts/response_authored_control_20261009/RUN_STATUS.md',
 'next_step':'Connect already-paid intermediate model refutation to an actual parent redecision using the existing supersession interface. Preserve the unfinished requested-point verdict as inconclusive. Do not infer physical refutation of an unvisited endpoint or prefer2169 before observing it.'}
out={'completed':s['completed'],'overall':s['overall'],'strata':s['strata'],'pair_distribution':{k:dict(Counter('reduced' if p[k]<0 else 'increased' if p[k]>0 else 'same' for p in s['paired'])) for k in ('actions_difference','compute_difference')},'changed_rules':dict(Counter(x['rule'] for d in diag for x in d['selections'] if x['changed'])),'member_consequences':s['member_consequences']['after'],'groups':summary_groups,'unconfirmed_common_positive':failed,'interpretation':'Original1 two changed initial nonlocal tests only reorder the same endpoint tests, with no action or confirmation gain and122 extra compute. Later original3 higher decisions also change under nonlocal ordering; across all four nonlocal-changed cases actions increase6 and compute2399 without confirmation gain. Common-positive ordering supplies the limited physical savings; this is not an isolated causal ablation of the two rules. Withdraw the combined candidate.','next_interface':interface}
(HERE/'outcome_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'groups':summary_groups,'pair_distribution':out['pair_distribution'],'changed_rules':out['changed_rules'],'remaining_before37':before,'remaining_after37':after,'initial_supporters':supporters,'remaining_supporters':interface['remaining_supporters_of_requested_setting']},indent=2))
