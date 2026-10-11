"""Preserve development and extension outcomes, then add the fixed twelve roots."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
DEVELOPMENT=BASE.parent/'portfolio_companion_20261004'

def read(path):
    return json.loads(path.read_text())

cohorts={'development':read(DEVELOPMENT/'compact_results.json'),
         'fixed_extension':read(BASE/'compact_results.json')}
scalar_sums=('tasks','passed','last7_passed','suffix_cpu_seconds','terminal_cpu_seconds',
    'measured_cpu_seconds','executed_visits','program_changed_visits','fresh_pairs',
    'positive_fresh_pairs','negative_fresh_pairs','selected_local_buy_false',
    'selected_predicted_loss_reduction','portfolio_decisions','feedback_queue_refreshes')
resources=('observation_cost','measurement_cost','compute_cost')
combined={}
for arm in ('baseline','candidate'):
    records=[cohort['arms'][arm] for cohort in cohorts.values()]
    metrics=[record['metrics'] for record in records]
    total={key:sum(item[key] for item in metrics) for key in scalar_sums}
    total['mse']=sum(item['mse']*item['tasks'] for item in metrics)/total['tasks']
    total['last7_mse']=sum(item['last7_mse']*item['tasks'] for item in metrics)/total['tasks']
    for group in ('suffix_costs','all_learning_costs','terminal_costs','all_costs'):
        total[group]={key:sum(item[group][key] for item in metrics) for key in resources}
    combined[arm]={'metrics':total,
        'actual_restored_companion_purchase_count':sum(item['actual_restored_companion_purchase_count'] for item in records),
        'restored_purchase_costs':{key:sum(item['restored_purchase_costs'][key] for item in records) for key in resources},
        'restored_purchase_program_changes':sum(item['restored_purchase_program_changes'] for item in records),
        'restored_purchase_positive_fresh_pairs':sum(item['restored_purchase_positive_fresh_pairs'] for item in records)}
pairs=[pair for cohort in cohorts.values() for pair in cohort['terminal_pairs']]
roots={}
for root in sorted({p['root'] for p in pairs}):
    selected=[p for p in pairs if p['root']==root]
    before=sum(p['baseline_mse'] for p in selected)/len(selected)
    after=sum(p['candidate_mse'] for p in selected)/len(selected)
    roots[root]={'baseline_passed':sum(p['baseline_mse']<=1e-6 for p in selected),
        'candidate_passed':sum(p['candidate_mse']<=1e-6 for p in selected),
        'baseline_mse':before,'candidate_mse':after,
        'mse_outcome':'win' if after<before-1e-7 else 'loss' if after>before+1e-7 else 'tie'}
old_targets=[('root04',4),('root07',5),('root07',8),('root09',5),('root09',10),('root12',9)]
old_status=[]
for root,position in old_targets:
    pair=next(p for p in pairs if p['root']==root and p['position']==position)
    old_status.append({**pair,'candidate_passed':pair['candidate_mse']<=1e-6,
        'current_baseline_passed':pair['baseline_mse']<=1e-6})
a,c=combined['baseline']['metrics'],combined['candidate']['metrics']
result={'complete':True,'candidate_version':'0.178.0','baseline_version':'0.177.0',
    'new_continuations':24,'terminal_scores':336,'external_model_calls':0,
    'scope':'Two prespecified development roots plus fixed remaining ten-root extension. All are historically seen, identical real visit14 paid prefixes and terminal4x64/offset1000003; no source or parameter changes between cohorts.',
    'cohorts':cohorts,'combined':combined,'root_results':roots,
    'task_wtl':{'win':sum(p['candidate_mse']<p['baseline_mse']-1e-7 for p in pairs),
        'tie':sum(abs(p['candidate_mse']-p['baseline_mse'])<=1e-7 for p in pairs),
        'loss':sum(p['candidate_mse']>p['baseline_mse']+1e-7 for p in pairs)},
    'chain_mse_wtl':{key:sum(r['mse_outcome']==key for r in roots.values()) for key in ('win','tie','loss')},
    'candidate_minus_baseline':{'passed':c['passed']-a['passed'],'mse':c['mse']-a['mse'],
        'all_costs':{key:c['all_costs'][key]-a['all_costs'][key] for key in resources},
        'measured_cpu_seconds':c['measured_cpu_seconds']-a['measured_cpu_seconds']},
    'old_six_failure_positions':old_status,
    'new_failures':[p for p in pairs if p['baseline_mse']<=1e-6<p['candidate_mse']],
    'recoveries':[p for p in pairs if p['candidate_mse']<=1e-6<p['baseline_mse']],
    'pairs':pairs,
    'paid_mechanism_diagnosis':'artifacts/portfolio_companion_20261004/paid_companion_diagnosis.json',
    'conclusion':'The fixed extension is quality-neutral with extra research cost. Combined improvement is confined to the two development roots: two recoveries, no new failures, and four of the previous six failures unresolved. Selected companions are actually retained and purchased, but fresh confirmation remains non-discriminating in the restored visits.'}
(BASE/'combined_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print({key:result[key] for key in ('task_wtl','chain_mse_wtl','candidate_minus_baseline','old_six_failure_positions','new_failures','recoveries')})
