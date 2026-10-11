"""Combine selected-policy development and fixed extension; keep open admission separate."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text())
extension=read(BASE/'development_summary.json')
development=read(BASE.parent/'research_portfolio_admission_20261003/development_summary.json')
fields=('executed_visits','program_changed_visits','fresh_pairs','positive_fresh_pairs',
        'negative_fresh_pairs','selected_local_buy_false','portfolio_decisions',
        'feedback_queue_refreshes','tasks','passed','last7_passed',
        'suffix_cpu_seconds','terminal_cpu_seconds','measured_cpu_seconds')
totals={}
for arm,source_arm in [('baseline','baseline'),('candidate','portfolio_with_local_admission')]:
    a,b=development['arms'][source_arm],extension['arms'][arm]
    total={key:a[key]+b[key] for key in fields}
    for key in ('suffix_costs','all_learning_costs','terminal_costs','all_costs'):
        total[key]={cost:a[key][cost]+b[key][cost] for cost in a[key]}
    total['mse']=(a['mse']*a['tasks']+b['mse']*b['tasks'])/total['tasks']
    total['last7_mse']=(a['last7_mse']*21+b['last7_mse']*63)/84
    totals[arm]=total
pairs=[{'root':r['root'],'position':r['position'],'candidate_mse':r['admission_mse'],
        'baseline_mse':r['reference_mse']} for r in development['admission_pairs'] if r['reference']=='baseline']+extension['pairs']
def comparison(rows):
    return {'tasks':len(rows),'wins':sum(r['candidate_mse']<r['baseline_mse']-1e-12 for r in rows),
        'ties':sum(abs(r['candidate_mse']-r['baseline_mse'])<=1e-12 for r in rows),
        'losses':sum(r['candidate_mse']>r['baseline_mse']+1e-12 for r in rows),
        'newly_failed':[r for r in rows if r['candidate_mse']>1e-6>=r['baseline_mse']],
        'newly_solved':[r for r in rows if r['baseline_mse']>1e-6>=r['candidate_mse']]}
result={'complete':True,'external_model_calls':0,'selected_pair_continuations':24,
    'selected_pair_terminal_scores':336,'unrestricted_development_continuations':3,
    'unrestricted_development_terminal_scores':42,'selected_pair':totals,
    'development':comparison(pairs[:42]),'fixed_extension':comparison(pairs[42:]),
    'combined':comparison(pairs),'pairs':pairs,
    'unrestricted_development':development['arms']['candidate'],
    'runtime_scope':'All selected candidate runs use exactly0.176.0/admission-ablation frozen source; active0.176.1 cleanup is separately tested and was not swapped into an ongoing experiment.',
    'cohort_scope':'Roots01–03 informed admission selection. Roots04–12 were fixed before extension and remained unchanged, but are historically seen tasks. Complete true0.168 paid visit14 states are common to paired arms.',
    'cost_scope':'Full resources include the common paid prefix once per root and arm, continuation, and terminal scoring. Measured CPU includes this study continuation and scoring; no previously paid prefix CPU was rerun.'}
(BASE/'combined_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'totals':totals,'development':result['development'],
                  'fixed_extension':result['fixed_extension'],'combined':result['combined']},indent=2))
