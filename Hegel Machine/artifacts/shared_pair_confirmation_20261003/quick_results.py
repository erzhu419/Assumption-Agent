"""Read only completed small evaluation and accounting receipts."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
ARMS=('new_cumulative','prior_pair_queue','no_pair_cost_feedback')
KEYS=('observation_cost','measurement_cost','compute_cost')
def read(path):return json.loads(path.read_text())
results={}
for arm in ARMS:
    rows=[]; costs={key:0 for key in KEYS}; cpu=0.
    for directory in sorted((BASE/'successor_regression/runs'/arm).iterdir()):
        evaluation=read(directory/'evaluation.json')
        receipt=read(directory/'evaluate_execution.json')
        learning=read(directory/'learning_status.json')
        if not receipt['complete'] or not learning['complete']:raise ValueError('incomplete phase')
        cpu+=learning['research_cpu_seconds']+sum(row['cpu_seconds'] for row in evaluation)
        for key in KEYS:costs[key]+=learning['costs'][key]+sum(row['costs'][key] for row in evaluation)
        rows.extend({'root':directory.name,'position':row['position'],'mse':row['mse']} for row in evaluation)
    last=[row for row in rows if row['position']>=7]
    results[arm]={'tasks':len(rows),'passed':sum(row['mse']<=1e-6 for row in rows),
        'mse':sum(row['mse'] for row in rows)/len(rows),'last7_passed':sum(row['mse']<=1e-6 for row in last),
        'last7_mse':sum(row['mse'] for row in last)/len(last),'costs':costs,'all_cpu_seconds':cpu,'rows':rows}
base={ (row['root'],row['position']):row for row in results['prior_pair_queue']['rows']}
for arm in ('new_cumulative','no_pair_cost_feedback'):
    results[arm]['changed_scores_vs_prior']=[{'root':row['root'],'position':row['position'],
        'candidate_mse':row['mse'],'prior_mse':base[row['root'],row['position']]['mse']}
        for row in results[arm]['rows'] if row['mse']!=base[row['root'],row['position']]['mse']]
(BASE/'quick_terminal_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({arm:{key:value for key,value in result.items() if key!='rows'} for arm,result in results.items()},indent=2))
