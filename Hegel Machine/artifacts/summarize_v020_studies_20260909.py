"""Complete sequence changes and the actually funded research opportunities."""
from collections import Counter
import json
from pathlib import Path
from summarize_v016_studies_20260909 import ROOT, compare, run


def opportunities(session):
    visits = [row for row in session['episodes'] if row['scheduler'].get('research_opportunity')]
    records = []
    for episode in visits:
        decision = episode['scheduler']
        offer = decision['research_opportunity']
        feedback = decision.get('research_return', {})
        records.append({'task_index': episode['task_index'], 'task_id': episode['task_id'],
            'kind': offer['kind'], 'action': offer['action'], 'basis': offer['basis'],
            'expected_gain': offer['expected_gain'], 'actual_gain': feedback.get('gain'),
            'reference_rows': feedback.get('reference_rows'),
            'observation_cost': episode['observation_cost'], 'compute_cost': episode['compute_cost'],
            'quoted_observation_cost': offer['observation_cost'], 'quoted_compute_cost': offer['compute_cost']})
    return {'selected_kinds': dict(Counter(row['kind'] for row in records)),
            'outcomes': dict(Counter('unmeasured' if row['actual_gain'] is None else
                'positive' if row['actual_gain'] > 1e-9 else 'negative' if row['actual_gain'] < -1e-9 else 'zero'
                for row in records)),
            'revisit_observation_cost': sum(row['observation_cost'] for row in records),
            'revisit_compute_cost': sum(row['compute_cost'] for row in records),
            'records': records}


def main():
    report = {'experiment': 'v020_continuing_research', 'seeds': {}, 'method_selection_pairs': {}}
    for seed in (2027,2539,3181,4507):
        baseline = 'v020_baseline' if seed == 4507 else 'v019_release'
        report['seeds'][str(seed)] = {}
        for arm in ('methods','ordinary'):
            if seed == 2539 and arm == 'ordinary':
                continue  # no original 0.19 ordinary run for this seed
            before = run(seed, baseline, '20260909', arm)
            after = run(seed, 'v020_release', '20260909', arm)
            report['seeds'][str(seed)][arm] = {**compare(before,after), 'opportunities': opportunities(after[0])}
        report['method_selection_pairs'][str(seed)] = compare(
            run(seed,'v020_release','20260909','ordinary'),run(seed,'v020_release','20260909','methods'))
    path = ROOT / 'hegel_machine_v020_continuing_research_summary_20260909.json'
    with path.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    for seed, arms in report['seeds'].items():
        for arm, row in arms.items():
            print(json.dumps({'seed':seed,'arm':arm,
                'before': {key:row['before'][key] for key in ('mean_loss_by_objective','observation_cost','compute_cost')},
                'after': {key:row['after'][key] for key in ('mean_loss_by_objective','observation_cost','compute_cost')},
                'opportunities': {key:value for key,value in row['opportunities'].items() if key!='records'}}))
    print(path)


if __name__=='__main__':
    main()
