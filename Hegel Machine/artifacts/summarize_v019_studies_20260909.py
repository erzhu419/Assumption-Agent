"""Joint research revision: complete task outcomes, costs and executed mechanisms."""
from collections import Counter
import json
from summarize_v016_studies_20260909 import ROOT, compare, run


def mechanisms(session):
    wanted = {'retain_history_alternatives', 'revise_research_method', 'defer_source_refit',
              'retain_research_question', 'execute_research_question', 'measure_experiment_coverage'}
    events = []
    for episode in session['episodes']:
        for action in episode['actions']:
            if action['action'] in wanted:
                events.append({'task_index': episode['task_index'], 'task_id': episode['task_id'],
                    'visit': episode['visit'], 'round': action['round'], 'action': action['action'],
                    'observation_cost': action['observation_cost'], 'compute_cost': action['compute_cost'],
                    'details': action['details']})
    coverage = [event['details'] for event in events if event['action'] == 'measure_experiment_coverage']
    revisits = [episode for episode in session['episodes']
                if episode['scheduler']['reason'] == 'unresolved_distinguishing_experiment']
    return {'actions': dict(Counter(event['action'] for event in events)),
            'boundary_preparations': {'attempted': sum(item['attempted_boundaries'] for item in coverage),
                                     'reached': sum(item['reached_boundaries'] for item in coverage)},
            'experiment_revisits': {'count': len(revisits),
                'observation_cost': sum(row['observation_cost'] for row in revisits),
                'compute_cost': sum(row['compute_cost'] for row in revisits)},
            'events': events}


def main():
    result = {'experiment': 'v019_joint_research_capabilities', 'seeds': {}}
    for seed in (1237, 2027, 2539, 3181):
        baseline = 'v019_baseline' if seed == 3181 else 'v018_mainline'
        arms = ('methods', 'ordinary') if seed in (1237, 3181) else ('methods',)
        result['seeds'][str(seed)] = {}
        for arm in arms:
            before = run(seed, baseline, '20260909', arm)
            after = run(seed, 'v019_release', '20260909', arm)
            result['seeds'][str(seed)][arm] = {**compare(before, after), 'mechanisms': mechanisms(after[0])}
    result['method_selection_pairs'] = {}
    for seed in (1237, 2027, 3181):
        result['method_selection_pairs'][str(seed)] = compare(
            run(seed, 'v019_release', '20260909', 'ordinary'),
            run(seed, 'v019_release', '20260909', 'methods'))
    result['saved_experiment_ablation_seed1237'] = compare(
        run(1237, 'v019_without_experiment_frontier', '20260909', 'methods'),
        run(1237, 'v019_release', '20260909', 'methods'))
    path = ROOT / 'hegel_machine_v019_joint_research_summary_20260909.json'
    with path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for seed, arms in result['seeds'].items():
        for arm, row in arms.items():
            print(json.dumps({'seed': seed, 'arm': arm,
                'before': {key: row['before'][key] for key in ('mean_loss_by_objective','observation_cost','compute_cost')},
                'after': {key: row['after'][key] for key in ('mean_loss_by_objective','observation_cost','compute_cost')},
                'mechanisms': {key: value for key,value in row['mechanisms'].items() if key != 'events'}}))
    print(path)


if __name__ == '__main__':
    main()
