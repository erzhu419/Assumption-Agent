"""Compare the fixed four complete sequences and actual decision research."""
from collections import Counter
import argparse
import json
from pathlib import Path
from summarize_v016_studies_20260909 import compare, run
from summarize_v020_studies_20260909 import opportunities

ROOT = Path(__file__).resolve().parent


def decision_research(session, evaluation):
    records = []
    names = {'retain_decision_question', 'execute_decision_question',
             'compare_decision_question', 'compare_decision_plan', 'retain_decision_plan'}
    for episode in session['episodes']:
        selected = [action for action in episode['actions'] if action['action'] in names]
        if selected:
            records.append({'task_index': episode['task_index'], 'task_id': episode['task_id'],
                'visit': episode['visit'], 'reason': episode['scheduler']['reason'],
                'expression': episode['expression'],
                'observation_cost': episode['observation_cost'], 'compute_cost': episode['compute_cost'],
                'actions': [{'action': action['action'],
                    'details': {key: value for key, value in action['details'].items()
                        if key not in ('policies', 'design', 'candidate_program',
                                       'rejected_program', 'retained_program')}} for action in selected]})
    actions = [action for row in records for action in row['actions']]
    short = [row for row in records if any(action['action'] == 'execute_decision_question'
                                         for action in row['actions'])]
    evaluated = {row['task_index']: row for row in evaluation['tasks']}
    for visit in short:
        result = evaluated[visit['task_index']]
        before = result['first']['decision']['mean_decision_loss']
        after = result['latest']['decision']['mean_decision_loss']
        visit['independent_decision'] = {'before': before, 'after': after, 'gain': before-after,
            'scope': 'fixed_two_visit_study_evaluated_after_both_learning_arms_stop'}
    return {'counts': dict(Counter(action['action'] for action in actions)),
        'short_comparison_adoptions': sum(bool(action['details'].get('adopted')) for action in actions
                                         if action['action'] == 'execute_decision_question'),
        'short_observation_cost': sum(row['observation_cost'] for row in short),
        'short_compute_cost': sum(row['compute_cost'] for row in short),
        'independent_short_outcomes': dict(Counter('better' if row['independent_decision']['gain'] > 1e-9
            else 'worse' if row['independent_decision']['gain'] < -1e-9 else 'tied' for row in short)),
        'records': records}


def decision_changes(before, after):
    result = []
    for left, right in zip(before['tasks'], after['tasks']):
        if 'decision' not in left['latest']:
            continue
        a, b = [row['latest']['decision']['mean_decision_loss'] for row in (left, right)]
        result.append({'task_index': left['task_index'], 'task_id': left['task_id'],
            'before': a, 'after': b, 'before_minus_after': a-b,
            'before_visits': left['visits'], 'after_visits': right['visits']})
    return result


def main(label='v021_release', output='hegel_machine_v021_decision_research_summary_20260909.json'):
    report = {'experiment': 'v021_complete_decision_research', 'code_label': label,
              'seeds': {}, 'method_selection_pairs': {}}
    for seed in (2027, 3181, 4507, 5501):
        baseline = 'v021_baseline' if seed == 5501 else 'v020_release'
        report['seeds'][str(seed)] = {}
        after_runs = {}
        for arm in ('methods', 'ordinary'):
            before, after = run(seed, baseline, '20260909', arm), run(seed, label, '20260909', arm)
            row = {**compare(before, after), 'opportunities': opportunities(after[0]),
                'decision_research': decision_research(after[0], after[1]),
                'decision_changes': decision_changes(before[1], after[1])}
            report['seeds'][str(seed)][arm] = row
            after_runs[arm] = after
            print(json.dumps({'seed': seed, 'arm': arm,
                'before': {key: row['before'][key] for key in (
                    'mean_loss_by_objective', 'mean_decision_loss', 'observation_cost', 'compute_cost')},
                'after': {key: row['after'][key] for key in (
                    'mean_loss_by_objective', 'mean_decision_loss', 'observation_cost', 'compute_cost')},
                'decision_research': {key: value for key, value in row['decision_research'].items()
                                      if key != 'records'}}), flush=True)
            del before
        report['method_selection_pairs'][str(seed)] = compare(after_runs['ordinary'], after_runs['methods'])
        del after_runs
    path = ROOT / output
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='v021_release')
    parser.add_argument('--output', default='hegel_machine_v021_decision_research_summary_20260909.json')
    args = parser.parse_args()
    main(args.label, args.output)
