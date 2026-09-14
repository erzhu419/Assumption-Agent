"""Summarize fixed-version paired runs without pooling distinct loss objectives."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def run(seed, label, date, arm):
    prefix = f'hegel_machine_{label}_seed{seed}'
    session = read(f'{prefix}_{arm}_{date}.json')
    evaluation = read(f'{prefix}_{arm}_evaluation_{date}.json')
    return session, evaluation


def compact(session, evaluation):
    returns = [record for episode in session['episodes'] for action in episode['actions']
               if action['action'] == 'knowledge_return' for record in action['details']['returns']]
    return {
        'visited_tasks': session['visited_tasks'], 'completed_visits': session['completed_visits'],
        'observation_cost': session['total_observation_cost'], 'compute_cost': session['total_compute_cost'],
        'mean_loss_by_objective': {kind: mean(task['latest']['loss'] for task in evaluation['tasks']
                                             if task['latest']['loss_kind'] == kind)
                                   for kind in sorted({task['latest']['loss_kind'] for task in evaluation['tasks']})},
        'mean_decision_loss': mean(task['latest']['decision']['mean_decision_loss']
                                  for task in evaluation['tasks'] if 'decision' in task['latest']),
        'method_overrides': [{'task_index': ep['task_index'], 'task_id': ep['task_id'],
                              'visit': ep['visit'], 'round': action['round'],
                              'method': action['details']['learned_method']}
                             for ep in session['episodes'] for action in ep['actions']
                             if action['details'].get('learned_method', {}).get('changes_action')],
        'simplification_actions': sum(action['action'] == 'simplify_equivalent'
                                      for ep in session['episodes'] for action in ep['actions']),
        'knowledge_return_attempts': {
            'total': len(returns), 'evidence_status': dict(Counter(r['evidence_status'] for r in returns)),
            'retained': sum(r['retained'] for r in returns),
            'positive_retained': sum(r['retained_normalized_gain'] > 0 for r in returns),
            'negative': sum(r['retained_normalized_gain'] < 0 for r in returns),
            'interpretation': 'Source/template/binding attempts; shared program work can appear under several origins.'},
    }


def compare(before, after):
    old, old_eval = before
    new, new_eval = after
    rows = []
    for left, right in zip(old_eval['tasks'], new_eval['tasks']):
        assert (left['task_index'], left['task_id']) == (right['task_index'], right['task_id'])
        index = left['task_index']
        before_eps = [ep for ep in old['episodes'] if ep['task_index'] == index]
        after_eps = [ep for ep in new['episodes'] if ep['task_index'] == index]
        gain = left['latest']['loss'] - right['latest']['loss']
        row = {'task_index': index, 'task_id': left['task_id'], 'loss_kind': left['latest']['loss_kind'],
               'before_loss': left['latest']['loss'], 'after_loss': right['latest']['loss'],
               'before_minus_after_loss': gain,
               'material_outcome': 'after_better' if gain > .002 else 'before_better' if gain < -.002 else 'tied',
               'before_observations': sum(ep['observation_cost'] for ep in before_eps),
               'after_observations': sum(ep['observation_cost'] for ep in after_eps),
               'before_compute': sum(ep['compute_cost'] for ep in before_eps),
               'after_compute': sum(ep['compute_cost'] for ep in after_eps),
               'before_visits': len(before_eps), 'after_visits': len(after_eps),
               'before_expression': before_eps[-1]['expression'], 'after_expression': after_eps[-1]['expression']}
        if any((abs(gain) > 1e-9, row['before_observations'] != row['after_observations'],
                row['before_compute'] != row['after_compute'], row['before_expression'] != row['after_expression'])):
            rows.append(row)
    return {'before': compact(old, old_eval), 'after': compact(new, new_eval), 'changed_tasks': rows,
            'material_outcomes_all_65': dict(Counter(
                'after_better' if a['latest']['loss'] - b['latest']['loss'] > .002 else
                'before_better' if a['latest']['loss'] - b['latest']['loss'] < -.002 else 'tied'
                for a,b in zip(old_eval['tasks'], new_eval['tasks'])))}


if __name__ == '__main__':
    summary = {'experiment': 'v016_fixed_run_comparisons', 'seeds': {}}
    for seed, baseline in [(1237, 'v015_mainline'), (1613, 'v016_baseline')]:
        summary['seeds'][str(seed)] = {arm: compare(run(seed, baseline, '20260908', arm),
                                                   run(seed, 'v016_mainline', '20260909', arm))
                                      for arm in ('methods', 'ordinary')}
    summary['return_ordering_ablation_seed1237'] = compare(
        run(1237, 'v016_without_return_ordering', '20260909', 'methods'),
        run(1237, 'v016_mainline', '20260909', 'methods'))
    output = ROOT / 'hegel_machine_v016_knowledge_value_summary_20260909.json'
    with output.open('x') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(output)
