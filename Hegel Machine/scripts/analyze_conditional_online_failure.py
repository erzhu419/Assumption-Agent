#!/usr/bin/env python3
"""Post-hoc diagnostic of the completed online study, using already paid rows only.

The compared feature maps are host-informed. Their retrospective fit is not Luna
discovery, a new confirmation, or transfer evidence. Nothing is written to the
research workspace or passed to its author.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


def mse(actual, predicted):
    return float(np.mean((actual - predicted) ** 2))


def features(rows, threshold, *, form):
    result = []
    for row in rows:
        seq = [*row['history'], row['features']][-3:]
        if form in {'padded_polynomial', 'polynomial_with_count'}:
            padded = [{'x': 0., 'z': 0., 'c': 0.}] * (3 - len(seq)) + seq
            result.append([point[key] for point in padded for key in ('x', 'z', 'c')])
        else:
            branches = np.zeros((2, 3))
            for point in seq:
                condition = point['c'] if form == 'gate_inside_window' else row['features']['c']
                branches[int(condition < threshold)] += [1., point['x'], point['z'] ** 2]
            result.append(branches.ravel())
    matrix = np.asarray(result)
    if form in {'padded_polynomial', 'polynomial_with_count'}:
        matrix = PolynomialFeatures(2, include_bias=True).fit_transform(matrix)
        if form == 'polynomial_with_count':
            matrix = np.column_stack([matrix, [min(3, len(r['history']) + 1) for r in rows]])
        low = np.asarray([row['features']['c'] < threshold for row in rows])
        matrix = np.column_stack([matrix * (~low[:, None]), matrix * low[:, None]])
    return matrix


def fit_predict(matrix, targets, train, test, *, ridge):
    if ridge:
        # Two independent branch intercepts reproduce the author's call91 fit.
        half = matrix.shape[1] // 2
        predicted = np.zeros(int(np.sum(test)))
        for start in (0, half):
            part = matrix[:, start:start + half]
            branch = np.any(part != 0., axis=1)
            fit = train & branch
            selected = branch[test]
            if not np.any(selected):
                continue
            model = Ridge(alpha=1e-6).fit(part[fit], targets[fit])
            predicted[selected] = model.predict(part[test][selected])
        return predicted
    weights = np.linalg.lstsq(matrix[train], targets[train], rcond=None)[0]
    return matrix[test] @ weights


def diagnose(rows, groups, threshold, form):
    y = np.asarray([r['target'] for r in rows])
    matrix = features(rows, threshold, form=form)
    ridge = form in {'padded_polynomial', 'polynomial_with_count'}
    all_rows = np.ones(len(rows), dtype=bool)
    pooled = fit_predict(matrix, y, all_rows, all_rows, ridge=ridge)
    held_out = np.empty(len(rows))
    group_errors = []
    for group in np.unique(groups):
        test = groups == group
        held_out[test] = fit_predict(matrix, y, ~test, test, ridge=ridge)
        group_errors.append(mse(y[test], held_out[test]))
    result = {'form': form, 'threshold': threshold, 'rows': len(rows),
        'trajectories': len(group_errors), 'columns': matrix.shape[1],
        'pooled_mse': mse(y, pooled), 'paid_trajectory_holdout_row_mse': mse(y, held_out),
        'paid_trajectory_holdout_episode_mean_mse': float(np.mean(group_errors)),
        'paid_trajectory_holdout_worst_mse': max(group_errors)}
    if len(rows) > 1600:
        last = np.arange(len(rows)) >= 1600
        result['fit_first_1600_score_last_64_paid_mse'] = mse(
            y[last], fit_predict(matrix, y, ~last, last, ridge=ridge))
    return result


def aliasing(rows):
    # Group identical public feature vectors produced by the author's padding.
    grouped = {}
    for index, row in enumerate(rows):
        seq = [*row['history'], row['features']][-3:]
        padded = [{'x': 0., 'z': 0., 'c': 0.}] * (3 - len(seq)) + seq
        key = tuple(point[name] for point in padded for name in ('x', 'z', 'c'))
        grouped.setdefault(key, []).append((index, row['target']))
    contradictions = [items for items in grouped.values()
                      if max(y for _, y in items) - min(y for _, y in items) > 1e-10]
    irreducible = sum(sum((y - np.mean([v for _, v in items])) ** 2 for _, y in items)
                      for items in grouped.values()) / len(rows)
    return {'conflicting_identical_feature_groups': len(contradictions),
        'minimum_possible_paid_mse_for_this_feature_map': float(irreducible),
        'examples': [{'row_indices': [i for i, _ in items[:8]],
                      'targets': [y for _, y in items[:8]],
                      'history_lengths': [len(rows[i]['history']) for i, _ in items[:8]]}
                     for items in contradictions[:3]]}


def load_paid_trace(path):
    """Reconstruct reset histories from the small published paid-trajectory extract."""
    trace = json.loads(Path(path).read_text())
    training = trace['research_record']
    training['paid_observations'] = []
    for trajectory in trace['trajectories']:
        sequence = trajectory['sequence']
        training['paid_observations'].append({'rows': [
            {'features': point, 'target': target, 'history': sequence[:index]}
            for index, (point, target) in enumerate(zip(sequence, trajectory['targets']))]})
    return trace['source_study'], training


def analyze(study=None, *, paid_trace=None):
    if paid_trace is None:
        workspace = Path(study) / 'online/design_case_006/hegel'
        training = json.loads((workspace / 'training.json').read_text())
    else:
        study, training = load_paid_trace(paid_trace)
    records = training['paid_observations']
    rows = [row for record in records for row in record['rows']]
    groups = np.asarray([index for index, record in enumerate(records) for _ in record['rows']])
    events = training['events'][5:]
    usage = [e.get('usage', {}) for e in events]
    tests = [(-.175, 'padded_polynomial'), (-.176125, 'padded_polynomial'),
             (-.175, 'polynomial_with_count'),
             (-.176125, 'polynomial_with_count'), (-.176125, 'gate_inside_window'),
             (-.176125, 'gate_outside_window')]
    return {'source_study': str(Path(study).resolve()),
        'scope': 'Host-informed retrospective diagnostic on paid source rows only. '
            'Threshold interval (-0.17625,-0.176) is already in Luna paid evidence; '
            'window=3 and branch placement are examined using the laboratory definition. '
            'No new labels, no target data, no model publication, no changes to Luna files.',
        'costs': training['costs'], 'additional_costs': training['continuation_cost'],
        'action_counts': dict(Counter(e['action'] for e in events)),
        'successful_queries': sum(e['action'] == 'query' and not e.get('feedback', {}).get('error')
                                  for e in events),
        'new_confirmation_calls': [i + 1 for i, e in enumerate(events)
                                    if e.get('feedback', {}).get('fresh_confirmation')],
        'usage': {'input_tokens': sum(u.get('input_tokens', 0) for u in usage),
                  'output_tokens': sum(u.get('output_tokens', 0) for u in usage),
                  'cached_input_tokens': sum(u.get('cached_input_tokens', 0) for u in usage)},
        'structure': {'questions': len(training['online_questions']),
            'child_questions': sum(q.get('parent_id') is not None for q in training['online_questions']),
            'comparisons_including_inherited': len(training['experiment_comparisons']),
            'revisions_including_inherited': len(training['revision_proposals']),
            'conditional_models': len(training['conditional_models']),
            'observable_models': len(training['observable_models'])},
        'threshold_from_paid_evidence': {'lower': -.17625, 'upper': -.176, 'midpoint': -.176125},
        'padding_aliasing': aliasing(rows),
        'same_paid_1600_rows_as_call91': [diagnose(rows[:1600], groups[:1600], t, f) for t, f in tests],
        'all_paid_1664_rows': [diagnose(rows, groups, t, f) for t, f in tests],
        'limitations': 'The form and feature corrections are host-informed and chosen after seeing '
            'the run. These are diagnostic comparisons, not independent performance or autonomous '
            'discovery. A threshold interval fitting these rows still leaves unmeasured boundary '
            'ambiguity; no future-confirmation guarantee follows.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--study', type=Path)
    source.add_argument('--paid-trace', type=Path,
                        help='Small published source trajectories and research-account extract')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study, paid_trace=args.paid_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'padding_aliasing': result['padding_aliasing'],
                      'comparisons': result['same_paid_1600_rows_as_call91']}, indent=2))


if __name__ == '__main__':
    main()
