"""Compact fixed-cohort comparisons, explicitly identifying reused old results."""
from collections import Counter
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / 'artifacts/research_investment_attribution_20261001/fresh_replication/summary.json'


def read(path):
    return json.loads(path.read_text())


def comparison(candidate, control, candidate_arm, control_arm):
    result = {'families': {}, 'cost_chain_counts': {}}
    for family in ('noise', 'condition'):
        n = candidate['totals'][candidate_arm]['families'][family]
        o = control['totals'][control_arm]['families'][family]
        counts = Counter()
        for root in sorted({r['root'] for r in candidate['tasks'] if r['arm'] == candidate_arm}):
            groups = [[r['mse'] for r in d['tasks'] if r['arm'] == arm and r['root'] == root and r['family'] == family]
                for d, arm in ((candidate, candidate_arm), (control, control_arm))]
            new, old = [sum(values) / len(values) for values in groups]
            tie = math.isclose(new, old, rel_tol=1e-9, abs_tol=1e-12 if family == 'condition' else 0.)
            counts['tie' if tie else 'win' if new < old else 'loss'] += 1
        result['families'][family] = {'mean_mse_change_percent': (n['mean_mse'] / o['mean_mse'] - 1) * 100,
            'predictive_mse_change_percent': (n['mean_predictive_mse'] / o['mean_predictive_mse'] - 1) * 100,
            'chain_counts': dict(counts)}
    n, o = candidate['totals'][candidate_arm], control['totals'][control_arm]
    result['all_cost_change_percent'] = {key: (n['all_costs'][key] / o['all_costs'][key] - 1) * 100
        for key in ('observation_cost', 'compute_cost')}
    for key in ('observation_cost', 'compute_cost'):
        counts = Counter()
        for row in candidate['streams']:
            if row['arm'] != candidate_arm:
                continue
            before = next(r for r in control['streams'] if r['arm'] == control_arm and r['root'] == row['root'])
            delta = row['learning_costs'][key] - before['learning_costs'][key]
            counts['tie' if delta == 0 else 'lower' if delta < 0 else 'higher'] += 1
        result['cost_chain_counts'][key] = dict(counts)
    return result


def main():
    seen, fresh, old = [read(path) for path in (BASE / 'seen_regression/summary.json',
        BASE / 'fresh_replication/summary.json', OLD)]
    result = {'seen_regression': {'totals': seen['totals'],
        'historical_reused_totals': {version: old['totals'][arm] for version, arm in
            (('0.156.0', 'baseline_156'), ('0.157.1', 'candidate'))},
        'historical_source': str(OLD), 'historical_cpu_comparison': False,
        'comparisons': {'versus_156': comparison(seen, old, 'candidate', 'baseline_156'),
            'versus_157_1': comparison(seen, old, 'candidate', 'candidate'),
            'versus_no_target_withdrawal': comparison(seen, seen, 'candidate', 'no_target_withdrawal')}},
        'fresh_replication': {'totals': fresh['totals'],
            'comparisons': {'versus_157_1': comparison(fresh, fresh, 'candidate', 'baseline_157_1')}},
        'interpretation': 'Seen regression and same-factory new-instance generalization are separate. Quality families are separate. CPU reflects scheduling; primary cumulative efficiency uses charged observations and compute.',
        'external_model_calls': 0}
    (BASE / 'compact_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
