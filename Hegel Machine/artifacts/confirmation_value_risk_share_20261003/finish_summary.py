"""Small paired outcomes and paid decision evidence; no learning or scoring."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
COHORTS = ('successor_regression', 'new_apparatus')
COSTS = ('observation_cost', 'measurement_cost', 'compute_cost')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def trim(event):
    keep_programs = event.get('action') in (
        'plan_condition_confirmation', 'validate_continued_program',
        'compare_method_followup_on_fresh_response', 'compare_successor_method_on_fresh_response')
    omit = ('reading_mean_program', 'competing_hypotheses', 'alternatives')
    if not keep_programs:
        omit += ('program', 'baseline_program')
    return {key: value for key, value in event.items() if key not in omit}


def counts(pairs):
    tally = dict(win=0, tie=0, loss=0)
    for new, old in pairs:
        tally['tie' if math.isclose(new, old, rel_tol=1e-9, abs_tol=1e-12)
              else 'win' if new < old else 'loss'] += 1
    return tally


def value_counts(events):
    known = [e['revision_value'] for e in events if e.get('revision_value') is not None]
    absent = [e for e in events if e.get('revision_value') is None]
    return {'recorded_decisions': len(events), 'applicable': sum(v.get('applicable', False) for v in known),
            'zero_revision_value': sum(v.get('applicable', False) and v.get('expected_mse_reduction') == 0 for v in known),
            'missing_revision_value_metadata': len(absent),
            'missing_metadata_decision_reasons': dict(Counter(e.get('reason', 'not_recorded') for e in absent)),
            'buy': sum(bool(e.get('buy')) for e in events), 'defer': sum(not e.get('buy') for e in events)}


def main():
    result = {'complete': True, 'external_model_calls': 0,
              'new_learning_chains': 36, 'new_terminal_task_arm_scores': 504,
              'reused_learning_chains': 12, 'reused_terminal_scores': 168}
    details = {}
    previous = read(BASE.parent / 'confirmation_value_20261003/compact_results.json')
    previous_failures = previous['successor_regression']['comparisons']['old_cumulative']['newly_failed']
    for cohort in COHORTS:
        protocol = read(BASE / cohort / 'protocol.json')
        summary = read(BASE / cohort / 'summary.json')
        totals = summary['totals']
        indexed = {arm: {(r['root'], r['position']): r for r in summary['rows'] if r['arm'] == arm}
                   for arm in totals}
        roots = sorted(protocol['roots'])
        for total in totals.values():
            total['all_cpu_seconds'] = total['learning_cpu_seconds'] + total['terminal_cpu_seconds']
        selected, reference = indexed['new_cumulative'], indexed['old_cumulative']
        pairs = [{'root': root, 'position': position, 'candidate_mse': row['mse'],
                  'reference_mse': reference[(root, position)]['mse']}
                 for (root, position), row in selected.items()]
        comparison = {'task_counts': counts((r['candidate_mse'], r['reference_mse']) for r in pairs),
                      'chain_counts': counts((sum(r['candidate_mse'] for r in pairs if r['root'] == root),
                                             sum(r['reference_mse'] for r in pairs if r['root'] == root)) for root in roots),
                      'newly_failed': [r for r in pairs if r['candidate_mse'] > 1e-6 >= r['reference_mse']],
                      'newly_solved': [r for r in pairs if r['reference_mse'] > 1e-6 >= r['candidate_mse']],
                      'cost_delta': {key: totals['new_cumulative']['all_costs'][key] - totals['old_cumulative']['all_costs'][key] for key in COSTS}}
        for label, predicate in (('first_round', lambda p: p < 7), ('successors', lambda p: p >= 7)):
            subset = [r for r in pairs if predicate(r['position'])]
            comparison[label] = {'tasks': len(subset),
                'candidate_mse': sum(r['candidate_mse'] for r in subset) / len(subset),
                'reference_mse': sum(r['reference_mse'] for r in subset) / len(subset),
                'candidate_passed': sum(r['candidate_mse'] <= 1e-6 for r in subset),
                'reference_passed': sum(r['reference_mse'] <= 1e-6 for r in subset),
                'task_counts': counts((r['candidate_mse'], r['reference_mse']) for r in subset)}
        if cohort == 'successor_regression':
            comparison['prior_0175_failures'] = {'total': len(previous_failures),
                'now_passed': sum(selected[(r['root'], r['position'])]['mse'] <= 1e-6 for r in previous_failures),
                'remaining': [{**r, 'corrected_mse': selected[(r['root'], r['position'])]['mse']}
                    for r in previous_failures if selected[(r['root'], r['position'])]['mse'] > 1e-6]}
        evidence = {}
        detail = {'task_pairs': pairs, 'recorded_decisions': [], 'program_revisions': [], 'resource_use': []}
        for arm, settings in protocol['arms'].items():
            if settings['result_source'] != 'new run':
                continue
            decisions, revisions, caches, confirmation = [], [], [], []
            for root in roots:
                state = read(BASE / cohort / 'runs' / arm / root / 'research_evidence.json.gz')
                detail['resource_use'].append({'arm': arm, 'root': root, 'costs': state['costs'],
                    'remaining_pool': {key: protocol['limits']['chain_' + key.removesuffix('_cost')] - state['costs'][key] for key in COSTS},
                    'screening_compute_cost': state['screening_compute_cost'], 'revisits': len(state['revisits'])})
                for row in state['results']:
                    position = row['task_position']
                    for event in row['research']['actions']:
                        entry = {'arm': arm, 'root': root, 'position': position, **event}
                        if event['action'] == 'choose_condition_research_investment':
                            decisions.append(entry)
                        if event['action'] in ('revise_competing_observed_conditions', 'revise_condition_on_paid_response'):
                            revisions.append(entry)
                caches.extend({'root': root, **q} for q in state.get('condition_revisit_quotes', []) if q['screen_exists'])
                confirmation.extend({'root': root, **q} for q in state.get('posthoc_confirmation_diagnostics', []))
            evidence[arm] = {'revision_value': value_counts(decisions),
                'actual_program_revisions': sum(bool(e.get('changed_program')) for e in revisions),
                'final_cached_screens': {'count': len(caches),
                    'applicable': sum(bool(q.get('revision_value') and q['revision_value'].get('applicable')) for q in caches)},
                'confirmation': {'recorded_batches': len(confirmation),
                    'actual_discriminating_batches': sum(c.get('discriminating_rows', 0) > 0 for c in confirmation),
                    'positive_gain_batches': sum(c.get('gain_positive', False) for c in confirmation),
                    'negative_gain_batches': sum(c.get('gain_negative', False) for c in confirmation)}}
            detail['recorded_decisions'].extend(decisions)
            detail['program_revisions'].extend(revisions)
        result[cohort] = {'totals': totals, 'comparison': comparison, 'paid_evidence': evidence,
                         'scope': protocol['scope'],
                         'evidence_scope': 'Only recorded normal-research decisions are counted. Missing prospective metadata is not proof of no applicable revision; final screen caches are not historical quote counts.'}
        details[cohort] = detail
    save(BASE / 'compact_results.json', result)
    save(BASE / 'paired_details.json', details)
    print(json.dumps({name: {'candidate': result[name]['totals']['new_cumulative']['solved'],
                            'reference': result[name]['totals']['old_cumulative']['solved'],
                            'comparison': result[name]['comparison']} for name in COHORTS}, indent=2))


if __name__ == '__main__':
    main()
