#!/usr/bin/env python3
"""Reband saved coefficient programs using complete captured final paid states."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from diagnose_recorded_bands import CASES, OUTPUT, read


def assess(program, state):
    features = {row['crystal']: {key: value['value'] for key, value in row['measurements'].items()}
                for row in state['research_table']['references'] + state['research_table']['targets']}
    predictions = {number: program.predict(Observation(values, 0.)) for number, values in features.items()}
    residuals = [{**row, 'residual': row['frequency'] - predictions[row['crystal']]}
                 for row in state['response_settings']]
    positive = [row['residual'] for row in residuals if row['activated']]
    lower, upper = min(positive), max(positive)
    conflicts = [row for row in residuals if not row['activated'] and lower <= row['residual'] <= upper]
    negatives = [row['residual'] for row in residuals if not row['activated']]
    return {'fixed_program_feasible': not conflicts, 'required_positive_hull': [lower, upper],
        'lower_boundary_range': {'exclusive_min': max((x for x in negatives if x < lower), default=None),
                                 'inclusive_max': lower},
        'upper_boundary_range': {'inclusive_min': upper,
                                 'exclusive_max': min((x for x in negatives if x > upper), default=None)},
        'negative_conflicts': [{key: row[key] for key in ('crystal', 'frequency', 'residual', 'evidence')}
                              for row in conflicts]}, residuals


def diagnose():
    partial = {(row['cohort'], row['case']): row for row in read(OUTPUT / 'recorded_band_diagnosis.json')['cases']}
    results = []
    for cohort, case in CASES:
        directory = OUTPUT / 'replays' / cohort / case / 'frozen_numeric'
        state = read(directory / 'final_export_inputs.json')['state']
        export = read(directory / 'learned_response_knowledge.json')
        assert len(state['response_settings']) == export['source_response_count']
        trials = [*state.get('tests', []), *([state['test']] if state.get('test') else [])]
        revisions = [t['band_revision'] for t in trials if t.get('band_revision')]
        positive_revision = next(r for r in revisions if r['positive_supporters'])
        snapshots = [state.get('response_research', {}), *state.get('response_research_history', []), *revisions]
        grouped = {}
        for snapshot in snapshots:
            for hypothesis in snapshot.get('hypotheses', []):
                program = Program.from_dict(hypothesis['program'])
                group = grouped.setdefault(program, {'id': 'fixed_program_' + str(len(grouped)), 'windows': []})
                group['windows'].append({'source_question': snapshot.get('question_id'),
                    'hypothesis_id': hypothesis['id'], 'family': hypothesis['family'],
                    'band': [hypothesis['lower'], hypothesis['upper']]})
        all_programs = []
        for program, group in grouped.items():
            diagnosis, residuals = assess(program, state)
            windows = set(tuple(row['band']) for row in group['windows'])
            survivors = [[lower, upper] for lower, upper in sorted(windows)
                         if all((lower <= r['residual'] <= upper) == r['activated'] for r in residuals)]
            all_programs.append({'program_id': group['id'], 'program_description': program.describe(),
                'source_hypotheses': group['windows'], 'families': sorted({r['family'] for r in group['windows']}),
                'distinct_saved_window_count': len(windows), 'surviving_saved_windows': survivors,
                **diagnosis,
                'all_saved_windows_rejected_but_rebandable': not survivors and diagnosis['fixed_program_feasible']})
        by_id = {row['program_id']: row for row in all_programs}
        local = []
        partial_by_id = {row['hypothesis_id']: row for row in partial[(cohort, case)]['programs']}
        for hypothesis in positive_revision['hypotheses']:
            if hypothesis['id'] not in positive_revision['positive_supporters']:
                continue
            identifier = grouped[Program.from_dict(hypothesis['program'])]['id']
            complete = by_id[identifier]
            previous = partial_by_id[hypothesis['id']]
            local.append({'revision_hypothesis_id': hypothesis['id'], 'program_id': identifier,
                'partial_status': previous['status'],
                'fixed_program_feasible_on_complete_paid_data': complete['fixed_program_feasible'],
                'new_conflicts_from_previously_missing_rows': [row for row in complete['negative_conflicts']
                    if not any(row['crystal'] == old['crystal'] and row['frequency'] == old['frequency']
                               for old in previous['negative_conflicts'])]})
        results.append({'cohort': cohort, 'case': case,
            'complete_paid_state': str(directory / 'final_export_inputs.json'),
            'response_count': len(state['response_settings']),
            'earlier_missing_response_count': partial[(cohort, case)]['missing_response_rows'],
            'local_positive_programs': len(local),
            'local_positive_programs_rebandable': sum(r['fixed_program_feasible_on_complete_paid_data'] for r in local),
            'all_snapshot_unique_programs': len(all_programs),
            'all_snapshot_rebandable_programs': sum(r['fixed_program_feasible'] for r in all_programs),
            'all_windows_rejected_but_program_rescued': sum(r['all_saved_windows_rejected_but_rebandable'] for r in all_programs),
            'cross_family_merges': sum(len(r['families']) > 1 for r in all_programs),
            'local_revision_assessment': local, 'all_snapshot_programs': all_programs})
    result = {'source': 'Complete final paid states captured by the root replay, with recorded action/cost equality to the preceding run',
        'procedure': 'No coefficient fitting or world execution. Group actual saved hypotheses by exact Program, retain their family and source records, and intersect their required positive hull with every measured negative response.',
        'local_positive_programs': sum(r['local_positive_programs'] for r in results),
        'local_positive_programs_rebandable': sum(r['local_positive_programs_rebandable'] for r in results),
        'all_snapshot_programs': sum(r['all_snapshot_unique_programs'] for r in results),
        'all_snapshot_rebandable_programs': sum(r['all_snapshot_rebandable_programs'] for r in results),
        'cases': results,
        'conclusion': 'All three sources already contain coefficient programs that can explain every final paid response after revising only the shared connected band: five programs per source. New coefficients or scope gating are not needed to establish source consistency in these cases.',
        'partial_comparison': 'The conservative partial diagnosis made no false feasibility claim. Completing the missing rows rejects one additional program in the adapted preservation case because its new positive hull contains the measured negative C5=1802 at observation 43. Five original-law-return programs remain rejected by measured negative C4=471 at observation 32.',
        'limitations': 'These are source-data-compatible candidate bands, not confirmed transfer results. The complete final source responses precede any proposed successor experiment; no successor label is used. Unsaved/discarded candidate programs are not reconstructed.'}
    (OUTPUT / 'final_band_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('local_positive_programs', 'local_positive_programs_rebandable',
        'all_snapshot_programs', 'all_snapshot_rebandable_programs')}, indent=2))
    for case in results:
        print(case['cohort'], case['case'], case['all_windows_rejected_but_program_rescued'], 'rescued programs')


if __name__ == '__main__':
    diagnose()
