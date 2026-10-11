#!/usr/bin/env python3
"""Conservative fixed-program diagnosis on directly saved paid responses."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition import world_reference_recalibration as recipe

SOURCE = ROOT / 'artifacts/response_pending_band_revision_20261011'
OUTPUT = Path(__file__).resolve().parent
CASES = (
    ('preservation', 'square_original__gain_higher__adapted_law'),
    ('preservation', 'square_original__gain_higher__original_law_return'),
    ('new_objects', 'square_original__gain_lower__adapted_law'),
)


def read(path):
    return json.loads(path.read_text())


def diagnose():
    results = []
    for cohort, case in CASES:
        directory = SOURCE / cohort / 'after' / case / 'frozen_numeric'
        state = read(directory / 'research_trace.json')['state']
        trials = [*state.get('tests', []), *([state['test']] if state.get('test') else [])]
        records = read(directory / 'evidence_observations.json')
        _, crystals, reactors, _, _ = recipe._scan(records, records[-1]['observation'])
        crystals.update({str(t['number']): t['crystal'] for t in trials})
        reactors.update({str(t['number']): t['reactor'] for t in trials})
        saved = state['response_settings'] + recipe._response_settings(records, crystals, reactors)
        paid = {}
        for row in saved:
            key = (row['crystal'], row['reactor'], row['frequency'], row['activated'])
            entry = paid.setdefault(key, {**row, 'evidence': []})
            entry['evidence'] = sorted(set(entry['evidence'] + row['evidence']))
        responses = list(paid.values())
        exported_count = read(directory / 'learned_response_knowledge.json')['source_response_count']
        features = {r['crystal']: {k: v['value'] for k, v in r['measurements'].items()}
                    for r in state['research_table']['references'] + state['research_table']['targets']}
        revision = next(t['band_revision'] for t in trials
                        if t.get('band_revision', {}).get('positive_supporters'))
        programs = []
        for hypothesis in revision['hypotheses']:
            if hypothesis['id'] not in revision['positive_supporters']:
                continue
            program = Program.from_dict(hypothesis['program'])
            predictions = {n: program.predict(Observation(values, 0.)) for n, values in features.items()}
            residuals = [{**r, 'residual': r['frequency'] - predictions[r['crystal']]} for r in responses]
            positive = [r['residual'] for r in residuals if r['activated']]
            lower, upper = min(positive), max(positive)
            conflicts = [r for r in residuals if not r['activated'] and lower <= r['residual'] <= upper]
            negatives = [r['residual'] for r in residuals if not r['activated']]
            programs.append({'hypothesis_id': hypothesis['id'], 'original_band': [hypothesis['lower'], hypothesis['upper']],
                'required_positive_hull': [lower, upper],
                'lower_boundary_range': {'exclusive_min': max((v for v in negatives if v < lower), default=None),
                                         'inclusive_max': lower},
                'upper_boundary_range': {'inclusive_min': upper,
                                         'exclusive_max': min((v for v in negatives if v > upper), default=None)},
                'negative_conflicts': [{k: r[k] for k in ('crystal', 'frequency', 'residual', 'evidence')} for r in conflicts],
                'status': 'rejected_by_saved_counterexample' if conflicts else 'unknown_missing_final_response_rows',
                'compatible_with_saved_rows': not conflicts})
        results.append({'cohort': cohort, 'case': case,
            'complete_export_response_count': exported_count, 'directly_saved_response_count': len(responses),
            'missing_response_rows': exported_count - len(responses),
            'fixed_program_count': len(programs),
            'rejected_by_saved_rows': sum(not p['compatible_with_saved_rows'] for p in programs),
            'not_yet_rejected': sum(p['compatible_with_saved_rows'] for p in programs),
            'cross_target_consistency_established': False,
            'directly_saved_responses': responses, 'programs': programs})
    result = {'status': 'Conservative partial evidence diagnosis; awaiting complete paid state',
        'scope': 'Read saved records only. No environment execution, coefficient refitting, hidden world parameters, other-arm outcomes or inferred response labels.',
        'finding': 'Five programs in original_law_return are already inconsistent with a shared connected band: the positive hull required by final C5=1802 includes the actually negative C4=471. The other eight programs are not yet rejected by saved rows, but missing C5 negatives prevent a conclusion of complete compatibility.',
        'cases': results}
    (OUTPUT / 'recorded_band_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: row[key] for key in ('cohort', 'case', 'directly_saved_response_count',
        'missing_response_rows', 'fixed_program_count', 'rejected_by_saved_rows', 'not_yet_rejected')} for row in results], indent=2))


if __name__ == '__main__':
    diagnose()
