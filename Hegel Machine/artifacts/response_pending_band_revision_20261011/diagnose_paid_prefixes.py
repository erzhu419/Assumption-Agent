#!/usr/bin/env python3
"""Reconstruct only paid prefixes and refit bands around fixed archived programs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation

SOURCE = ROOT / 'artifacts/response_intermediate_redecision_20261011'
OUTPUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def paid_prefix(responses, last):
    rows = []
    for row in responses:
        evidence = [i for i in row['evidence'] if i <= last]
        if evidence:
            rows.append({**row, 'evidence': evidence})
    return rows


def diagnose():
    results = []
    for brief in read(SOURCE / 'diagnostic_brief.json'):
        if brief['phase'] != 'after' or not brief['retirement_count']:
            continue
        trace = read(SOURCE / 'after' / brief['case'] / 'frozen_numeric/research_trace.json')
        questions = {q['id']: q for q in trace['questions']}
        state = trace['state']
        for retired in brief['retirements']:
            trial = next(t for t in state['tests'] if t['question'] == retired['question_id'])
            opened = questions[trial['response_question']]['opened_before_observation']
            cutoff = max(trial['retirement']['evidence'])
            initial = paid_prefix(state['response_settings'], opened - 1)
            current = paid_prefix(state['response_settings'], cutoff)
            features, feature_evidence = {}, []
            for row in state['research_table']['references'] + state['research_table']['targets']:
                features[row['crystal']] = {}
                for name, measurement in row['measurements'].items():
                    indices = measurement['evidence']
                    if not isinstance(indices, list):
                        indices = [indices]
                    if min(indices) < opened:
                        features[row['crystal']][name] = measurement['value']
                        feature_evidence.append(min(indices))
            retained, seen, forecasts, supporters = [], set(), {}, []
            for archive_index, model in enumerate(state['frozen_response_knowledge']['models']):
                program = Program.from_dict(model['program'])
                predictions = {n: program.predict(Observation(values, 0.)) for n, values in features.items()}
                if all((model['lower'] <= row['frequency'] - predictions[row['crystal']] <= model['upper'])
                       == row['activated'] for row in initial):
                    retained.append((archive_index, model, program, predictions))
            for index, (archive_index, model, program, predictions) in enumerate(retained):
                identity = (program, model['lower'], model['upper'])
                if identity in seen:
                    continue
                seen.add(identity)
                identifier = 'joint_response_' + str(index)
                requested = trial['prediction'] - predictions[trial['number']]
                positive = model['lower'] <= requested <= model['upper']
                forecasts[identifier] = positive
                if not positive:
                    continue
                residuals = [{**row, 'residual': row['frequency'] - predictions[row['crystal']]}
                             for row in current]
                positives = [row['residual'] for row in residuals if row['activated']]
                lower, upper = min(positives), max(positives)
                conflicts = [row for row in residuals if not row['activated'] and lower <= row['residual'] <= upper]
                supporters.append({'hypothesis_id': identifier, 'archive_model_index': archive_index,
                    'source_id': model.get('source_id', state['frozen_response_knowledge']['source_id']),
                    'family': model['family'], 'original_band': [model['lower'], model['upper']],
                    'required_positive_hull': [lower, upper], 'requested_residual': requested,
                    'negative_conflicts': [{key: row[key] for key in ('crystal', 'frequency', 'residual', 'evidence')}
                                          for row in conflicts],
                    'fixed_program_feasible': not conflicts,
                    'request_in_required_positive_hull': lower <= requested <= upper,
                    'supports_continuation': not conflicts and lower <= requested <= upper})
            latest = max((r for r in current if r['crystal'] == trial['number']), key=lambda r: max(r['evidence']))
            supported = sum(item['supports_continuation'] for item in supporters)
            results.append({'case': brief['case'], 'question_id': trial['question'],
                'number': trial['number'], 'requested_setting': trial['prediction'],
                'initial_selection_opened_before_observation': opened, 'paid_cutoff_observation': cutoff,
                'initial_response_count': len(initial), 'paid_response_count': len(current),
                'feature_evidence': sorted(set(feature_evidence)),
                'reconstructed_forecasts_equal_saved': forecasts == trial['response_forecasts'],
                'initial_compatible_archive_members': len(retained),
                'original_request_supporters': len(supporters),
                'feasible_fixed_programs': sum(s['fixed_program_feasible'] for s in supporters),
                'request_inside_positive_hull_members': supported,
                'latest_target_response': latest,
                'target_currently_active': latest['activated'],
                'frozen_rule_decision': 'continue_requested_trial' if supported and not latest['activated'] else 'return_to_parent',
                'supporters': supporters})
    result = {'source': str(SOURCE),
        'scope': 'Seven recorded retirements, reconstructed using only public measurements available at the initial choice and paid responses no later than that retirement. No world execution, fitting of coefficients, future response, or endpoint label is used.',
        'frozen_rule': 'Continue only if an initially positive archived fixed program remains feasible under a revised shared connected band and the requested residual lies within its required positive hull. Return if the target is already actually active.',
        'retirement_count': len(results),
        'exact_initial_forecast_reconstructions': sum(r['reconstructed_forecasts_equal_saved'] for r in results),
        'continue_count': sum(r['frozen_rule_decision'] == 'continue_requested_trial' for r in results),
        'cases': results}
    (OUTPUT / 'paid_prefix_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
    protocol = read(SOURCE / 'protocol.json')
    new_cases = deepcopy(protocol['cases'])
    for config in new_cases.values():
        config['signal_values'].update({'4': 7.25, '5': 14.75})
    (OUTPUT / 'recommended_new_object_cases.json').write_text(json.dumps({
        'status': 'Configuration only; no experiment prepared or executed', 'source_protocol': str(SOURCE / 'protocol.json'),
        'source_bundles': str(SOURCE / 'after/sources'), 'new_target_signals': [7.25, 14.75],
        'unchanged': 'All 13 archives, both adapted/original-return mechanisms, references, declarations, and limits',
        'scope': 'New target values within the existing deterministic fixture; not new random seeds or held-out mechanism families',
        'limits': protocol['limits'], 'cases': new_cases}, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('retirement_count', 'exact_initial_forecast_reconstructions', 'continue_count')}))
    for row in results:
        print(row['case'], row['requested_setting'], row['original_request_supporters'],
              row['feasible_fixed_programs'], row['request_inside_positive_hull_members'], row['frozen_rule_decision'])


if __name__ == '__main__':
    diagnose()
