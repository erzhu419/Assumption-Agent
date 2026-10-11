"""Transfer fixed response programs through the existing physical researcher.

The archive is a set of conditional numerical hypotheses, not source-world
observations. Coefficients and response bands stay fixed until current evidence
rejects them; the ordinary researcher then handles new fits and revisions.
"""

from copy import deepcopy

from .language import Program
from .models import Observation


KNOWLEDGE_PATH = 'shared/frozen_response_knowledge.json'


def _measurements(estimators):
    return {name: deepcopy(spec) for estimator in estimators
            for name, spec in estimator.get('measurements', {}).items()}


def _features(state):
    table = state.get('research_table', {})
    return {row['crystal']: {name: value['value']
                            for name, value in row['measurements'].items()}
            for row in table.get('references', []) + table.get('targets', [])}


def export_response_knowledge(state, estimators, source_id):
    """Freeze surviving models and revise expired bands on final source data."""
    from .world_response_hypotheses import (
        _compatible_joint_models, _residual_band, _response_band_witnesses,
    )

    features, seen, candidates = _features(state), set(), []
    snapshots = [state.get('response_research', {}),
                 *state.get('response_research_history', []),
                 *[trial['band_revision'] for trial in
                   [*state.get('tests', []), state.get('test', {})] if trial.get('band_revision')]]
    for snapshot in snapshots:
        for item in snapshot.get('hypotheses', []):
            program = Program.from_dict(item['program'])
            key = (program, item['lower'], item['upper'])
            if key in seen:
                continue
            seen.add(key)
            if any(not program.required_features() <= values.keys()
                   for values in features.values()):
                continue
            candidates.append({key: deepcopy(item[key])
                               for key in ('program', 'lower', 'upper', 'family')})
            candidates[-1].update(source_question=item.get('source_question', snapshot.get('question_id')),
                                  design_target_number=None)
    cost = {'compute_cost': 0}
    retained = _compatible_joint_models(state, {'models': candidates}, cost)
    models = [{**{key: deepcopy(model[key]) for key in
                  ('lower', 'upper', 'family', 'source_question')},
               'program': model['program'].to_dict()} for _, model in retained]
    # A refuted interval does not refute its coefficient program. Reuse the
    # ordinary researcher's band inference only when no old interval survives.
    # This uses source-episode evidence; target-episode reuse stays frozen.
    revisions, surviving = [], {model['program'] for _, model in retained}
    by_program = {}
    for candidate in candidates:
        program = Program.from_dict(candidate['program'])
        if program not in surviving:
            by_program.setdefault(program, []).append(candidate)
    responses = state.get('response_settings', [])
    for program, expired in by_program.items():
        if not any(row['activated'] for row in responses):
            continue
        predictions = {n: program.predict(Observation(values, 0.)) for n, values in features.items()}
        residuals = [{**row, 'residual': row['frequency'] - predictions[row['crystal']]}
                     for row in responses]
        cost['compute_cost'] += len(features) + len(residuals)
        low, high, conflicts, before, after = _residual_band(residuals)
        table = state['research_table']
        evidence = sorted({i for row in responses for i in row.get('evidence', [])}
            | {i for row in table.get('references', []) + table.get('targets', [])
               for name in program.required_features()
               for refs in [row['measurements'][name]['evidence']]
               for i in (refs if isinstance(refs, list) else [refs])})
        source = expired[0]
        revision = {'program': program.to_dict(), 'family': source['family'],
            'source_question': source['source_question'],
            'status': 'refuted' if conflicts else 'rebuilt',
            'rejected_band_count': len(expired), 'positive_residual_span': [low, high],
            'lower_boundary_range': {'exclusive_min': before, 'inclusive_max': low},
            'upper_boundary_range': {'inclusive_min': high, 'exclusive_max': after},
            'conflicting_responses': deepcopy(conflicts), 'evidence': evidence, 'model_count': 0}
        revisions.append(revision)
        if conflicts:
            continue
        for lower, upper in _response_band_witnesses(low, high, before, after):
            models.append({'program': program.to_dict(), 'family': source['family'],
                'lower': lower, 'upper': upper, 'source_question': source['source_question'],
                'band_revision': {'origin': 'final_paid_response_band_revision',
                    'evidence': evidence, 'rejected_band_count': len(expired)}})
            revision['model_count'] += 1
    required = set().union(*(Program.from_dict(model['program']).required_features() for model in models))
    declarations = _measurements(estimators)
    return {'source_id': source_id, 'models': models,
            'measurements': {name: declarations.get(name) for name in sorted(required)},
            'source_response_count': len(state.get('response_settings', [])),
            'candidate_count': len(candidates), 'rejected_count': len(candidates) - len(retained),
            'revised_model_count': sum(item['model_count'] for item in revisions),
            'band_revisions': revisions,
            'compute_cost': cost['compute_cost']}


def combine_response_knowledge(previous, revised):
    """Keep fixed laws from both episodes; current evidence selects them at reuse."""
    archives = [archive for archive in (previous, revised) if archive and archive['models']]
    if not archives:
        return deepcopy(revised or previous)
    if len(archives) == 1:
        return deepcopy(archives[0])
    measurements = {}
    for archive in archives:
        required = set().union(*(Program.from_dict(model['program']).required_features()
                                 for model in archive['models']))
        if not required <= archive['measurements'].keys():
            raise ValueError('Frozen response models require their measurement declarations')
        for name, declaration in archive['measurements'].items():
            if name in measurements and measurements[name] != declaration:
                raise ValueError('Cannot combine changed measurement declarations: ' + name)
            measurements[name] = deepcopy(declaration)
    models, seen = [], set()
    for archive in archives:
        for model in archive['models']:
            key = (Program.from_dict(model['program']), model['lower'], model['upper'], model['family'])
            if key in seen:
                continue
            seen.add(key)
            models.append({**deepcopy(model), 'source_id': model.get('source_id', archive['source_id'])})
    return {'source_id': 'retained_response_archives', 'models': models,
            'measurements': measurements}


def forecast_response_knowledge(knowledge, features, settings):
    """Predict before response feedback; disagreement is not a probability."""
    models = [(Program.from_dict(model['program']).predict(Observation(features, 0.)),
               model['lower'], model['upper']) for model in knowledge['models']]
    forecasts = []
    for setting in settings:
        positive = sum(lower <= setting - prediction <= upper
                       for prediction, lower, upper in models)
        forecasts.append({'setting': setting, 'model_count': len(models),
            'positive_models': positive, 'negative_models': len(models) - positive,
            'prediction': (positive == len(models)) if models and positive in (0, len(models)) else None})
    return forecasts


def reuse_response_knowledge(state, estimators, path):
    """Offer frozen hypotheses to the existing chooser, with local refutation."""
    from .world_response_hypotheses import _compatible_joint_models, _retain_response_models

    knowledge = state.get('frozen_response_knowledge')
    if knowledge is None:
        return None
    number = next((n for n in state.get('target_numbers', []) if n not in state.get('completed', [])), None)
    responses, features = state.get('response_settings', []), _features(state)
    transfer = {'source_id': knowledge['source_id'], 'status': 'unavailable',
                'model_count': len(knowledge['models']), 'retained_count': 0, 'rejected_count': 0}
    if any('source_id' in model for model in knowledge['models']):
        transfer['source_counts'] = {}
        for model in knowledge['models']:
            source = model.get('source_id', knowledge['source_id'])
            counts = transfer['source_counts'].setdefault(source,
                {'model_count': 0, 'retained_count': 0, 'rejected_count': 0})
            counts['model_count'] += 1
    result = {'status': 'response_evidence_needed', 'target_number': number,
        'response_count': len(responses), 'candidates': [], 'hypotheses': [], 'selection': None,
        'selected_setting': None, 'selected_number': None, 'compute_cost': 0,
        'family_history_checks': deepcopy(state.get('family_history_checks', {})),
        'assumptions': ['Source coefficient programs and shared response intervals may recur in this world',
                        'Current responses can refute these fixed members; they do not confirm the full family'],
        'evidence': sorted({index for row in responses for index in row.get('evidence', [])}),
        'frozen_transfer': transfer}
    measurements = _measurements(estimators)
    if any(measurements.get(name) != spec for name, spec in knowledge['measurements'].items()):
        transfer['reason'] = 'measurement_declaration_changed'
        return result
    required = knowledge['measurements'].keys()
    if (number not in features or any(not required <= values.keys() for values in features.values())
            or any(row['crystal'] not in features for row in responses)
            or not any(row['activated'] for row in responses)):
        transfer['reason'] = 'current_public_measurements_or_responses_needed'
        return result
    table = state['research_table']
    for row in table['references'] + table.get('targets', []):
        for feature in required:
            evidence = row['measurements'][feature]['evidence']
            result['evidence'].extend(evidence if isinstance(evidence, list) else [evidence])
    result['evidence'] = sorted(set(result['evidence']))
    saved = [{**model, 'design_target_number': number,
              'target_setting': Program.from_dict(model['program']).predict(Observation(features[number], 0.))
                                + (model['lower'] + model['upper']) / 2,
              'fit_provenance': {'origin': 'frozen_response_transfer',
                                 'source_id': model.get('source_id', knowledge['source_id']),
                                 'source_question': model['source_question']}}
             for model in knowledge['models']]
    retained = _compatible_joint_models(state, {'models': saved}, result)
    transfer.update(retained_count=len(retained), rejected_count=len(saved) - len(retained))
    if 'source_counts' in transfer:
        for _, model in retained:
            transfer['source_counts'][model.get('source_id', knowledge['source_id'])]['retained_count'] += 1
        for counts in transfer['source_counts'].values():
            counts['rejected_count'] = counts['model_count'] - counts['retained_count']
    if not retained:
        transfer['status'] = 'refuted'
        return result
    result = _retain_response_models(result, state, retained, path)
    transfer['status'] = 'selected' if (result.get('selection') or {}).get('selected_experiment') else 'exhausted'
    if transfer['status'] == 'selected':
        result['selection']['consequence_scope'] = 'frozen_numeric_response_members'
    return result
