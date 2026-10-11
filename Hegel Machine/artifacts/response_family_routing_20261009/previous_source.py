"""Current-world execution and extension of discovered reference hypotheses.

This is a host integration of the source authors' density-linear and gap-linear
families, their public instrument operations, and the existing relation fitter.
Role binding is a host adapter over the first public object states, not a
model-discovered scientific relation.
It is not an autonomous invention of those families. Export this file unchanged
as an ordinary select_actions program; its JSON contains current-episode state.
Models extend ESTIMATORS with measured-variable bases; the same acquisition,
coefficient fitter, physical controller and outcome analysis execute them.
"""

import json
from pathlib import Path
import re


STATE = Path('shared/reference_recalibration.json')
FAMILIES = (('density', 'densitometer'), ('gap', 'microscope'))
ESTIMATORS = ({'id': 'density_square',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'density'},
                          {'op': 'feature', 'name': 'density'}]}],
  'rationale': 'Test whether the measured quantity enters quadratically.'},
 {'id': 'gap_square',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'gap'}, {'op': 'feature', 'name': 'gap'}]}],
  'rationale': 'Test whether the measured quantity enters quadratically.'},
 {'id': 'grammar_density_1',
  'basis': [{'op': 'feature', 'value': None, 'name': 'density', 'lag': 0, 'children': []}],
  'origin': 'finite_grammar_search/public_calibration/old_refit_composition',
  'rationale': 'fit the observed current features jointly'},
 {'id': 'grammar_density_2',
  'basis': [{'op': 'feature', 'value': None, 'name': 'density', 'lag': 0, 'children': []},
            {'op': 'mul',
             'value': None,
             'name': None,
             'lag': 0,
             'children': [{'op': 'feature',
                           'value': None,
                           'name': 'density',
                           'lag': 0,
                           'children': []},
                          {'op': 'feature',
                           'value': None,
                           'name': 'density',
                           'lag': 0,
                           'children': []}]}],
  'origin': 'finite_grammar_search/public_calibration/old_refit_composition',
  'rationale': 'test a nonlinear coordinate composed from an observable'},
 {'id': 'grammar_gap_1',
  'basis': [{'op': 'feature', 'value': None, 'name': 'gap', 'lag': 0, 'children': []}],
  'origin': 'finite_grammar_search/public_calibration/old_refit_composition',
  'rationale': 'fit the observed current features jointly'},
 {'id': 'grammar_gap_2',
  'basis': [{'op': 'feature', 'value': None, 'name': 'gap', 'lag': 0, 'children': []},
            {'op': 'mul',
             'value': None,
             'name': None,
             'lag': 0,
             'children': [{'op': 'feature', 'value': None, 'name': 'gap', 'lag': 0, 'children': []},
                          {'op': 'feature',
                           'value': None,
                           'name': 'gap',
                           'lag': 0,
                           'children': []}]}],
  'origin': 'finite_grammar_search/public_calibration/old_refit_composition',
  'rationale': 'test a nonlinear coordinate composed from an observable'},
 {'id': 'signal_linear_5',
  'basis': [{'op': 'feature', 'name': 'signal'}],
  'rationale': 'Signal readings distinguish crystal 5 (19.887) from activated references 1–3 (1, '
               '2, 3). Test a simple fitted signal-magnitude basis as an exploratory frequency '
               'predictor; its extrapolation is uncertain and only physical activation can '
               'validate it.'},
 {'id': 'signal_linear_measured_5',
  'basis': [{'op': 'feature', 'name': 'signal'}],
  'measurements': {'signal': {'instrument': 'signal meter',
                              'pattern': 'signal magnitude = ([0-9.]+) units',
                              'group': 1}},
  'rationale': 'Test signal as an exploratory linear predictor after acquiring the same '
               'signal-meter measurement for references and target. Prior attempt failed in data '
               'assembly (KeyError: signal_1), not in a physical prediction; explicit measurement '
               'declaration should supply those readings.'},
 {'id': 'grammar_signal_1',
  'basis': [{'op': 'mul',
             'value': None,
             'name': None,
             'lag': 0,
             'children': [{'op': 'feature',
                           'value': None,
                           'name': 'signal',
                           'lag': 0,
                           'children': []},
                          {'op': 'feature',
                           'value': None,
                           'name': 'signal',
                           'lag': 0,
                           'children': []}]},
            {'op': 'feature', 'value': None, 'name': 'signal', 'lag': 0, 'children': []}],
  'origin': 'finite_grammar_search/public_calibration/old_refit_composition',
  'rationale': 'test a nonlinear coordinate composed from an observable',
  'measurements': {'signal': {'instrument': 'signal meter',
                              'pattern': 'signal magnitude = ([0-9.]+) units',
                              'group': 1}}})


def _basis_features(basis):
    names = []
    for term in basis:
        if term.get('op') == 'feature' and term['name'] not in names:
            names.append(term['name'])
        for name in _basis_features(term.get('children', [])):
            if name not in names:
                names.append(name)
    return names


def _estimators():
    inherited = [{'id': feature, 'features': [feature], 'operations': ['refit']}
                 for feature, _ in FAMILIES]
    return inherited + [{**item, 'features': _basis_features(item['basis'])} for item in ESTIMATORS]


def _save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def _objects(view):
    objects = list(view.get('inventoryObjects', [])) + list(view.get('accessibleEnvironmentObjects', []))
    for group in view.get('nearbyObjects', {}).get('objects', {}).values():
        objects.extend(group)
    return [obj for obj in objects if isinstance(obj, dict) and 'uuid' in obj]


def _view(record):
    return record.get('observation', record)


def _dialog(view):
    text = view.get('dialog_box', {}).get('dialogIn', '')
    match = re.search(r'reactor #(\d+).*?current resonance frequence is:\s*([\d.]+)', text, re.I | re.S)
    return (int(match[1]), float(match[2])) if match else (None, None)


def _installed(objects, crystal, reactor):
    return any(obj['uuid'] == crystal and re.search(
        r'\(in crystal reactor.*?\[uuid:\s*' + str(reactor) + r'\]\)',
        obj.get('description', ''), re.I) for obj in objects)


def _activated(objects, reactor):
    return any(obj['uuid'] == reactor and '(activated)' in obj.get('name', '').lower()
               for obj in objects)


def _scan(observations, current, estimators=None):
    known, crystals, reactors, readings, references = {}, {}, {}, {}, {}
    initial = {}
    for record in observations:
        view = _view(record)
        objects = _objects(view)
        known.update({str(obj['uuid']): obj for obj in objects})
        for obj in objects:
            initial.setdefault(obj['uuid'], obj)
            match = re.search(r'quantum crystal (\d+)', obj.get('name', ''), re.I)
            if match:
                crystals[str(int(match[1]))] = obj['uuid']
        number, frequency = _dialog(view)
        if number is not None:
            action = record.get('action', {})
            reactor = action.get('arg1') if action.get('action') == 'TALK' else reactors.get(str(number))
            if reactor is not None:
                reactors[str(number)] = reactor
                crystal = crystals.get(str(number))
                first = list(initial.values())
                if (_activated(first, reactor) and _installed(first, crystal, reactor)
                        and _activated(objects, reactor) and _installed(objects, crystal, reactor)):
                    references[str(number)] = {'value': frequency, 'evidence': record['index']}
        if record.get('action', {}).get('action') != 'USE' or not record.get('feedback', {}).get('success'):
            continue
        text = view.get('lastActionMessage', '')
        crystal = re.search(r'quantum crystal (\d+)', text, re.I)
        if not crystal:
            continue
        patterns = {'density': r'density.*?approximately\s*([\d.]+)\s*grams',
                    'gap': r'quantum gap.*?be\s*([\d.]+)\s*nm'}
        for feature, pattern in patterns.items():
            match = re.search(pattern, text, re.I | re.S)
            if match:
                readings[feature + '_' + str(int(crystal[1]))] = {
                    'value': float(match[1]), 'evidence': record['index']}
        # All channels are already present in the same public spectrometer use.
        for channel, value in re.findall(r'Channel\s+(\d+):\s*([\d.]+)', text, re.I):
            readings['channel' + channel + '_' + str(int(crystal[1]))] = {
                'value': float(value), 'evidence': record['index']}
        for estimator in ESTIMATORS if estimators is None else estimators:
            for feature, measurement in estimator.get('measurements', {}).items():
                instrument = known.get(str(record['action'].get('arg1')), {})
                if instrument.get('name', '').lower() != measurement['instrument'].lower():
                    continue
                match = re.search(measurement['pattern'], text, re.I | re.S)
                if match:
                    readings[feature + '_' + str(int(crystal[1]))] = {
                        'value': float(match[measurement.get('group', 1)]), 'evidence': record['index']}
    known.update({str(obj['uuid']): obj for obj in _objects(current)})
    for obj in known.values():
        match = re.search(r'quantum crystal (\d+)', obj.get('name', ''), re.I)
        if match:
            crystals[str(int(match[1]))] = obj['uuid']
    return known, crystals, reactors, readings, references


def _action(name, arg1=None, arg2=None):
    action = {'action': name}
    if arg1 is not None:
        action['arg1'] = arg1
    if arg2 is not None:
        action['arg2'] = arg2
    return {'actions': [action]}


def _response_settings(observations, crystals, reactors):
    """Retain measured control responses, without inventing a resonance label."""
    settings = {}
    for record in observations:
        view = _view(record)
        number, frequency = _dialog(view)
        reactor = reactors.get(str(number))
        if reactor is None or not view.get('dialog_box', {}).get('is_in_dialog'):
            continue
        objects = _objects(view)
        positive = _activated(objects, reactor)
        negative = any(obj['uuid'] == reactor and '(uncalibrated)' in obj.get('name', '').lower()
                       for obj in objects)
        if not (positive or negative):
            continue
        for crystal_number, crystal in crystals.items():
            if _installed(objects, crystal, reactor):
                key = (int(crystal_number), reactor, frequency, positive)
                item = settings.setdefault(key, {'crystal': int(crystal_number), 'reactor': reactor,
                    'frequency': frequency, 'activated': positive, 'evidence': []})
                item['evidence'].append(record['index'])
    return list(settings.values())


def refresh_response_evidence(state, observations, current, estimators):
    """Refresh the solver's public inputs after ordinary agent experiments too."""
    _, crystals, reactors, readings, references = _scan(observations, current, estimators)
    state['response_settings'] = _response_settings(observations, crystals, reactors)
    table = state.get('research_table', {})
    for row in table.get('references', []) + table.get('targets', []):
        number = str(row['crystal'])
        row['measurements'].update({key.rsplit('_', 1)[0]: value for key, value in readings.items()
                                    if key.endswith('_' + number)})
        if 'frequency' in row and number in references:
            row['frequency'] = references[number]
        if 'response_settings' in row:
            row['response_settings'] = [item for item in state['response_settings']
                                       if item['crystal'] == row['crystal']]
    grids = _control_grids(observations, state['response_settings'],
                           {row['crystal'] for row in table.get('references', [])})
    state['control_grids'] = grids
    number = next((n for n in state.get('target_numbers', []) if n not in state.get('completed', [])), None)
    if str(number) in grids:
        state['control_grid'] = grids[str(number)]


def _control_grids(observations, responses, reference_ids):
    # Initial reference readouts need not lie on the adjustable control lattice.
    # Learn from actual frequency-changing actions; returning to an initial
    # calibration readout does not overwrite that observed lattice.
    baselines = {}
    for response in responses:
        if response['crystal'] in reference_ids and response['activated']:
            baselines.setdefault(response['crystal'], response['frequency'])
    grids, adjusted_grids, previous_view = {}, {}, {}
    for record in observations:
        view = _view(record)
        number, frequency = _dialog(view)
        dialog = view.get('dialog_box', {})
        steps = [float(match[1]) for label in dialog.get('dialogOptions', {}).values()
                 if (match := re.search(r'frequency by ([\d.]+) hz', label, re.I))]
        bounds = re.search(r'allowable range is ([\d.]+) to ([\d,]+)', dialog.get('dialogIn', ''), re.I)
        if number is not None and steps and bounds:
            step = min(steps)
            grid = {'step': step, 'origin': frequency % step,
                'lower': float(bounds[1]), 'upper': float(bounds[2].replace(',', ''))}
            grids[str(number)] = grid
            choice = record.get('action', {}).get('chosen_dialog_option_int')
            options = previous_view.get('dialog_box', {}).get('dialogOptions', {})
            label = options.get(choice, options.get(str(choice), ''))
            if (_dialog(previous_view)[0] == number and frequency != baselines.get(number)
                    and re.search(r'(increase|decrease) frequency by', label, re.I)):
                adjusted_grids[str(number)] = grid
        previous_view = view
    grids.update(adjusted_grids)
    return grids


def _tune(view, prediction):
    _, frequency = _dialog(view)
    delta = prediction - frequency
    choices = []
    for key, label in view['dialog_box'].get('dialogOptions', {}).items():
        match = re.search(r'(increase|decrease) frequency by (\d+) hz', label, re.I)
        if match:
            step = int(match[2]) * (1 if match[1].lower() == 'increase' else -1)
            if abs(delta - step) < abs(delta):
                choices.append((abs(delta - step), int(key)))
    return {'actions': [{'chosen_dialog_option_int': min(choices)[1]}]} if choices else None


def _location(public_spec, part):
    names = public_spec.get('capabilities', {}).get('teleport_locations', {})
    name = next((name for name in names if part in name.lower()), None)
    return _action('TELEPORT_TO_LOCATION', name) if name else {'stop': 'No public location for ' + part}


def _approach(uuid, view):
    # TELEPORT_TO_OBJECT uses public IDs and faces its target. If already adjacent
    # but not facing it, the public directional grouping supplies the rotation.
    for direction, objects in view.get('nearbyObjects', {}).get('objects', {}).items():
        if direction in ('north', 'south', 'east', 'west') and any(
                obj.get('uuid') == uuid and obj.get('distance') == 1 for obj in objects):
            return _action('ROTATE_DIRECTION', direction)
    return _action('TELEPORT_TO_OBJECT', uuid)


def _exit(view):
    options = view.get('dialog_box', {}).get('dialogOptions', {})
    option = next(int(key) for key, label in options.items() if str(label).lower() == 'exit')
    return {'actions': [{'chosen_dialog_option_int': option}]}


def bind(public_spec, observations, memory):
    """Supply explicitly cited current rows to the historical finite refitter."""
    return json.loads(STATE.read_text())['binding']


def answer(public_spec, observations, memory):
    state = json.loads(STATE.read_text())
    test = state['test']
    response_feedback = None
    if test.get('response_question'):
        _, crystals, reactors, _, _ = _scan(observations, public_spec['current_observation'])
        state['response_settings'] = _response_settings(observations, crystals, reactors)
        measured = [{**item, 'evidence': [i for i in item['evidence'] if i >= test['started']]}
                    for item in state['response_settings'] if item['crystal'] == test['number']
                    and any(i >= test['started'] for i in item['evidence'])]
        hypotheses = state['response_research']['hypotheses']
        refuted = [item['id'] for item in hypotheses if any(
            (item['lower'] <= response['frequency'] -
                item['predictions_by_crystal'][str(test['number'])] <= item['upper'])
                != response['activated'] for response in measured)]
        response_feedback = {'requested_setting': test['prediction'],
            'observed_responses': measured,
            'remaining_hypotheses': [item['id'] for item in hypotheses if item['id'] not in refuted],
            'refuted_hypotheses': refuted,
            'scope': 'Fixed response hypotheses compared with actual intermediate settings; '
                     'a successful intermediate setting does not confirm the requested setting'}
        state['response_feedback'] = response_feedback
        _save(state)
    for record in reversed(observations):
        if record['index'] < test['started']:
            break
        view = _view(record)
        number, frequency = _dialog(view)
        tolerance = 1e-6 if 'restore_frequency' in test else .51
        if number != test['number'] or abs(frequency - test['prediction']) > tolerance:
            continue
        objects = _objects(view)
        if not _installed(objects, test['crystal'], test['reactor']):
            continue
        positive = _activated(objects, test['reactor'])
        negative = any(obj['uuid'] == test['reactor'] and '(uncalibrated)' in obj.get('name', '').lower()
                       for obj in objects)
        if positive or negative:
            return {'status': 'supported' if positive else 'refuted',
                    'answer': {'family': test['family'], 'prediction': test['prediction'],
                               'actual_frequency': frequency, 'activated': positive,
                               'scope': 'This installed crystal at this current fitted prediction',
                               **({'response_feedback': response_feedback} if response_feedback else {})},
                    'evidence': [record['index']]}
    if response_feedback:
        return {'status': 'inconclusive', 'answer': response_feedback,
                'evidence': sorted({i for item in response_feedback['observed_responses']
                                    for i in item['evidence']})}
    return {'status': 'inconclusive', 'answer': 'No installed-crystal physical test at the predicted setting',
            'evidence': []}


def _extend_hypotheses(state, memory, number, path):
    """Return an exhausted scientific search to the existing local researcher."""
    estimator_trial = memory.get('program_arguments', {}).get('estimator_trial', False)
    table = state.get('research_table', {})
    rows = table.get('references', []) + table.get('targets', [])
    features = set.intersection(*(set(row['measurements']) for row in rows)) if rows else set()
    if not estimator_trial and features - set(state.get('expanded_features', [])):
        _save(state)
        return {'action': 'run', 'arguments': {'operator': 'reference_expansion', 'path': path}}
    context = {'number': number, 'estimators': list(ESTIMATORS),
        'eligible_targets': [n for n in state.get('target_numbers', [])
                             if n not in state.get('completed', [])
                             and any(row['crystal'] == n for row in rows)],
        'control_grids': state.get('control_grids', {}),
        'responses': [[r['crystal'], r['frequency'], r['activated']]
                      for r in state.get('response_settings', [])],
        'measurements': [[row['crystal'], {key: value['value']
            for key, value in row['measurements'].items()}] for row in rows]}
    response = state.get('response_research', {})
    untried_constraints = (response.get('status') in
        ('no_compatible_response_hypotheses', 'no_untried_supported_setting')
        and 'joint_parameter_search' not in response and state.get('control_grid'))
    if ESTIMATORS and (state.get('response_context') != context or untried_constraints):
        return {'action': 'run', 'arguments': {'operator': 'reference_response',
                                             'path': path, 'context': context}}
    remaining = memory['limits']['max_model_calls'] - memory['costs']['model_calls']
    selection = response.get('selection')
    if not estimator_trial and response.get('status') == 'family_research_needed' and (
            not memory.get('model_available', True) or remaining <= 0):
        # A family-review request does not erase a still executable operational
        # experiment when no live researcher is available in this invocation.
        selection = response.get('operational_fallback')
    experiment = (selection or {}).get('selected_experiment')
    if experiment is not None:
        return {'response_trial': {'family': experiment['family'],
            'prediction': experiment['prediction'],
            'response_question': response['question_id'],
            'response_forecasts': experiment['predictions'],
            **{key: experiment[key] for key in ('number', 'restore_frequency') if key in experiment},
            'question_id': response['question_id'] + '_physical'}}
    if estimator_trial:
        obligation = response.get('family_research', {}).get('research_obligation', {})
        return {'stop': 'Declared estimators need further research after ' +
                response.get('status', 'exhausted prediction trials') + ': ' +
                obligation.get('question', 'no remaining experiment selected from the declared response families') +
                '; return the acquired outcomes to the proposing researcher'}
    if not memory.get('model_available', True) or remaining <= 0:
        return {'stop': 'Inherited hypotheses are exhausted; new research needs a live model and remaining calls'}
    suffix, attempt, returned = str(number), 1, []
    questions = {item['id']: item for item in memory.get('questions', [])}
    question_id = 'reference_extension_' + suffix
    while memory.get('llm_inquiries', {}).get(question_id, {}).get('invocations', 0):
        question = questions.get(question_id, {})
        fresh_refutation = question.get('status') == 'refuted' and any(
            index >= question.get('opened_before_observation', float('inf'))
            for index in question.get('evidence', []))
        if not fresh_refutation:
            return {'stop': 'Local research returned without a usable extension for this crystal; '
                            'its notebook and rejected predictions remain available'}
        returned.append({**question,
            'notes': memory['llm_inquiries'][question_id].get('notes', '')})
        attempt += 1
        suffix = str(number) + '_' + str(attempt)
        question_id = 'reference_extension_' + suffix
    previous = 'shared/reference_before_extension_' + suffix + '.py'
    Path(previous).write_text(Path(path).read_text())
    failure_file = 'shared/reference_extension_' + suffix + '.json'
    failures = [item for item in state.get('tests', []) if item['number'] == number]
    Path(failure_file).write_text(json.dumps({
        'crystal_number': number, 'rejected_families': state['rejected'].get(str(number), []),
        'physical_tests': failures, 'returned_refutations': returned,
        'fit_refutations': [item for item in state.get('fit_refutations', []) if item['number'] == number],
        'out_of_range_predictions': state.get('out_of_range', []),
        'prediction_selection': state.get('prediction_selection'),
        'research_table': state.get('research_table', {}), 'estimators': list(ESTIMATORS),
        'family_research': response.get('family_research'),
        'prior_program': previous, 'current_program': path}, indent=2))
    _save(state)
    return {'action': 'run', 'arguments': {'operator': 'inquire',
        'max_model_calls': min(32, remaining), 'question': {
            'id': question_id, 'parent_id': 'task', 'depends_on': [],
            'question': 'Which shared response premise, measured variable or functional form explains the current crystal ' +
                str(number) + ' after the inherited hypotheses failed? Inspect ' + failure_file +
                ' and the current public records. Returned refutations rule out their tested predictions; '
                'use them to revise the next explanation rather than ending the entire search. '
                'Distinguish current-reference fit failures, out-of-range predictions and physical '
                'refutations. Submit a measured-variable/basis hypothesis through run '
                "{operator:'reference_hypothesis',path:'" + path + "',hypothesis:{id:YOUR_ID,basis:[PROGRAM_AST,...]}}. "
                'The existing fitter estimates the intercept and basis coefficients from current references; '
                'the existing controller installs, tunes and physically tests targets, returning results here. '
                'Choose the basis and revise it using physical counterexamples. Each distinct basis gets a '
                'new id and is retained in the reusable parent source. Existing measured features and their '
                'actual control responses constrain subsequent research. Declared bases also fit one '
                'observed successful setting per previously solved target. These are operational control '
                'labels, not exact latent resonance values; fit_tolerance is the declared MSE in Hz squared '
                '(default 1e-4). Reusing or fitting a success is not an independent confirmation. '
                'Strict fit failures remain in the numerical record, but approximate bases can still '
                'propose physical tests. Available declared bases are compared by leave-one-crystal-out '
                'error; this is a reuse diagnostic, not a noise estimate or an activation-width model. '
                'The last prediction comparison and actual outcomes are in the failure file. '
                'Additional measurements may be declared by public instrument '
                'name and response pattern. If the present variables cannot explain the evidence, select '
                'a new measurement or return the specific missing information. Current measured table: ' +
                json.dumps(state.get('research_table', {}))}}}


def select_actions(public_spec, observations, memory):
    state = json.loads(STATE.read_text()) if STATE.exists() else {'completed': [], 'rejected': {}}
    view = public_spec['current_observation']
    known, crystals, reactors, readings, references = _scan(observations, view)
    responses = _response_settings(observations, crystals, reactors)
    state['response_settings'] = responses
    state['control_grids'] = _control_grids(observations, responses, set(map(int, references)))
    accessible = {obj['uuid'] for obj in view.get('accessibleEnvironmentObjects', [])}
    inventory = {obj['uuid'] for obj in view.get('inventoryObjects', [])}
    path = memory['program_path']
    questions = {item['id']: item for item in memory.get('questions', [])}
    test = state.get('test')
    if test and (questions.get(test['question'], {}).get('status') in ('supported', 'refuted')
                 or (test.get('tuning_finished')
                     and questions.get(test['question'], {}).get('status') == 'inconclusive')):
        status = questions[test['question']]['status']
        if 'restore_frequency' in test:
            state['recovery'] = {'number': test['number'], 'frequency': test['restore_frequency'],
                'crystal': test['crystal'], 'reactor': test['reactor'],
                'response_question': test['response_question']}
        elif status == 'supported':
            state['completed'].append(test['number'])
            state['survivor'] = test['family']
        else:
            # Retire this family's point-fit branch. Response research below
            # can still test other coefficient/window hypotheses in that basis.
            state['rejected'].setdefault(str(test['number']), []).append(test['family'])
        state.setdefault('tests', []).append({**test, 'status': status})
        state.pop('test')
        _save(state)
        test = None
    recovery = state.get('recovery')
    if recovery and (not view.get('dialog_box', {}).get('is_in_dialog')
                     or _dialog(view)[0] != recovery['number']):
        if view.get('dialog_box', {}).get('is_in_dialog'):
            return _exit(view)
        return (_action('TALK', recovery['reactor']) if recovery['reactor'] in accessible
                else _approach(recovery['reactor'], view))
    if view.get('dialog_box', {}).get('is_in_dialog'):
        number, frequency = _dialog(view)
        if test and number == test['number']:
            tuning = _tune(view, test['prediction'])
            if tuning:
                if test.get('response_question'):
                    tuning['after_observation'] = [{'action': 'evaluate', 'arguments': {
                        'path': path, 'question_id': test['question']}}]
                return tuning
            if 'restore_frequency' in test:
                test['tuning_finished'] = True
                _save(state)
            result = _exit(view)
            result['after_observation'] = [{'action': 'evaluate', 'arguments': {
                'path': path, 'question_id': test['question']}}]
            return result
        recovery = state.get('recovery')
        if recovery and number == recovery['number']:
            tuning = _tune(view, recovery['frequency'])
            if tuning:
                return tuning
            supported = (_installed(_objects(view), recovery['crystal'], recovery['reactor'])
                         and _activated(_objects(view), recovery['reactor']))
            state.setdefault('setting_recoveries', []).append({**recovery,
                'actual_frequency': frequency,
                'restored': supported, 'evidence': [observations[-1]['index']]})
            if supported:
                state.pop('recovery')
            _save(state)
            if not supported:
                return {'stop': 'Previously successful setting did not restore activation; revise its conditions'}
        return _exit(view)

    # Reacquire instrument handles and reference roles from this world only.
    for feature, name in FAMILIES:
        instrument = next((obj['uuid'] for obj in known.values() if obj.get('name', '').lower() == name), None)
        if instrument is None:
            return _location(public_spec, 'instruments')
        if instrument not in inventory:
            return _action('PICKUP', instrument) if instrument in accessible else _approach(instrument, view)
    reactor = next((obj['uuid'] for obj in known.values()
        if 'crystal reactor' in obj.get('name', '').lower() and obj['uuid'] not in reactors.values()), None)
    if reactor is not None:
        return _action('TALK', reactor) if reactor in accessible else _approach(reactor, view)
    if len(reactors) < len(crystals) or not reactors:
        return _location(public_spec, 'reactor')
    if set(crystals) != set(reactors):
        return _location(public_spec, 'crystal bench')
    if len(references) < 2:
        return {'stop': 'Current public records lack two initially activated installed references'}
    if 'reference_numbers' not in state:
        state['reference_numbers'] = sorted(map(int, references))
        state['target_numbers'] = sorted(int(n) for n in crystals if n not in references)
        _save(state)
    for feature, name in FAMILIES:
        instrument = next(obj['uuid'] for obj in known.values() if obj.get('name', '').lower() == name)
        for number in sorted(map(int, crystals)):
            if feature + '_' + str(number) not in readings:
                crystal = crystals[str(number)]
                return (_action('USE', instrument, crystal) if crystal in accessible | inventory
                        else _approach(crystal, view))

    successful = {}
    for response in sorted(responses, key=lambda item: item['evidence'][-1]):
        if response['activated']:
            successful[response['crystal']] = response
    if not test:
        state['completed'] = [n for n in state['target_numbers'] if n in successful
            and _installed(list(known.values()), crystals[str(n)], reactors[str(n)])
            and _activated(list(known.values()), reactors[str(n)])]
    state['research_table'] = {'references': [], 'targets': []}
    for n in sorted(map(int, crystals)):
        values = {key.rsplit('_', 1)[0]: value for key, value in readings.items()
                  if key.endswith('_' + str(n))}
        row = {'crystal': n, 'measurements': values}
        if n in state['reference_numbers']:
            row['frequency'] = references[str(n)]
            state['research_table']['references'].append(row)
        else:
            row['response_settings'] = [item for item in responses if item['crystal'] == n]
            state['research_table']['targets'].append(row)
    _save(state)
    number = next((n for n in state['target_numbers'] if n not in state['completed']), None)
    if number is None:
        return {'stop': 'All current targets have physically observed activation support'}
    if not test and number in successful:
        setting = successful[number]
        crystal, reactor = crystals[str(number)], reactors[str(number)]
        if not _installed(_objects(view), crystal, reactor):
            if crystal not in inventory:
                return _action('PICKUP', crystal) if crystal in accessible else _approach(crystal, view)
            return _action('PUT', crystal, reactor) if reactor in accessible else _approach(reactor, view)
        if reactor not in accessible:
            return _approach(reactor, view)
        state['recovery'] = {'number': number, 'frequency': setting['frequency'],
            'crystal': crystal, 'reactor': reactor, 'source_evidence': setting['evidence']}
        _save(state)
        return _action('TALK', reactor)
    estimators = _estimators()
    measurements = {feature: measurement for estimator in estimators
                    for feature, measurement in estimator.get('measurements', {}).items()}
    estimators.sort(key=lambda item: item['id'] != state.get('survivor'))
    trials, selected = [], None
    for estimator in estimators:
        family, features = estimator['id'], estimator['features']
        if family in state['rejected'].get(str(number), []):
            fit_only = ('basis' in estimator and any(item['number'] == number and item['family'] == family
                for item in state.get('fit_refutations', [])) and not any(
                    item.get('number') == number and item.get('family') == family
                    for key in ('tests', 'response_constraints', 'out_of_range')
                    for item in state.get(key, [])))
            if not fit_only:
                continue
            # The strict fit remains rejected. Its shape can still propose an
            # approximate control prediction whose success needs a real trial.
            state['rejected'][str(number)].remove(family)
        # A later declaration can supply the acquisition for an earlier basis
        # using the same feature. Acquire only the current candidate's readings.
        for feature in features:
            if feature not in measurements:
                continue
            measurement = measurements[feature]
            if all(feature + '_' + str(n) in readings for n in map(int, crystals)):
                continue
            instrument = next((obj['uuid'] for obj in known.values()
                               if obj.get('name', '').lower() == measurement['instrument'].lower()), None)
            if instrument is None:
                return _location(public_spec, 'instruments')
            if instrument not in inventory:
                return _action('PICKUP', instrument) if instrument in accessible else _approach(instrument, view)
            for n in sorted(map(int, crystals)):
                if feature + '_' + str(n) not in readings:
                    crystal = crystals[str(n)]
                    return (_action('USE', instrument, crystal) if crystal in accessible | inventory
                            else _approach(crystal, view))
        qid = 'reference_fit_' + family + '_' + str(number)
        fitted = memory.get('relation_research', {}).get(qid)
        rows = [{'features': {feature: readings[feature + '_' + str(n)]['value'] for feature in features},
                     'target': references[str(n)]['value'],
                     'evidence': sorted({references[str(n)]['evidence'],
                         *[readings[feature + '_' + str(n)]['evidence'] for feature in features]})}
                for n in state['reference_numbers']]
        if 'basis' in estimator:
            for n in state['target_numbers']:
                if n == number or n not in successful:
                    continue
                setting = successful[n]
                rows.append({'features': {feature: readings[feature + '_' + str(n)]['value'] for feature in features},
                    'target': setting['frequency'], 'target_kind': 'observed_successful_setting',
                    'crystal': n, 'evidence': sorted({setting['evidence'][-1],
                        *[readings[feature + '_' + str(n)]['evidence'] for feature in features]})})
        prior_rows = state.get('fit_bindings', {}).get(qid, (fitted or {}).get('fit_rows', rows))
        if (fitted is None or prior_rows != rows
                or ('basis' in estimator and 'empirical_prediction' not in fitted)):
            state.setdefault('fit_bindings', {})[qid] = rows
            state['binding'] = {'rows': rows, 'operations': estimator.get('operations', ['refit']),
                'tolerance': estimator.get('fit_tolerance', 1e-4),
                'probes': [{'id': 'current_crystal_' + str(number),
                    'features': {feature: readings[feature + '_' + str(number)]['value'] for feature in features},
                    'actions': [{'action': 'TALK', 'arg1': reactors[str(number)]}]}]}
            if 'basis' in estimator:
                state['binding']['basis'] = estimator['basis']
                state['binding']['prediction_objective'] = 'operational_setting'
            _save(state)
            return {'action': 'run', 'arguments': {'operator': 'research_update', 'path': path,
                'question': {'id': qid, 'question': 'Fit the ' + family +
                             ' hypothesis to current activated references; predict current crystal ' + str(number),
                             'parent_id': 'task', 'depends_on': []}}}
        predictions = (fitted.get('selected_probe') or {}).get('predictions', [])
        empirical = fitted.get('empirical_prediction') if 'basis' in estimator else None
        if empirical:
            predictions = [item['prediction'] for item in empirical['probe_predictions']]
        if predictions:
            measured = [item for item in responses if item['crystal'] == number
                        and abs(item['frequency'] - predictions[0]) < .5]
            if measured:
                response = max(measured, key=lambda item: item['evidence'][-1])
                if not response['activated']:
                    state['rejected'].setdefault(str(number), []).append(family)
                    state.setdefault('response_constraints', []).append({'number': number, 'family': family,
                        'prediction': predictions[0], 'setting': response['frequency'],
                        'status': 'refuted', 'evidence': response['evidence'], 'fit_question': qid})
                    _save(state)
                    continue
            if not 0 <= predictions[0] <= 10000:
                state['rejected'].setdefault(str(number), []).append(family)
                state.setdefault('out_of_range', []).append({'number': number, 'family': family,
                    'prediction': predictions[0], 'allowed_range': [0, 10000], 'fit_question': qid})
                _save(state)
                continue
            proposal = {'family': family, 'prediction': predictions[0], 'fit_question': qid}
            if empirical:
                proposal.update(fit_mse=empirical['fit_mse'],
                    leave_one_out_mse=empirical['leave_one_out_mse'],
                    fit_status=('within_tolerance' if empirical['fit_mse'] <= estimator.get('fit_tolerance', 1e-4)
                                else 'approximate'), numerical_question_status=fitted['status'])
                trials.append(proposal)
                continue
            selected = proposal
            break
        if fitted.get('status') != 'refuted':
            return {'stop': 'Current reference fit has no unique numerical prediction: ' + fitted['diagnosis']}
        state['rejected'].setdefault(str(number), []).append(family)
        state.setdefault('fit_refutations', []).append({'number': number, 'family': family,
            'fit_question': qid, 'diagnosis': fitted['diagnosis'], 'evidence': fitted['evidence']})
        _save(state)
    if selected is None and trials:
        selected = min(trials, key=lambda item: (
            item['leave_one_out_mse'] if item['leave_one_out_mse'] is not None else float('inf'),
            item['fit_mse']))
        state['prediction_selection'] = {'number': number, 'selected_family': selected['family'],
            'criterion': 'leave_one_crystal_out_mse', 'candidates': trials,
            'scope': 'Prediction trials ordered by reused calibration evidence; '
                     'not fresh confirmation, a noise estimate or a known activation-range contrast'}
        _save(state)
    if selected is None:
        extension = _extend_hypotheses(state, memory, number, path)
        selected = extension.get('response_trial')
        if selected is None:
            return extension
    family, prediction = selected['family'], selected['prediction']
    number = selected.get('number', number)
    crystal, reactor = crystals[str(number)], reactors[str(number)]
    if not _installed(_objects(view), crystal, reactor):
        if crystal not in inventory:
            return _action('PICKUP', crystal) if crystal in accessible else _approach(crystal, view)
        return _action('PUT', crystal, reactor) if reactor in accessible else _approach(reactor, view)
    if reactor not in accessible:
        return _approach(reactor, view)
    if not test:
        state['test'] = {'number': number, 'family': family, 'prediction': prediction,
            'started': len(observations), 'crystal': crystal, 'reactor': reactor,
            'question': selected.get('question_id', 'reference_test_' + str(number) + '_' + family),
            **{key: selected[key] for key in ('fit_mse', 'fit_status', 'leave_one_out_mse',
                'response_question', 'response_forecasts', 'restore_frequency') if key in selected}}
        _save(state)
    result = _action('TALK', reactor)
    result['question'] = {'id': state['test']['question'], 'question':
        'Does the installed current crystal activate at the current ' + family + ' prediction?',
        'parent_id': state['test'].get('response_question', 'task'), 'depends_on': []}
    return result
