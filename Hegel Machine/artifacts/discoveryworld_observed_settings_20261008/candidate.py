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
FAMILIES = (('density', 'densitometer'), ('gap', 'microscope'), ('channel5', 'spectrometer'))
ESTIMATORS = ({'id': 'spectral_multichannel_linear_4',
  'basis': [{'op': 'feature', 'name': 'channel1'},
            {'op': 'feature', 'name': 'channel2'},
            {'op': 'feature', 'name': 'channel3'},
            {'op': 'feature', 'name': 'channel4'},
            {'op': 'feature', 'name': 'channel5'}],
  'rationale': 'The individual density, gap, and channel5 linear models failed on reference fit. '
               'Test whether frequency is a linear combination of the full spectrometer signature; '
               'the current three-reference dataset makes this an exploratory, underdetermined fit '
               'that must be judged by its physical test.'},
 {'id': 'spectral_channels123_linear_4',
  'basis': [{'op': 'feature', 'name': 'channel1'},
            {'op': 'feature', 'name': 'channel2'},
            {'op': 'feature', 'name': 'channel3'}],
  'rationale': 'The five-channel linear prediction for crystal 4 (4697.099 Hz) failed physical '
               'activation at 4697 Hz (record 61), despite near-zero training residual, so it does '
               'not generalize. Test a lower-dimensional spectral model using channels 1–3, which '
               'is identifiable with the three known references; its physical trial is still '
               'required and a failure will motivate another revision.'},
 {'id': 'spectral_channels45_linear_4',
  'basis': [{'op': 'feature', 'name': 'channel4'}, {'op': 'feature', 'name': 'channel5'}],
  'rationale': 'The five-channel and channels 1–3 linear predictions both failed to activate '
               'Crystal 4 (records 61 and 72). Test a parsimonious model using channels 4–5; its '
               'fit and physical activation test will determine whether these features carry the '
               'missing signal.'},
 {'id': 'spectral_pair_interaction_4',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'channel1'},
                          {'op': 'feature', 'name': 'channel2'}]},
            {'op': 'mul',
             'children': [{'op': 'feature', 'name': 'channel3'},
                          {'op': 'feature', 'name': 'channel4'}]},
            {'op': 'feature', 'name': 'channel5'}],
  'rationale': 'Previous single-variable and linear spectral models failed physical tests or '
               'reference fit. Test whether frequencies depend on spectral channel interactions '
               'rather than additive channel values. This is an exploratory fitted basis; only the '
               'installed crystal’s physical activation test can assess its prediction.'},
 {'id': 'spectral_interaction_alt_4',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'channel1'},
                          {'op': 'feature', 'name': 'channel5'}]},
            {'op': 'mul',
             'children': [{'op': 'feature', 'name': 'channel2'},
                          {'op': 'feature', 'name': 'channel3'}]},
            {'op': 'feature', 'name': 'channel4'}],
  'rationale': 'The additive spectral models were physically refuted, while the first interaction '
               'basis predicted just beyond the allowed range. Test a distinct, compact set of '
               'channel interactions; its fitted prediction must be judged by an in-range physical '
               'activation test, not fit error alone.'},
 {'id': 'density_gap_linear_4',
  'basis': [{'op': 'feature', 'name': 'density'}, {'op': 'feature', 'name': 'gap'}],
  'rationale': 'Previously tested spectral additive and interaction predictions did not activate '
               'Crystal 4; several other candidates were out of range. Test a compact model '
               'combining two independent measured properties, density and quantum gap. Training '
               'fit alone is insufficient; only an in-range physical activation test can assess '
               'its prediction.'},
 {'id': 'density_gap_quadratic_4',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'density'},
                          {'op': 'feature', 'name': 'density'}]},
            {'op': 'mul',
             'children': [{'op': 'feature', 'name': 'gap'}, {'op': 'feature', 'name': 'gap'}]}],
  'rationale': 'The density-plus-gap linear prediction was physically refuted at 2804 Hz (record '
               '106); spectral additive and interaction candidates also failed or were out of '
               'range. Test a small nonlinear basis of density squared and gap squared. The three '
               'reference points identify its intercept and two coefficients, but fitting alone is '
               'not evidence of correctness; judge only the in-range physical test on Crystal 4.'},
 {'id': 'density_gap_product_4',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'density'}, {'op': 'feature', 'name': 'gap'}]},
            {'op': 'feature', 'name': 'density'}],
  'rationale': 'Density+gap linear and separate squared terms both failed their Crystal 4 '
               'activation tests (records 106 and 118), so their individual additive effects do '
               'not explain the target. Test a compact density-gap interaction plus density term. '
               'Treat the fit as exploratory; only the physical activation result can assess this '
               'prediction.'},
 {'id': 'density_gap_product_gap_4',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'density'}, {'op': 'feature', 'name': 'gap'}]},
            {'op': 'feature', 'name': 'gap'}],
  'rationale': 'Physical tests refuted the previously tested additive, squared, and '
               'density×gap-plus-density predictions for Crystal 4 (records 61, 72, 91, 106, 118, '
               '123). The last interaction model used density×gap plus density; replace its '
               'standalone density term with gap to test a distinct, equally compact '
               'material-feature basis. Its fitted prediction and physical response are unknown; '
               'only an in-range activation test can validate it.'},
 {'id': 'density_squared_density_gap_4',
  'basis': [{'op': 'mul',
             'children': [{'op': 'feature', 'name': 'density'},
                          {'op': 'feature', 'name': 'density'}]},
            {'op': 'mul',
             'children': [{'op': 'feature', 'name': 'density'}, {'op': 'feature', 'name': 'gap'}]}],
  'rationale': 'Six tested in-range Crystal 4 predictions failed activation (records 61, 72, 91, '
               '106, 118, 123, 134); two further candidates were out of range, not tested. '
               'Density-gap plus gap failed at 2320 Hz (134). Test a distinct compact nonlinear '
               'basis combining density squared and density-gap interaction. The three activated '
               'references fit this two-term basis, but the resulting C4 prediction and physical '
               'outcome are unknown; only an in-range activation test can assess it.'})


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


def _scan(observations, current):
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
                    'gap': r'quantum gap.*?be\s*([\d.]+)\s*nm',
                     'channel5': r'channel 5.*?([\d.]+)'}
        for feature, pattern in patterns.items():
            match = re.search(pattern, text, re.I | re.S)
            if match:
                readings[feature + '_' + str(int(crystal[1]))] = {
                    'value': float(match[1]), 'evidence': record['index']}
        # All channels are already present in the same public spectrometer use.
        for channel, value in re.findall(r'Channel\s+(\d+):\s*([\d.]+)', text, re.I):
            readings['channel' + channel + '_' + str(int(crystal[1]))] = {
                'value': float(value), 'evidence': record['index']}
        for estimator in ESTIMATORS:
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
    test = json.loads(STATE.read_text())['test']
    for record in reversed(observations):
        if record['index'] < test['started']:
            break
        view = _view(record)
        number, frequency = _dialog(view)
        if number != test['number'] or abs(frequency - test['prediction']) > .51:
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
                               'scope': 'This installed crystal at this current fitted prediction'},
                    'evidence': [record['index']]}
    return {'status': 'inconclusive', 'answer': 'No installed-crystal physical test at the predicted setting',
            'evidence': []}


def _extend_hypotheses(state, memory, number, path):
    """Return an exhausted scientific search to the existing local researcher."""
    if memory.get('program_arguments', {}).get('estimator_trial'):
        return {'stop': 'Declared estimators are exhausted for this target; return their measured outcomes to the researcher'}
    remaining = memory['limits']['max_model_calls'] - memory['costs']['model_calls']
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
        'research_table': state.get('research_table', {}), 'estimators': list(ESTIMATORS),
        'prior_program': previous, 'current_program': path}, indent=2))
    _save(state)
    return {'action': 'run', 'arguments': {'operator': 'inquire',
        'max_model_calls': min(32, remaining), 'question': {
            'id': question_id, 'parent_id': 'task', 'depends_on': [],
            'question': 'Which new measured variable or functional form explains the current crystal ' +
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
                'actual evidence are below; additional measurements may be declared by public instrument '
                'name and response pattern. If the present variables cannot explain the evidence, select '
                'a new measurement or return the specific missing information. Current measured table: ' +
                json.dumps(state.get('research_table', {}))}}}


def select_actions(public_spec, observations, memory):
    state = json.loads(STATE.read_text()) if STATE.exists() else {'completed': [], 'rejected': {}}
    view = public_spec['current_observation']
    known, crystals, reactors, readings, references = _scan(observations, view)
    responses = _response_settings(observations, crystals, reactors)
    state['response_settings'] = responses
    accessible = {obj['uuid'] for obj in view.get('accessibleEnvironmentObjects', [])}
    inventory = {obj['uuid'] for obj in view.get('inventoryObjects', [])}
    path = memory['program_path']
    questions = {item['id']: item for item in memory.get('questions', [])}
    test = state.get('test')
    if test and questions.get(test['question'], {}).get('status') in ('supported', 'refuted'):
        status = questions[test['question']]['status']
        if status == 'supported':
            state['completed'].append(test['number'])
            state['survivor'] = test['family']
        else:
            state['rejected'].setdefault(str(test['number']), []).append(test['family'])
        state.setdefault('tests', []).append({**test, 'status': status})
        state.pop('test')
        _save(state)
        test = None
    if view.get('dialog_box', {}).get('is_in_dialog'):
        number, frequency = _dialog(view)
        if test and number == test['number']:
            tuning = _tune(view, test['prediction'])
            if tuning:
                return tuning
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
            state.setdefault('setting_recoveries', []).append({**state.pop('recovery'),
                'restored': supported, 'evidence': [observations[-1]['index']]})
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
    estimators.sort(key=lambda item: item['id'] != state.get('survivor'))
    for estimator in estimators:
        family, features = estimator['id'], estimator['features']
        if family in state['rejected'].get(str(number), []):
            continue
        # New instruments are prerequisites only for the candidate that uses
        # them. They must not interrupt an already sufficient inherited law.
        for feature, measurement in estimator.get('measurements', {}).items():
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
        if fitted is None or prior_rows != rows:
            state.setdefault('fit_bindings', {})[qid] = rows
            state['binding'] = {'rows': rows, 'operations': estimator.get('operations', ['refit']),
                'tolerance': estimator.get('fit_tolerance', 1e-4),
                'probes': [{'id': 'current_crystal_' + str(number),
                    'features': {feature: readings[feature + '_' + str(number)]['value'] for feature in features},
                    'actions': [{'action': 'TALK', 'arg1': reactors[str(number)]}]}]}
            if 'basis' in estimator:
                state['binding']['basis'] = estimator['basis']
            _save(state)
            return {'action': 'run', 'arguments': {'operator': 'research_update', 'path': path,
                'question': {'id': qid, 'question': 'Fit the ' + family +
                             ' hypothesis to current activated references; predict current crystal ' + str(number),
                             'parent_id': 'task', 'depends_on': []}}}
        predictions = (fitted.get('selected_probe') or {}).get('predictions', [])
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
            break
        if fitted.get('status') != 'refuted':
            return {'stop': 'Current reference fit has no unique numerical prediction: ' + fitted['diagnosis']}
        state['rejected'].setdefault(str(number), []).append(family)
        state.setdefault('fit_refutations', []).append({'number': number, 'family': family,
            'fit_question': qid, 'diagnosis': fitted['diagnosis'], 'evidence': fitted['evidence']})
        _save(state)
    else:
        return _extend_hypotheses(state, memory, number, path)
    prediction = predictions[0]
    if not 0 <= prediction <= 10000:
        state['rejected'].setdefault(str(number), []).append(family)
        state.setdefault('out_of_range', []).append({'number': number, 'family': family,
            'prediction': prediction, 'allowed_range': [0, 10000], 'fit_question': qid})
        _save(state)
        return {'action': 'run', 'arguments': {'operator': 'question', 'id': qid + '_range',
            'question': 'Current candidate lies outside the publicly allowed control range', 'parent_id': 'task'}}
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
            'question': 'reference_test_' + str(number) + '_' + family}
        _save(state)
    result = _action('TALK', reactor)
    result['question'] = {'id': state['test']['question'], 'question':
        'Does the installed current crystal activate at the current ' + family + ' prediction?',
        'parent_id': 'task', 'depends_on': []}
    return result
