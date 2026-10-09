"""Actual scoped refitting and measured repair support different publication claims."""

from copy import deepcopy
from dataclasses import asdict
import json

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import PreparedPythonMethodAgent
from hegel_machine.cognition.python_scoped_method_learning import (
    ScopedPythonMethodLearner, scoped_revision_evidence,
)
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_candidate_retention_cycle_llm import prepare_episode
from test_python_method_learning import archived_predictor, publish


# A test-only numerical fixture, never supplied to the method author or runner.
SCOPED_TEST_METHOD = '''import json, random
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    steps = state['method_results']
    prior = memory['research_roles']['predictor']
    if not steps:
        return {'model': prior}
    feedback = steps[-1]['feedback']
    results = state['operator_results']
    refits = [item for item in results if item.get('operator') == 'refit']
    if not refits:
        return {'action': 'run', 'arguments': {'operator': 'refit',
            'model': prior, 'path': 'refits/initial'}}
    latest = refits[-1]
    if latest['status'] == 'refit_supported_on_validation':
        if feedback.get('fresh_confirmation', {}).get('passed') is False:
            rng = random.Random(len(records) + 193)
            return {'actions': [{'kind': 'trajectory', 'channels': [],
                'sequence': [{'u': rng.uniform(-1, 1)} for _ in range(24)]}],
                'after_observation': [{'operator': 'refit', 'model': prior,
                    'path': 'refits/settled', 'records': [len(records)-1, len(records)],
                    'validation_records': [len(records)]}]}
        return {'model': latest['path']}
    responses = [item for item in results if item.get('operator') == 'response']
    if responses:
        candidate = min(responses[-1]['candidates'], key=lambda item: item['validation_mse'])
        return {'model': candidate['path']}
    instruments = [item for item in results if item.get('operator') == 'instrument']
    if instruments:
        fitted = instruments[-1]
        assessed = [item for item in results if item.get('operator') == 'assess_instrument'][-1]
        return {'action': 'run', 'arguments': {'operator': 'response',
            'instrument': fitted['path'], 'records': fitted['records'] + assessed['records'],
            'validation_records': assessed['records'], 'output_dir': 'repaired'}}
    rng = random.Random(len(records) + 199)
    start = len(records)
    return {'actions': [{'kind': 'trajectory', 'channels': ['meter'],
        'sequence': [{'u': rng.uniform(-1, 1)} for _ in range(length)]}
        for length in (24, 24, 24)], 'after_observation': [
        {'operator': 'instrument', 'path': 'instrument.json', 'records': [start, start+1]},
        {'operator': 'assess_instrument', 'instrument': 'instrument.json', 'records': [start+2]}]}
'''


def make_scoped_learner(path, candidate):
    base = {'seed': 71, 'apparatus_seed': 910061, 'changed_response_seed': 910079}
    task = {'task_index': 1, 'seed': 713, 'apparatus_seed': 910061,
            'response_seed': 910061, 'instrument_mode': 'finite2'}
    changed = {**task, 'response_seed': 910079}
    checkpoint = {'after_rows': 96, 'apparatus_seed': 910061, 'response_seed': 1410088}
    specs = {'alpha': task, 'beta': changed, 'gamma': {**task, 'instrument_mode': 'recurrent'},
        'delta': {**changed, 'calibration_change': checkpoint},
        'epsilon': {**changed, 'calibration_change': {**checkpoint, 'instrument_mode': 'recurrent'}}}
    cases = {key: {'task': spec, 'confirmation_seed': 19371} for key, spec in specs.items()}
    prepared = []

    def laboratory(case):
        return TemporalInquiryLab(**case['task'])

    def prepare(case, directory, source, limits):
        plan = {'limits': asdict(limits),
                'prefix': {'trajectories': 3, 'length': 24, 'channels': []}}
        payload = prepare_episode(base, 1, directory, [candidate], plan,
            laboratory=laboratory(case), agent_class=PreparedPythonMethodAgent)
        payload['agent'].knowledge['research_roles'] = {'predictor': candidate['model']}
        payload['agent'].bind_method(source)
        prepared.append(payload)
        return payload

    learner = ScopedPythonMethodLearner(None, path, cases=cases,
        knowledge={'files': {METHOD_PATH: SCOPED_TEST_METHOD},
            'research_roles': {'predictor': candidate['model']}},
        lab_factory=laboratory, prepare_trial=prepare,
        config=InquiryLLMConfig(observation_budget=1800, compute_budget=3000000,
            execution_seconds=180., confirmation_rows=24, max_model_calls=12))
    return learner, prepared


def test_real_five_case_publication_requires_both_current_record_and_measured_recovery(
        tmp_path, archived_predictor):
    learner, prepared = make_scoped_learner(tmp_path / 'learner', archived_predictor)
    result = publish(learner)
    assert result.get('published') == METHOD_PATH, result
    assert len(prepared) == 5
    coverage = result['current_source_coverage']
    assert coverage['missing_cases'] == [] and coverage['prior_reuse_cases'] == ['alpha']
    assert coverage['scoped_revision'] == {'zero_meter_scoped_recovery_cases': ['delta'],
        'measured_recovery_cases': ['epsilon'], 'distinct_cases': True}
    scoped_trial, measured_trial = learner.practice_trials[-2:]
    assert scoped_trial['costs']['measurement_cost'] == 0
    assert measured_trial['costs']['measurement_cost'] > 0
    scoped = scoped_trial['scoped_revision']['zero_meter_scoped_recoveries'][0]
    assert scoped['validation_mse'] < 1e-10 < scoped['fresh_mse']
    assert scoped['refit_validation_mse'] < 1e-10 and scoped['recovery_mse'] < 1e-10
    assert scoped['records'] == [4, 5] and scoped['new_validation_records'] == [5]
    assert scoped['rejection_step'] < scoped['refit_step'] < scoped['recovery_step']
    measured = measured_trial['scoped_revision']['measured_recoveries'][0]
    assert measured['rejection_step'] < measured['instrument_step'] < measured['response_step'] < measured['recovery_step']
    for resource in ('observation_cost', 'measurement_cost', 'compute_cost', 'python_execution_seconds'):
        assert learner.costs[resource] == sum(trial['costs'][resource] for trial in learner.practice_trials)
    before = deepcopy(learner.costs)
    assert publish(learner)['publication_practices'] == []
    assert learner.costs == before
    learner._execute('write', {'path': METHOD_PATH,
        'code': SCOPED_TEST_METHOD + '\n# not yet executed\n'}, learner.interface)
    assert not learner._current_source_coverage()['scoped_revision']['distinct_cases']


def test_old_selective_gate_does_not_publish_when_measured_recovery_is_absent(tmp_path, archived_predictor):
    learner, prepared = make_scoped_learner(tmp_path / 'learner', archived_predictor)
    del learner.cases['epsilon']
    result = publish(learner)
    assert result['current_source_coverage']['missing_cases'] == []
    assert result['current_source_coverage']['selective_revision']['distinct_cases']
    assert result['scoped_revision_coverage']['zero_meter_scoped_recovery_cases'] == ['delta']
    assert result['scoped_revision_coverage']['measured_recovery_cases'] == []
    assert learner.model is learner.published_source is None and len(prepared) == 4


def test_scope_evidence_uses_actual_selected_and_validation_records(tmp_path, archived_predictor):
    learner, prepared = make_scoped_learner(tmp_path / 'learner', archived_predictor)
    outcome = learner._practice('delta')
    assert outcome['status'] == 'completed'
    result = json.loads((prepared[0]['agent'].workspace / 'result.json').read_text())
    assert scoped_revision_evidence(result)['zero_meter_scoped_recoveries']
    for change in ('include_old', 'validate_old_window', 'bought_meter'):
        altered = deepcopy(result)
        if change == 'bought_meter':
            altered['costs']['measurement_cost'] = 3
        for step in altered['method_results']:
            for item in [step['feedback'], *step['feedback'].get('after_observation_results', [])]:
                if item.get('operator') == 'refit' and item['split']['records'] == [4, 5]:
                    if change == 'include_old':
                        item['split']['records'] = [0, 4, 5]
                    elif change == 'validate_old_window':
                        item['split']['validation_records'] = [4]
        assert scoped_revision_evidence(altered)['zero_meter_scoped_recoveries'] == []
