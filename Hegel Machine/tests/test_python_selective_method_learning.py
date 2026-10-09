"""Publication needs actual selective refit and counterexample-recovery executions."""

from copy import deepcopy
from dataclasses import asdict
import json

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import PreparedPythonMethodAgent
from hegel_machine.cognition.python_selective_method_learning import (
    SelectivePythonMethodLearner, selective_revision_evidence,
)
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_candidate_retention_cycle_llm import prepare_episode
from test_python_method_learning import archived_predictor, ADAPTIVE_METHOD, publish


# Test-only program: selects by actual feedback, never by task identity. The
# production learner/runner must still learn its own program from paid practice.
SELECTIVE_TEST_METHOD = '''import json, random
def select_actions(spec, records, memory):
    state = json.load(open('research_state.json'))
    steps = state['method_results']
    prior = memory['research_roles']['predictor']
    if not steps:
        return {'model': prior}
    feedback = steps[-1]['feedback']
    outcomes = state['operator_results']
    refits = [item for item in outcomes if item.get('operator') == 'refit']
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
    return state['research_advice']['next_action']
'''


def selective_learner(path, candidate, source=SELECTIVE_TEST_METHOD, *, cases=None):
    base = {'seed': 71, 'apparatus_seed': 910061, 'changed_response_seed': 910079}
    task = {'task_index': 1, 'seed': 713, 'apparatus_seed': 910061,
            'response_seed': 910061, 'instrument_mode': 'finite2'}
    specs = {
        'alpha': task,
        'beta': {**task, 'response_seed': 910079},
        'gamma': {**task, 'instrument_mode': 'recurrent'},
        'delta': {**task, 'response_seed': 910079, 'calibration_change': {
            'after_rows': 96, 'apparatus_seed': 910061, 'response_seed': 1410088}},
    }
    cases = cases or {key: {'task': spec, 'confirmation_seed': 19371}
                      for key, spec in specs.items()}
    prepared = []

    def laboratory(case):
        return TemporalInquiryLab(**case['task'])

    def prepare(case, directory, method_source, limits):
        plan = {'limits': asdict(limits),
                'prefix': {'trajectories': 3, 'length': 24, 'channels': []}}
        payload = prepare_episode(base, 1, directory, [candidate], plan,
            laboratory=laboratory(case), agent_class=PreparedPythonMethodAgent)
        payload['agent'].knowledge['research_roles'] = {'predictor': candidate['model']}
        payload['agent'].bind_method(method_source)
        prepared.append(payload)
        return payload

    learner = SelectivePythonMethodLearner(None, path, cases=cases,
        knowledge={'files': {METHOD_PATH: source},
                   'research_roles': {'predictor': candidate['model']}},
        lab_factory=laboratory, prepare_trial=prepare,
        config=InquiryLLMConfig(observation_budget=1800, compute_budget=3000000,
            execution_seconds=180., confirmation_rows=24, max_model_calls=12))
    return learner, prepared


def test_real_four_case_publication_preserves_selective_refit_and_recovers_after_fresh_rejection(
        tmp_path, archived_predictor):
    learner, prepared = selective_learner(tmp_path / 'learner', archived_predictor)
    result = publish(learner)
    assert result.get('published') == METHOD_PATH, result
    assert len(prepared) == 4
    coverage = result['current_source_coverage']
    assert coverage['confirmed_cases'] == ['alpha', 'beta', 'delta', 'gamma']
    assert coverage['prior_reuse_cases'] == ['alpha']
    assert coverage['selective_revision']['zero_meter_refit_cases'] == ['beta', 'delta']
    assert coverage['selective_revision']['validated_refit_recovery_cases'] == ['delta']
    assert coverage['selective_revision']['distinct_cases']
    recovery = learner.practice_trials[-1]['selective_revision']
    assert recovery['recoveries']
    witness = recovery['recoveries'][0]
    assert witness['validation_mse'] < 1e-10 < witness['fresh_mse']
    assert witness['recovery_mse'] < 1e-10
    assert witness['validation_step'] < witness['rejection_step'] < witness['recovery_step']
    actual = prepared[-1]['agent'].method_results
    assert any(step['feedback'].get('after_observation_results') for step in actual)
    assert actual[witness['rejection_step']]['feedback']['method_submission'] == witness['submitted_model']
    assert prepared[-1]['agent'].confirmations[-1]['passed']
    assert learner.costs['model_calls'] == learner.costs['reported_tokens'] == 0
    for resource in ('observation_cost', 'measurement_cost', 'compute_cost', 'python_execution_seconds'):
        assert learner.costs[resource] == sum(trial['costs'][resource] for trial in learner.practice_trials)

    # Both paid execution and evidence parsing are retained across resubmission.
    before = deepcopy(learner.costs)
    assert publish(learner)['publication_practices'] == []
    assert learner.costs == before and len(prepared) == 4
    learner._execute('write', {'path': METHOD_PATH,
        'code': SELECTIVE_TEST_METHOD + '\n# unexecuted revision\n'}, learner.interface)
    revised = learner._current_source_coverage()
    assert revised['confirmed_cases'] == []
    assert not revised['selective_revision']['distinct_cases']


def test_original_accuracy_reuse_revision_gate_cannot_publish_without_observed_recovery(
        tmp_path, archived_predictor):
    task = {'task_index': 1, 'seed': 713, 'apparatus_seed': 910061,
            'response_seed': 910061, 'instrument_mode': 'finite2'}
    cases = {'old': {'task': task, 'confirmation_seed': 19371},
             'changed': {'task': {**task, 'response_seed': 910079}, 'confirmation_seed': 19372}}
    learner, prepared = selective_learner(tmp_path / 'learner', archived_predictor,
        ADAPTIVE_METHOD, cases=cases)
    result = publish(learner)
    assert result['current_source_coverage']['missing_cases'] == []
    assert result['current_source_coverage']['prior_reuse_cases'] == ['old']
    assert result['current_source_coverage']['revised_cases'] == ['changed']
    assert result['selective_revision_coverage']['zero_meter_refit_cases'] == ['changed']
    assert result['selective_revision_coverage']['validated_refit_recovery_cases'] == []
    assert learner.model is learner.published_source is None
    assert len(prepared) == 2
    assert learner.costs['observation_cost'] == sum(trial['costs']['observation_cost']
                                                 for trial in learner.practice_trials)


def test_validated_refit_fresh_rejection_without_repair_is_not_recovery(tmp_path, archived_predictor):
    source = SELECTIVE_TEST_METHOD.replace("if feedback.get('fresh_confirmation', {}).get('passed') is False:",
        "if feedback.get('fresh_confirmation', {}).get('passed') is False:\n"
        "            return {'stop': 'the separately validated fit was refuted'}\n"
        "        if False:")
    learner, prepared = selective_learner(tmp_path / 'learner', archived_predictor, source)
    result = learner._practice('delta')
    assert result['status'] == 'method_stopped'
    evidence = learner.practice_trials[0]['selective_revision']
    assert evidence['validated_refit_rejections']
    assert evidence['recoveries'] == evidence['zero_meter_refit_confirmations'] == []
    assert evidence['validated_refit_rejections'][0]['fresh_mse'] > 1e-10
    assert learner.model is None
    # Printed claims and a different submitted model path do not establish a
    # rejection of the validated candidate, even if that submission really fails.
    actual = json.loads((prepared[0]['agent'].workspace / 'result.json').read_text())
    copied = deepcopy(actual)
    for step in copied['method_results']:
        if step['feedback'].get('method_submission', '').startswith('refits/'):
            step['feedback']['method_submission'] = 'some_other_model.py'
            step['feedback']['stdout'] = 'validated refit recovered successfully'
    assert selective_revision_evidence(copied)['validated_refit_rejections'] == []
