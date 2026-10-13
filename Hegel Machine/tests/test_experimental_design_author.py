"""Use saved trace shapes to test task feedback, never scientific discovery."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.experimental_design_author import (
    ExperimentalDesignMethodLearner, FEEDBACK_FILE, design_practice_feedback)
from test_anonymous_experiment_design_trace import example


SOURCE = "def select_actions(spec, records, memory): return {'stop': 'fixture'}\n"


def prediction_result():
    return {'status': 'completed', 'model': 'model.py', 'events': [], 'method_results': [],
        'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96},
        'costs': {'observation_cost': 169, 'measurement_cost': 0,
                  'compute_cost': 190, 'python_execution_seconds': 2.}}


def design_result(*, both_refuted=False):
    result = example()
    result['experiment_comparisons'][0]['selected_experiment']['actions'] = deepcopy(
        result['revision_proposals'][0]['actions'])
    if both_refuted:
        revision = result['revision_proposals'][0]
        revision['forecasts'][1]['prospective_mse'] = .25
        revision['authored_answer'] = {'status': 'refuted',
            'answer': 'Both existing explanations were refuted; revise them.', 'evidence': [72]}
        result['model'] = 'revised_after_probe.py'
        result['events'][-1]['arguments']['model'] = result['model']
    result['costs'] = prediction_result()['costs']
    return result


def author(tmp_path, *, provider=None, prepare_trial=None, max_model_calls=2):
    return ExperimentalDesignMethodLearner(provider, tmp_path,
        knowledge={'files': {METHOD_PATH: SOURCE}},
        cases={'case_000': {'confirmation_seed': 42}, 'case_001': {'confirmation_seed': 43}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(max_model_calls=max_model_calls, observation_budget=1152),
        prepare_trial=prepare_trial)


def test_no_progress_practice_returns_to_author_before_its_next_method_edit(tmp_path):
    from random import Random
    from scripts.experimental_design_control import ExperimentalDesignMethodAgent
    from hegel_machine.cognition.code_provider_context import _compact_practice_feedback

    stuck = "def select_actions(spec, records, memory): raise ValueError('unchanged failure')\n"
    revised = "def select_actions(spec, records, memory): return {'stop':'inspect paid evidence'}\n"

    def prepare(case, path, source, limits):
        lab = AnonymousHistoryLab(kind='memoryless', seed=42)
        agent = ExperimentalDesignMethodAgent(None, path, arm='hegel', source_task=False,
            knowledge={'files': {METHOD_PATH: SOURCE}}, config=limits)
        agent._execute('query', {'actions': lab.confirmation_actions(72, Random(43))}, lab)
        agent.bind_method(source)
        return {'agent': agent, 'laboratory': lab, 'inherited_candidates': []}

    class Provider:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 2:
                feedback = request['events'][-1]['feedback']['edit_practice']
                assert feedback['status'] == 'method_no_progress'
                assert len(feedback['steps']) == 2
                assert feedback['costs']['observation_cost'] == 72
                assert feedback['fresh_confirmation'] is None
                assert feedback['no_progress']
                compact = _compact_practice_feedback(feedback, ['feedback'])
                assert compact['no_progress'] == feedback['no_progress']
            return {'action': 'write', 'arguments': {'path': METHOD_PATH,
                'code': stuck if self.calls == 1 else revised}, 'usage': {'total_tokens': 1}}

    provider = Provider()
    learner = author(tmp_path, provider=provider, prepare_trial=prepare)
    result = learner.run()
    assert provider.calls == 2
    assert [t['status'] for t in result['practice_trials']] == ['method_no_progress', 'method_stopped']
    assert result['practice_trials'][0]['no_progress']
    assert (tmp_path / 'trials/trial_000' / METHOD_PATH).read_text() == stuck
    assert (tmp_path / METHOD_PATH).read_text() == revised
    assert result['costs']['observation_cost'] == 144
    assert result['model'] is None


def save_support(learner, case, result, *, source=SOURCE, reused=False):
    directory = f'trials/trial_{len(learner.practice_trials):03d}'
    path = learner.workspace / directory
    (path / METHOD_PATH).parent.mkdir(parents=True)
    (path / METHOD_PATH).write_text(source)
    (path / 'result.json').write_text(json.dumps(result))
    learner.practice_trials.append({'case': case, 'directory': directory,
        'method_source': source, 'status': result['status'], 'costs': result['costs'],
        'selected_prior_predictor': reused, 'fresh_confirmation': result['fresh_confirmation']})


def test_prediction_publication_remains_supported_but_author_continues_same_account(tmp_path):
    class Provider:
        def next_action(self, request):
            return {'action': 'finish', 'arguments': {'path': METHOD_PATH},
                    'usage': {'total_tokens': 1}}

    learner = author(tmp_path, provider=Provider())
    save_support(learner, 'case_000', prediction_result(), reused=True)
    save_support(learner, 'case_001', prediction_result())
    before = deepcopy(learner.costs)

    result = learner.run()

    assert result['status'] == 'model_call_budget_exhausted'
    assert result['method_source'] is result['model'] is None
    assert learner.published_source is None
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 2
    for name in ('observation_cost', 'measurement_cost', 'compute_cost', 'python_execution_seconds'):
        assert result['costs'][name] == before[name]
    assert len(result['practice_trials']) == 2
    assert len(result['predictive_publications']) == 1
    assert result['predictive_publication'] == result['predictive_publications'][0]
    assert not result['design_objective']['supported']
    assert result['design_objective']['task_status'] == 'prediction_supported_design_unresolved'
    for event in result['events']:
        assert event['feedback']['task_status'] == 'prediction_supported_design_unresolved'
        assert 'published' not in event['feedback']
        assert event['feedback']['current_source_coverage']['confirmed_cases'] == ['case_000', 'case_001']
    report = json.loads((tmp_path / FEEDBACK_FILE).read_text())
    assert report == json.loads(learner.files[FEEDBACK_FILE])
    assert report['design_supported_cases'] == []
    assert all(item['prediction_supported'] for item in report['practices'])


@pytest.mark.parametrize('both_refuted', [False, True])
def test_one_actual_design_chain_can_publish_with_direct_reuse_on_other_case(tmp_path, both_refuted):
    learner = author(tmp_path)
    save_support(learner, 'case_000', prediction_result(), reused=True)
    save_support(learner, 'case_001', design_result(both_refuted=both_refuted))

    feedback = learner._execute('finish', {'path': METHOD_PATH}, learner.interface)

    assert feedback['published'] == METHOD_PATH
    assert feedback['task_status'] == 'prediction_and_design_supported'
    assert feedback['experimental_design']['design_supported_cases'] == ['case_001']
    assert learner.model == METHOD_PATH and learner.published_source == SOURCE
    assert learner.costs['observation_cost'] == 0
    revision = feedback['experimental_design']['practices'][1]['revisions'][0]
    assert revision['measured_outcome'] == (
        'hypothesis_set_refuted' if both_refuted else 'supported_and_refuted')


def test_other_source_and_unlisted_case_cannot_supply_current_design_support(tmp_path):
    learner = author(tmp_path)
    save_support(learner, 'case_001', design_result(), source=SOURCE + '# old\n')
    save_support(learner, 'old_case', design_result())
    save_support(learner, 'case_000', prediction_result(), reused=True)
    save_support(learner, 'case_001', prediction_result())

    feedback = learner._execute('finish', {'path': METHOD_PATH}, learner.interface)

    assert feedback['task_status'] == 'prediction_supported_design_unresolved'
    assert feedback['experimental_design']['design_supported_cases'] == []
    assert [item['case'] for item in feedback['experimental_design']['practices']] == ['case_000', 'case_001']


def test_inconclusive_family_readout_can_record_refuted_instances_before_success():
    result = design_result(both_refuted=True)
    result['revision_proposals'][0]['authored_answer']['status'] = 'inconclusive'

    feedback = design_practice_feedback(result)

    assert feedback['design_supported']
    assert feedback['revisions'][0]['readout_recorded']
    assert 'readout_consumed' not in feedback['revisions'][0]
    assert feedback['revisions'][0]['subsequent_submission'] == {
        'position': 2, 'step': 2, 'model': 'revised_after_probe.py'}
    assert not feedback['revisions'][0]['authored_readout']  # Older report has a stricter label.


def test_restored_paid_feedback_renames_recorded_readout_without_reopening_result(tmp_path):
    learner = author(tmp_path)
    save_support(learner, 'case_001', design_result(both_refuted=True))
    learner._design_feedback()
    trial = learner.practice_trials[0]
    revision = trial['experimental_design']['revisions'][0]
    revision['readout_consumed'] = revision.pop('readout_recorded')
    (tmp_path / trial['directory'] / 'result.json').unlink()

    report = learner._design_feedback()

    assert report['design_supported_cases'] == ['case_001']
    restored = report['practices'][0]['revisions'][0]
    assert restored['readout_recorded']
    assert 'readout_consumed' not in restored
    assert 'readout_consumed' not in revision
    assert len(learner.practice_trials) == 1
    assert learner.costs['observation_cost'] == 0


@pytest.mark.parametrize('change', ['unsupported', 'different_acquisition', 'uncited', 'late_readout', 'failed'])
def test_unfulfilled_scientific_link_is_reported_without_false_design_support(change):
    result = design_result()
    revision = result['revision_proposals'][0]
    if change == 'unsupported':
        revision['prior_fit'][0]['mse'] = .1
    elif change == 'different_acquisition':
        result['experiment_comparisons'][0]['selected_experiment']['actions'] = []
    elif change == 'uncited':
        revision['authored_answer']['evidence'] = [0]
    elif change == 'late_readout':
        result['events'][1]['feedback']['revision'] = deepcopy(revision)
        result['events'][1]['feedback']['revision'].pop('analysis_is_agent_authored')
        result['events'].append({'step': 3, 'action': 'run', 'arguments': {},
                                 'feedback': {'revision': revision}})
    else:
        result['model'] = None
        result['status'] = 'confirmation_failed'
        result['fresh_confirmation']['passed'] = False

    feedback = design_practice_feedback(result)

    assert not feedback['design_supported']
    assert feedback['missing'] and feedback['revisions'][0]['missing']


def test_practice_adds_real_trace_feedback_without_reexecution_or_double_charging(tmp_path):
    calls = []

    def prepare(case, workspace, source, limits):
        workspace.mkdir(parents=True)

        def execute(lab, max_steps):
            calls.append(case)
            return design_result()

        return {'agent': SimpleNamespace(workspace=workspace,
                    confirmation_rng=SimpleNamespace(seed=lambda _: None), execute_method=execute),
                'laboratory': None, 'inherited_candidates': []}

    learner = author(tmp_path, prepare_trial=prepare)
    feedback = learner._practice('case_001')

    assert len(calls) == len(learner.practice_trials) == 1
    assert feedback['experimental_design']['design_supported']
    for name, value in prediction_result()['costs'].items():
        assert learner.costs[name] == value
    assert learner.practice_trials[0]['experimental_design']['design_supported']
    assert json.loads((tmp_path / FEEDBACK_FILE).read_text())['design_supported_cases'] == ['case_001']
