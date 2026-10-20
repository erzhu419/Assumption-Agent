"""Read the real frozen publication pair; retain no-submission failures and fees."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition import method_effect_query
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.method_effect_query import PUBLISHED_EFFECT_KIND, evaluate_method_effect
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / 'docs/evidence/conditional_online_failure_20261013/window_parameter_transfer_20261017.json'


def query(file=REGISTRATION, **fields):
    return {'op': 'MethodEffect', 'effect_kind': PUBLISHED_EFFECT_KIND,
        'comparison_ref': {'file': str(file), 'json_path': ['execution', 'final_outcome']},
        'left_arm': 'baseline_13of13', 'right_arm': 'published_14of14', **fields}


@pytest.fixture(scope='module')
def real_effect(tmp_path_factory):
    result = evaluate_method_effect(query(), tmp_path_factory.mktemp('published_effect'))
    if result['status'] == 'scoped_unknown' and 'No such file' in result.get('cause', ''):
        pytest.skip('The retained frozen publication source is not mounted')
    return result


def test_real_frozen_publication_pair_preserves_all_eight_outcomes_missing_mse_and_complete_costs(real_effect):
    assert real_effect['status'] == 'answered', real_effect
    value = real_effect['value']
    assert value['intervention'] == 'frozen_published_method_package'
    assert value['left']['case_count'] == value['right']['case_count'] == len(value['cases']) == 8
    assert [value['left'][key] for key in ('submitted', 'confirmation_passed', 'terminal_passed')] == [0, 0, 0]
    assert [value['right'][key] for key in ('submitted', 'confirmation_passed', 'terminal_passed')] == [8, 8, 8]
    assert value['left']['costs']['observation_cost'] == 16960
    assert value['right']['costs']['observation_cost'] == 6464
    assert value['left']['costs']['compute_cost'] == 0
    assert value['left']['costs']['python_execution_seconds'] > 23000
    assert value['right_minus_left']['costs']['observation_cost'] == -10496
    assert value['right_minus_left']['costs']['compute_cost'] == 6464
    for case in value['cases']:
        assert case['left']['status'] == 'method_stopped' and case['left']['model'] is None
        assert case['left']['terminal'] == {'status': 'no_submitted_model', 'mse': None, 'passed': False}
        assert case['right']['status'] == 'completed' and case['right']['terminal']['passed']
        assert case['right_minus_left']['terminal_mse']['status'] == 'scoped_unknown'
        assert case['left']['costs']['model_calls'] == case['right']['costs']['model_calls'] == 0
    scope = real_effect['scope']
    assert scope['independent_parameter_groups'] == 4 and scope['conditions_per_group'] == 2
    assert scope['target_runner'] == 'ExperimentalDesignMethodAgent' and scope['prefix_rows'] == 72
    assert len(scope['executed_method_refs']) == 16 and len(scope['same_initial_catalog_refs']) == 16
    assert scope['methods_include_publication_support_provenance'] and scope['literal_source_bytes_only'] is False
    assert scope['general_method_superiority'] == 'not_established'
    lineage = value['source_cost_lineage']
    assert lineage['shared_prior_author_phase_refs'] and lineage['phase_accounts_summed'] is False
    assert lineage['nonidentical_phases_are_independent'] == 'not_established'
    assert real_effect['compute_cost'] == real_effect['new_method_executions'] == 0
    assert real_effect['independent_confirmation'] is False


def test_reversed_published_pair_reverses_coverage_and_cost_not_missing_baseline_mse(tmp_path):
    result = evaluate_method_effect(query(left_arm='published_14of14', right_arm='baseline_13of13'), tmp_path)
    assert result['status'] == 'answered', result
    assert result['value']['right_minus_left']['submitted'] == -8
    assert result['value']['right_minus_left']['costs']['observation_cost'] == 10496
    assert all(case['right_minus_left']['terminal_mse']['status'] == 'scoped_unknown'
               for case in result['value']['cases'])
    phase = result['value']['source_cost_lineage']['shared_prior_author_phase_refs'][0]
    assert phase['baseline_13of13']['file'].endswith('/baseline_13of13/protocol.json')
    assert not phase['published_14of14']['file'].endswith('/baseline_13of13/protocol.json')
    assert evaluate_method_effect(query(outcome='literal_source_only_effect'), tmp_path)['status'] == 'scoped_unknown'


@pytest.mark.parametrize('changed', ['budget', 'missing_failure', 'catalog'])
def test_actual_shared_contract_or_failure_loss_prevents_a_published_effect_answer(tmp_path, monkeypatch, changed):
    original = method_effect_query.load_source_values

    def altered(references, workspace):
        loaded = original(references, workspace)
        for item in loaded:
            path = item['source_ref'].get('file', '')
            if changed == 'budget' and path.endswith('/baseline_13of13/protocol.json'):
                item['value']['limits']['execution_seconds'] += 1
            elif changed == 'missing_failure' and path.endswith('/baseline_13of13/summary.json'):
                item['value']['new_parameters'].pop()
            elif changed == 'catalog' and '/baseline_13of13/frozen/' in path and path.endswith('/source_catalog.json'):
                item['value'] = []
        return loaded

    monkeypatch.setattr(method_effect_query, 'load_source_values', altered)
    result = evaluate_method_effect(query(), tmp_path)
    assert result['status'] == 'scoped_unknown' and 'value' not in result
    assert result['detail'] == 'published_method_transfer_contract_unavailable'


def test_a_changed_executed_method_source_is_not_hidden_by_the_saved_publication(tmp_path, monkeypatch):
    original = Path.read_text

    def altered(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if str(path).endswith('/baseline_13of13/frozen/window_parameter_target_000/shared/consolidated_method.py'):
            return text + '\n# Changed after the recorded publication.\n'
        return text

    monkeypatch.setattr(Path, 'read_text', altered)
    result = evaluate_method_effect(query(), tmp_path)
    assert result['status'] == 'scoped_unknown'
    assert result['cause'] == 'executed_frozen_method_differs_from_publication'


def test_Common_published_effect_reads_actual_pair_and_withdraws_answer_when_reference_disappears(tmp_path):
    alias = tmp_path / 'registered_window_pair.json'
    alias.symlink_to(REGISTRATION)
    agent = CommonArchiveInquiryAgent(None, tmp_path / 'workspace', arm='hegel', source_task=False,
        config=InquiryLLMConfig(compute_budget=10000, observation_budget=8,
            max_model_calls=0, execution_seconds=30.))
    agent.research_mission = {'purpose': 'method_learning', 'mission_id': 'published_window_transfer'}

    class ReadOnlyLab:
        def public_spec(self):
            return {'controls': []}

        def execute(self, action):
            raise AssertionError('MethodEffect executed a new method or acquisition')

    lab = ReadOnlyLab()
    question = {'id': 'published_method_effect', 'question': 'What happened in the registered frozen publication pair?',
        'parent_id': 'task', 'depends_on': [], 'kind': 'method_effect', 'answer_type': 'relation',
        'answer_resolution_ref': {'kind': 'registered_target_coverage_and_cost'},
        'query_semantics': {'mode': 'model_query', 'query_expression': query(alias)}, 'use_sites': []}
    costs = deepcopy(agent.costs)

    def step(declaration):
        return agent._execute('run', {'operator': 'research_step', 'research_question': declaration}, lab)

    first = step(question)
    assert first['status'] == 'answer_ready_no_parent_use'
    assert first['answer']['value']['right_minus_left']['terminal_passed'] == 8
    assert first['parent_revision_completed'] is False
    assert step({'id': question['id']})['answer'] == first['answer']
    alias.unlink()
    missing = step({'id': question['id']})
    assert missing['status'] == 'scoped_unknown'
    current = next(item for item in agent.questions if item['id'] == question['id'])
    assert 'semantic_answer' not in current and current['query_execution']['answer']['status'] == 'scoped_unknown'
    assert agent.costs == costs and agent.model is None and not agent.confirmations
