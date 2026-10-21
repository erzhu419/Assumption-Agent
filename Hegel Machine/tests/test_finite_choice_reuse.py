"""Reuse complete same-scope comparisons after actual zero-output operators."""
from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.finite_native_author import export_finite_native_author, select_actions
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent, ORGANIZATION_ORIGIN
from test_finite_native_author import ChoiceLab, POLICY, declaration
from test_inquiry_revision import action
from test_registered_finite_policy import run_formal_steps


def choice_scene(tmp_path):
    library = tmp_path / 'empty.json'
    library.write_text('{"candidates":{}}')
    contract = declaration()
    contract.update(max_compute=0, predictor_paths=['archive/bad.py', 'archive/worse.py'])
    files = {POLICY: export_finite_native_author(), 'archive/bad.py': 'def predict(row): return 10.\n',
             'archive/worse.py': 'def predict(row): return 20.\n'}
    agent = RegisteredOrganizationAgent(None, tmp_path / 'work', arm='fixed_pr2', source_task=False,
        research_organization={'phase': 'target', 'initial_library_path': str(library)},
        knowledge={'files': files, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=50, compute_budget=3000,
            confirmation_rows=2, execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(1.), action(2.)]}, lab)
    return agent, lab


def test_registered_refit_unavailable_and_empty_native_round_do_not_rescore_same_complete_pool(tmp_path, monkeypatch):
    agent, lab = choice_scene(tmp_path)
    def no_acquisition(*args, **kwargs):
        raise AssertionError('The cached comparison needs no further fixture labels')
    monkeypatch.setattr(lab, 'execute', no_acquisition)
    before = deepcopy(agent.costs)
    result = run_formal_steps(agent, lab, 4)
    proposals = [step['proposal'] for step in result['steps']]
    operators = [item['operator'] for item in agent.operator_results]
    assert operators[:3] == ['research_choice', 'refit', 'native_condition_research'], {
        'operators': operators, 'proposals': proposals}
    assert operators.count('research_choice') == 1
    assert agent.operator_results[1]['status'] == 'refit_unavailable'
    native = agent.operator_results[2]['native_condition_research']
    assert native['candidates'] == native['exported_models'] == []
    assert proposals[3].get('arguments', {}).get('operator') != 'research_choice'
    assert agent.method_results[0]['feedback']['research_question_id'] == declaration()['question']['id']
    assert agent.method_results[0]['feedback']['question_meaning_version'] == 1
    assert agent.operator_results[0]['arguments']['models'] == ['archive/bad.py', 'archive/worse.py']
    assert len(agent.operator_results[0]['evaluations']) == 2
    assert agent.operator_results[0]['compute_cost'] == 4
    assert agent.evaluated_model['source_model'] == 'archive/bad.py'
    assert agent.costs['observation_cost'] == before['observation_cost'] == 2
    assert agent.costs['model_calls'] == 0
    assert len([event for event in agent.events if event.get('origin') == ORGANIZATION_ORIGIN]) >= 1


@pytest.mark.parametrize('change', ['rows', 'meaning', 'validated_export'])
def test_actual_new_evidence_meaning_or_validated_export_requires_a_new_comparison(tmp_path, change):
    agent, lab = choice_scene(tmp_path)
    first = run_formal_steps(agent, lab, 1)
    assert first['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    prior = deepcopy(agent.operator_results[0])
    contract = agent.knowledge['finite_native_author']
    if change == 'rows':
        agent._execute('query', {'actions': [action(2.)]}, lab)
    elif change == 'meaning':
        updated = deepcopy(contract['question'])
        updated['context_ref'] = {**updated['context_ref'], 'scope': 'revised_public_task'}
        updated['theory_target']['context_ref'] = deepcopy(updated['context_ref'])
        current = agent._open_question(updated)
        assert current['meaning_version'] == 2
        contract['question'] = deepcopy(current)
    else:
        agent._execute('write', {'path': 'refittable.py', 'code':
            'gain = 20.\ndef predict(row): return gain * row["features"]["u"]\n'
            'def fit(rows):\n    global gain\n    gain = rows[0]["target"] / rows[0]["features"]["u"]\n'}, lab)
        refit = agent._execute('run', {'operator': 'refit', 'model': 'refittable.py',
            'path': 'shared/validated_new'}, lab)
        assert refit['status'] == 'refit_supported_on_validation'
        assert refit['path'] in agent.candidate_paths
    result = run_formal_steps(agent, lab, 1)
    proposal = result['steps'][0]['proposal']
    assert proposal['arguments']['operator'] == 'research_choice'
    choices = [item for item in agent.operator_results if item['operator'] == 'research_choice']
    assert len(choices) == 2 and agent.operator_results[0] == prior
    expected_rows = 3 if change == 'rows' else 2
    assert choices[-1]['paid_rows'] == expected_rows
    if change == 'validated_export':
        assert refit['path'] in proposal['arguments']['models']
        assert len(choices[-1]['evaluations']) == 3
    if change == 'meaning':
        assert agent.method_results[-1]['feedback']['question_meaning_version'] == 2


def test_budget_interrupted_comparison_is_not_reused_as_complete(tmp_path, monkeypatch):
    agent, lab = choice_scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=3)
    result = run_formal_steps(agent, lab, 1)
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    assert 'candidate-row scoring budget exhausted' in agent.operator_results[-1]['error']
    assert agent.costs['compute_cost'] == 2
    monkeypatch.chdir(agent.workspace)
    memory = agent._program_memory(POLICY, {'path': POLICY})
    proposal = select_actions(lab.public_spec(), agent.observations, memory)
    assert proposal['arguments']['operator'] == 'research_choice'
    assert agent.costs['compute_cost'] == 2
