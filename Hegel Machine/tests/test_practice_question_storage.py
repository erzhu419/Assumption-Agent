"""Practice banks live once in state; result metadata remains the latest view."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.anonymous_method_learning import SingleSubmissionMethodAgent
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_resume import CONFIG, SEED
from test_temporal_prospective_inputs import trajectory


MARKER = 'unique_native_recipe_' + 'x' * 20000


def practice(path):
    return SingleSubmissionMethodAgent(None, path, arm='hegel', source_task=False,
        config=CONFIG, confirmation_seed=SEED, knowledge={'files': {}})


def saved_practice(path):
    agent, lab = practice(path), TemporalInquiryLab(task_index=1)
    agent._execute('query', {'actions': [trajectory(0.)]}, lab)
    target = agent.rows[0].target
    agent._save('model.py', f'def predict(row): return {target!r}\n')
    evaluated = agent._execute('evaluate', {'model': 'model.py'}, lab)
    model = evaluated['evaluated_model']['model']
    agent.questions[0]['native_condition_research'] = [{
        'round_index': 0, 'status': 'supported', 'fit_evidence': [0],
        'exported_models': [{'model': model, 'fit_mse': 0.}],
        'candidates': [{'model': model, 'research_program': {'recipe': MARKER}}]}]
    agent._prepare_program_state()
    # The saved bank is current while this later question metadata is not in state.
    agent.questions[0].update(status='reopened', evidence=[0],
        reviewed_child_answers=[{'question_id': 'later_review', 'answer': 'latest'}])
    return agent, lab, model


def test_matching_practice_bank_is_referenced_and_restores_latest_metadata_and_model(tmp_path):
    old, lab, model = saved_practice(tmp_path / 'old')
    previous = old.result('method_stopped')
    reference = previous['online_questions'][0]['native_condition_research']
    assert reference == {'full_content_file': 'research_state.json',
        'json_path': ['questions', 0, 'native_condition_research'], 'round_count': 1}
    assert previous['outgoing_knowledge']['question_traces'][0]['native_condition_research'] == reference
    assert MARKER not in json.dumps(previous)
    state = json.loads((old.workspace / 'research_state.json').read_text())
    assert state['questions'][0]['status'] != previous['online_questions'][0]['status']
    assert json.dumps(state).count(MARKER) == 1
    assert old.questions[0]['native_condition_research'][0]['candidates'][0][
        'research_program']['recipe'] == MARKER

    # A subsequent step can append a round before its next checkpoint is saved.
    later = deepcopy(state['questions'][0]['native_condition_research'][0])
    later['round_index'] = 1
    later['candidates'][0]['research_program']['recipe'] = 'later round outside this checkpoint'
    state['questions'][0]['native_condition_research'].append(later)
    (old.workspace / 'research_state.json').write_text(json.dumps(state))

    resumed, target = practice(tmp_path / 'resumed'), TemporalInquiryLab(task_index=1)
    restored = resumed.restore_practice_research(previous, old.workspace, target,
                                               confirmation_seed=SEED)
    assert restored['new_observations'] == 0 and resumed.costs == old.costs
    assert target.costs == lab.costs and resumed.rows == old.rows
    assert resumed.questions == old.questions
    assert previous['online_questions'][0]['native_condition_research'] == reference
    # The candidate named by the restored bank still executes on the retained fit.
    assert evaluate_python_model(resumed.workspace, model, resumed.rows)['mse'] == 0


@pytest.mark.parametrize('state_kind', ['older', 'different_content', 'missing_question'])
def test_state_without_identical_bank_preserves_inline_unique_rounds_on_restore(tmp_path, state_kind):
    old, _, _ = saved_practice(tmp_path / 'old')
    state_path = old.workspace / 'research_state.json'
    state = json.loads(state_path.read_text())
    if state_kind == 'older':
        state['questions'][0]['native_condition_research'] = []
    elif state_kind == 'different_content':
        state['questions'][0]['native_condition_research'][0]['candidates'][0][
            'research_program']['recipe'] = 'a different saved recipe with the same round count'
    else:
        state['questions'] = []
    state_path.write_text(json.dumps(state))
    previous = old.result('method_stopped')
    bank = deepcopy(old.questions[0]['native_condition_research'])
    assert previous['online_questions'][0]['native_condition_research'] == bank
    assert previous['outgoing_knowledge']['question_traces'][0]['native_condition_research'] == bank
    resumed = practice(tmp_path / 'resumed')
    resumed.restore_practice_research(previous, old.workspace, TemporalInquiryLab(task_index=1),
                                     confirmation_seed=SEED)
    assert resumed.questions == old.questions
