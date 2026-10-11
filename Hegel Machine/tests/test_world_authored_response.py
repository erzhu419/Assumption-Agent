"""Author-selected setpoints use the same physical response controller."""

import json
import math

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.world_reference_hypothesis import HYPOTHESIS_INTERFACE
from test_world_reference_hypothesis import PATH, prepared
from test_world_response_controller import BoundedReferenceLab, ImmediateCompletionLab


def _ready(tmp_path, *, lab_type=BoundedReferenceLab, intercept=100.25):
    agent, _ = prepared(tmp_path, intercept=intercept)
    lab = lab_type(intercept=intercept)
    agent.lab = lab
    for number in range(1, 6):
        agent._execute('query', {'actions': [
            {'action': 'TALK', 'arg1': 100 + number}, {'chosen_dialog_option_int': 1}]}, lab)
    return agent, lab


def _experiment(agent, lab, number, setting):
    return agent._execute('run', {'operator': 'reference_response', 'path': PATH,
                                 'experiment': {'number': number, 'setting': setting}}, lab)


def test_authored_shifted_setting_uses_feedback_instead_of_action_list(tmp_path):
    agent, lab = _ready(tmp_path)
    lab.frequencies[4] = 312.25
    agent._execute('query', {'actions': [{'action': 'TALK', 'arg1': 104},
                                        {'chosen_dialog_option_int': 1}]}, lab)
    before = len(lab.actions)
    result = _experiment(agent, lab, 4, 1012.25)

    assert result['status'] == 'method_stopped'
    assert result['prediction_status'] == 'refuted'
    assert result['requested_setting_reached']
    assert result['answer']['actual_frequency'] == pytest.approx(1012.25)
    assert not result['answer']['activated']
    assert lab.frequencies[4] == pytest.approx(1012.25)
    actions = lab.actions[before:]
    assert actions.count({'chosen_dialog_option_int': 2}) == 1
    assert actions.count({'chosen_dialog_option_int': 7}) == 3
    assert len(actions) == 8  # pickup, put, talk, +1000 and three -100 steps, exit
    assert result['new_observation_indices']
    state = json.loads((agent.workspace / recipe.STATE).read_text())
    assert state['rejected'] == {}  # a chosen point does not exclude a scientific family
    assert 'experiment:{number:PUBLIC_CRYSTAL_NUMBER' in HYPOTHESIS_INTERFACE
    assert agent.costs['model_calls'] == 0


def test_ordinary_python_transforms_paid_readings_and_controller_tests_prediction(tmp_path):
    class RootResponseLab(BoundedReferenceLab):
        def __init__(self, *, intercept):
            super().__init__(intercept=intercept)
            self.required = {n: intercept + 100 * math.sqrt(value)
                             for n, value in self.density.items()}
            self.frequencies.update({n: self.required[n] for n in (1, 2, 3)})

    agent, lab = _ready(tmp_path, lab_type=RootResponseLab, intercept=100.)
    agent._execute('query', {'actions': [
        {'action': 'USE', 'arg1': 91, 'arg2': 200 + n} for n in (1, 2, 3, 4)]}, lab)
    paid_count = len(agent.observations)
    source_before = (agent.workspace / PATH).read_text()
    calculation = agent._execute('run', {'code': r'''
import json, math, re
rows = json.load(open('laboratory_observations.json'))
measurements, references, evidence = {}, {}, []
for row in rows:
    action = row['action']
    if action.get('action') == 'USE' and action['arg1'] == 91:
        measurements[action['arg2'] - 200] = float(re.search(
            r'density is approximately ([\d.]+)', row['observation']['lastActionMessage'])[1])
        evidence.append(row['index'])
    if action.get('action') == 'TALK' and action['arg1'] in (101, 102, 103):
        references[action['arg1'] - 100] = float(re.search(
            r'frequence is: ([\d.]+)', row['observation']['dialog_box']['dialogIn'])[1])
        evidence.append(row['index'])
xs = [math.sqrt(measurements[n]) for n in sorted(references)]
ys = [references[n] for n in sorted(references)]
xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x-xmean)*(y-ymean) for x,y in zip(xs,ys)) / sum((x-xmean)**2 for x in xs)
intercept = ymean - slope*xmean
derived = math.sqrt(measurements[4])
prediction = intercept + slope*derived
print(json.dumps({'raw_feature': measurements[4], 'derived_feature': derived,
    'prediction': prediction, 'setting': round(prediction),
    'reference_residuals': [intercept + slope*x - y for x,y in zip(xs,ys)],
    'evidence': evidence}))
'''}, lab)

    assert calculation['returncode'] == 0, calculation['stderr']
    prediction = json.loads(calculation['stdout'])
    assert prediction['raw_feature'] == 4 and prediction['derived_feature'] == 2
    assert prediction['reference_residuals'] == pytest.approx([0., 0., 0.], abs=1e-10)
    assert prediction['prediction'] == pytest.approx(300.)
    assert len(prediction['evidence']) == 7
    assert len(agent.observations) == paid_count
    # The calculation stays ordinary Python; it need not add an AST estimator.
    assert (agent.workspace / PATH).read_text() == source_before

    result = _experiment(agent, lab, 4, prediction['setting'])

    assert result['prediction_status'] == 'supported' and result['requested_setting_reached']
    assert result['answer']['actual_frequency'] == pytest.approx(prediction['prediction'])
    assert lab.active(4) and not lab.active(5)
    assert min(result['new_observation_indices']) >= paid_count
    assert agent.costs['model_calls'] == 0


def test_unreachable_authored_setting_returns_grid_without_buying_actions(tmp_path):
    agent, lab = _ready(tmp_path)
    lab.frequencies[4] = 312.25
    agent._execute('query', {'actions': [{'action': 'TALK', 'arg1': 104},
                                        {'chosen_dialog_option_int': 1}]}, lab)
    before = len(lab.actions)
    for setting in (1012, 10001.25):
        result = _experiment(agent, lab, 4, setting)
        assert result['status'] == 'unreachable_setting'
        assert result['control_grid']['origin'] == pytest.approx(.25)
        assert len(lab.actions) == before
        assert result['new_observation_indices'] == []


def test_authored_reference_counterexample_restores_and_allows_next_experiment(tmp_path):
    agent, lab = _ready(tmp_path)
    result = _experiment(agent, lab, 1, 111.25)
    assert result['prediction_status'] == 'refuted'
    assert result['answer']['actual_frequency'] == pytest.approx(111.25)
    assert result['setting_recoveries'][-1]['restored']
    assert lab.active(1) and lab.frequencies[1] == pytest.approx(110.25)
    assert any(row['frequency'] == 111.25 and not row['activated'] for row in result['observed_responses'])
    question = next(q for q in agent.questions if q['id'] == result['question_id'])
    assert question['status'] == 'refuted'  # successful restoration is not support for the tested point

    following = _experiment(agent, lab, 4, 260.)
    assert following['prediction_status'] == 'supported'
    assert following['answer']['actual_frequency'] == 260
    assert lab.active(4) and not lab.active(5)
    assert not agent.llm_inquiries and agent.costs['model_calls'] == 0


def test_authored_intermediate_activation_cannot_confirm_unreached_setting(tmp_path):
    agent, lab = _ready(tmp_path, lab_type=ImmediateCompletionLab, intercept=100.)
    lab.installed.update({4, 5})
    lab.frequencies[4] = lab.required[4]
    lab.frequencies[5], lab.required[5] = 4000., 4056.
    result = _experiment(agent, lab, 5, 4055.)
    assert result['status'] == 'completed' and lab.success
    assert lab.frequencies[5] == 4056
    assert result['prediction_status'] == 'inconclusive'
    assert not result['requested_setting_reached']
    assert result['answer']['requested_setting'] == 4055
    assert any(row['frequency'] == 4056 and row['activated'] for row in result['observed_responses'])


@pytest.mark.parametrize('revise', [False, True])
def test_step_limited_trial_resumes_or_revises_without_hidden_old_experiment(tmp_path, revise):
    agent, lab = _ready(tmp_path, intercept=100.)
    arguments = {'operator': 'reference_response', 'path': PATH,
                 'experiment': {'number': 4, 'setting': 1012}, 'max_steps': 4}
    first = agent._execute('run', arguments, lab)
    assert first['status'] == 'method_step_limit'
    assert not first['requested_setting_reached']
    following = _experiment(agent, lab, 4, 260 if revise else 1012)
    assert following['requested_setting_reached']
    if revise:
        assert following['question_id'] != first['question_id']
        assert following['prediction_status'] == 'supported'
        assert following['superseded_experiment']['question'] == first['question_id']
        assert following['superseded_experiment']['status'] == 'inconclusive'
        assert not any(row['frequency'] == 1012 for row in following['observed_responses'])
        old = next(q for q in agent.questions if q['id'] == first['question_id'])
        assert old['status'] == 'inconclusive'
        assert old['evidence']
    else:
        assert following['question_id'] == first['question_id']
        assert following['prediction_status'] == 'refuted'
        assert len([q for q in agent.questions if q['id'].startswith('reference_authored_')]) == 1


@pytest.mark.parametrize('question_key', ['question', 'research_question'])
def test_authored_question_survives_interrupted_controller_and_undeclared_resume(tmp_path, question_key):
    agent, lab = _ready(tmp_path, intercept=100.)
    agent._question({'id': 'response_family', 'parent_id': 'task',
                     'question': 'Does this shared response family explain the target?'})
    prerequisite = agent._question({'id': 'control_reading', 'parent_id': 'response_family',
                                    'question': 'Was the public control menu observed?'})
    agent._record_answer(prerequisite, {'status': 'supported', 'answer': True, 'evidence': [0]})
    declaration = {'id': 'authored_target_prediction', 'parent_id': 'response_family',
                   'question': 'Does the proposed family activate crystal 4 at 1012 Hz?',
                   'depends_on': ['control_reading']}
    existing = {q['id'] for q in agent.questions}
    before = len(agent.observations)

    first = agent._execute('run', {'operator': 'reference_response', 'path': PATH,
        'experiment': {'number': 4, 'setting': 1012}, 'max_steps': 4,
        question_key: declaration}, lab)

    assert first['status'] == 'method_step_limit'
    assert not first['requested_setting_reached']
    assert first['question_id'] == declaration['id']
    assert first['new_observation_indices']
    assert min(first['new_observation_indices']) >= before
    question = next(q for q in agent.questions if q['id'] == declaration['id'])
    assert {key: question[key] for key in declaration} == declaration
    pending = json.loads((agent.workspace / recipe.STATE).read_text())['test']
    assert pending['question'] == declaration['id']

    # The caller can resume the physical point without re-declaring the same
    # scientific question; the controller must finish its original assignment.
    resumed = _experiment(agent, lab, 4, 1012)

    assert resumed['requested_setting_reached']
    assert resumed['prediction_status'] == 'refuted'
    assert resumed['question_id'] == declaration['id']
    assert {key: question[key] for key in declaration} == declaration
    assert question['status'] == 'refuted'
    assert question['evidence']
    assert set(question['evidence']) & set(first['new_observation_indices'] + resumed['new_observation_indices'])
    assert {q['id'] for q in agent.questions} - existing == {declaration['id']}
    assert agent.costs['model_calls'] == 0
