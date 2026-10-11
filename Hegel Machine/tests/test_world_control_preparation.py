"""A prepared reference experiment must execute and recover by observation."""

import json

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from scripts import run_response_matched_adoption as matched
from test_world_reference_hypothesis import PATH, prepared
from test_world_response_controller import BoundedReferenceLab


class SaturatingLab(BoundedReferenceLab):
    def act(self, action):
        result = super().act(action)
        if action.get('chosen_dialog_option_int') not in (None, 1):
            n = self.dialog_number
            self.frequencies[n] = min(10000., max(0., self.frequencies[n]))
            result['observation'] = self.public_observation()
        return result


class NarrowSaturatingLab(SaturatingLab):
    def active(self, number):
        return number in self.installed and abs(self.frequencies[number] - self.required[number]) < .1


def setup_trial(tmp_path, laboratory):
    agent, _ = prepared(tmp_path)
    lab = laboratory(intercept=5.8)
    agent.lab = lab
    agent._execute('query', {'actions': [{'action': 'TALK', 'arg1': 101}]}, lab)
    question = 'prepared_reference_physical'
    agent._question({'id': question, 'question': 'Does the reference activate at 15?', 'parent_id': 'task'})
    state = {'completed': [], 'rejected': {}, 'target_numbers': [4, 5],
        'response_research': {'hypotheses': []}, 'test': {
            'number': 1, 'family': 'conditional_response', 'prediction': 15.,
            'started': 1, 'crystal': 201, 'reactor': 101, 'question': question,
            'response_question': 'prepared_reference', 'restore_frequency': 15.8,
            'control_preparation': {'boundary': 0.,
                'grid': {'origin': 0., 'step': 1., 'lower': 0., 'upper': 10000.},
                'restore_settings': [16., 15.]}}}
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))
    return agent, lab


@pytest.mark.parametrize('laboratory,restored', [(SaturatingLab, True), (NarrowSaturatingLab, False)])
def test_preparation_buys_new_grid_evidence_and_checks_reference_recovery(tmp_path, laboratory, restored):
    agent, lab = setup_trial(tmp_path, laboratory)
    for _ in range(16):
        result = agent._run_procedure(lab, PATH, max_steps=1)
        state = json.loads((agent.workspace / recipe.STATE).read_text())
        if result['status'] == 'method_stopped' or state.get('setting_recoveries'):
            break
    assert agent.costs['model_calls'] == 0
    trial = state['tests'][0]
    assert trial['status'] == 'refuted'
    assert trial['preparation_evidence']
    boundary_record = agent.observations[trial['preparation_evidence'][0]]
    assert recipe._dialog(recipe._view(boundary_record)) == (1, 0.)
    question = next(q for q in agent.questions if q['id'] == trial['question'])
    assert question['answer']['actual_frequency'] == 15.
    recovery = state['setting_recoveries'][-1]
    assert recovery['actual_frequency'] == 16.
    assert recovery['original_frequency'] == 15.8
    assert recovery['restored'] is restored
    assert lab.active(1) is restored
    assert all(r['crystal'] == 1 for r in state['response_settings'])
    if restored:
        assert 'prepared_recovery' not in state
        assert state['control_grids']['1']['origin'] == 0.
    else:
        final = agent._run_procedure(lab, PATH, max_steps=1)
        assert final['status'] == 'method_stopped'
        assert 'no confirmed reference recovery' in final['steps'][-1]['proposal']['stop']
        assert not lab.success


def test_failed_boundary_prediction_does_not_invent_a_rebased_measurement(tmp_path):
    # A documented range alone does not guarantee saturation by the implementation.
    agent, lab = setup_trial(tmp_path, BoundedReferenceLab)
    agent._run_procedure(lab, PATH, max_steps=1)
    result = agent._run_procedure(lab, PATH, max_steps=1)
    assert result['status'] == 'method_stopped'
    assert 'proposed boundary preparation' in result['steps'][-1]['proposal']['stop']
    state = json.loads((agent.workspace / recipe.STATE).read_text())
    assert 'preparation_evidence' not in state['test']
    assert not state.get('tests')


def test_archived_export_carries_preparation_without_replacing_other_authored_steps():
    import ast
    from pathlib import Path
    source = Path(recipe.__file__).read_text()
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    removed = [nodes['_prepared_reference_step']]
    hook = next(node for node in nodes['select_actions'].body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'prepared_step' for t in node.targets))
    removed.extend([hook, nodes['select_actions'].body[nodes['select_actions'].body.index(hook) + 1]])
    removed.append(next(node for node in ast.walk(nodes['_trial_action']) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "selected.get('control_preparation')"))
    lines = source.splitlines(keepends=True)
    for node in sorted(removed, key=lambda node: node.lineno, reverse=True):
        del lines[node.lineno - 1:node.end_lineno]
    source = ''.join(lines)
    source = source.replace('    view = public_spec[\'current_observation\']',
        "    if memory.get('author_stop'):\n        return {'stop': 'Keep authored branch'}\n"
        "    view = public_spec['current_observation']", 1)
    bundle = {'files': {PATH: source}, 'methods': []}
    exported = matched.prepare_bundle(bundle)
    assert matched.prepare_bundle(exported) == exported
    assert bundle['files'][PATH] == source
    code = exported['files'][PATH]
    assert "'Keep authored branch'" in code
    nodes = {node.name: ast.get_source_segment(code, node) for node in ast.parse(code).body
             if isinstance(node, ast.FunctionDef)}
    assert '_prepared_reference_step(state, view, observations, questions)' in nodes['select_actions']
    assert "state['test']['control_preparation']" in nodes['_trial_action']
    assert 'control_preparation' in nodes['_extend_hypotheses']
