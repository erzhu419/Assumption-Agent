"""Revisit refuted point predictions through shared response hypotheses.

These integration fixtures test controller behavior using the existing public
laboratory, not model-authored discovery or a DiscoveryWorld benchmark.
"""

import ast
from copy import deepcopy
import json

from hegel_machine.cognition import world_reference_recalibration as recipe
from test_world_reference_hypothesis import PATH, ReferenceLab, prepared, square, submit


def _state(agent):
    return json.loads((agent.workspace / recipe.STATE).read_text())


def _questions(agent):
    return {question["id"]: question for question in agent.questions}


class BoundedReferenceLab(ReferenceLab):
    def public_observation(self):
        view = super().public_observation()
        dialog = view["dialog_box"]
        if "dialogIn" in dialog:
            dialog["dialogIn"] += " The allowable range is 0 to 10,000 Hz."
        return view


class ImmediateCompletionLab(BoundedReferenceLab):
    """The environment ends on activation, even partway through tuning."""

    def act(self, action):
        result = super().act(action)
        self.completed = self.success = all(self.active(n) for n in (4, 5))
        result["observation"] = self.public_observation()
        return result


def _submit_point_prefix(agent, lab, hypothesis):
    # These controller tests start at a particular acquired point-refutation
    # prefix. The full declaration entry now continues into response research;
    # isolate only this setup, then restore the actual recipe with its retained
    # declaration before testing unrestricted continuation. The unwrapped entry
    # is covered by test_declaration_continues_refuted_point_into_actual_boolean_family_trials.
    point_prefix = """
def _extend_hypotheses(state, memory, number, path):
    return {'stop': 'Test setup reached the acquired point-refutation prefix'}
"""
    agent._execute('write', {'path': PATH,
        'code': (agent.workspace / PATH).read_text() + point_prefix}, lab)
    first = submit(agent, lab, hypothesis)
    declared_source = (agent.workspace / PATH).read_text()
    assert declared_source.endswith(point_prefix)
    agent._execute('write', {'path': PATH,
        'code': declared_source[:-len(point_prefix)]}, lab)
    return first


def _quantized_extrapolation(tmp_path, *, target_setting=4055.3, immediate=False):
    """One successful rounded label moves the fitted extrapolation too far."""
    agent, _ = prepared(tmp_path, intercept=100.3)
    lab = (ImmediateCompletionLab if immediate else BoundedReferenceLab)(intercept=100.3)
    agent.lab = lab
    lab.density[5] = 395.5 ** .5
    lab.required[5] = target_setting
    first = _submit_point_prefix(agent, lab, square("density", "density_square"))
    assert not lab.success and first["status"] != "completed"
    old = _questions(agent)["reference_test_5_density_square"]
    assert old["status"] == "refuted"
    assert lab.frequencies[4] == 260 and lab.frequencies[5] != 4055
    return agent, lab, deepcopy(old)


def _exhausted_grammar(agent):
    state = _state(agent)
    state["expanded_features"] = ["density", "gap"]
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))


def test_supported_basis_point_failure_enters_response_before_other_point_trials(tmp_path):
    agent, lab, _ = _quantized_extrapolation(tmp_path)
    before = deepcopy(_state(agent))
    assert recipe._point_needs_response(before, 5)
    assert not before.get('expanded_features')
    actions = len(lab.actions)
    result = agent._run_procedure(lab, PATH, max_steps=1)
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'reference_response'
    assert len(lab.actions) == actions
    state = _state(agent)
    assert state['tests'] == before['tests']
    assert len(state['response_research_history']) == 1
    assert not state.get('expanded_features')


def test_response_first_still_acquires_a_subsequently_declared_instrument(tmp_path):
    from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration

    agent, lab, _ = _quantized_extrapolation(tmp_path)
    declaration = square('signal', 'new_signal_square')
    declaration['measurements'] = {'signal': {'instrument': 'signal meter',
        'pattern': r'signal magnitude = ([\d.]+) units', 'group': 1}}
    source = _retain_declaration((agent.workspace / PATH).read_text(), declaration)
    agent._execute('write', {'path': PATH, 'code': source}, lab)
    before = len(lab.actions)
    agent._run_procedure(lab, PATH, max_steps=6)
    assert lab.actions[before:] == [{'action': 'PICKUP', 'arg1': 93}] + [
        {'action': 'USE', 'arg1': 93, 'arg2': 200 + n} for n in range(1, 6)]
    result = agent._run_procedure(lab, PATH, max_steps=1)
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'reference_response'
    table = _state(agent)['research_table']
    assert all('signal' in row['measurements'] for row in table['references'] + table['targets'])


def test_response_search_success_routes_new_target_without_a_shared_formula(tmp_path):
    agent, lab, _ = _quantized_extrapolation(tmp_path)
    before = _state(agent)
    # Reproduce the recorded transition: the solved target's operational search
    # label is deliberately not the failed point predictor's family name.
    successful = next(t for t in before['tests'] if t['number'] == 4 and t['status'] == 'supported')
    successful.update(family='attainable_target_search', response_question='reference_response_4_1')
    (agent.workspace / recipe.STATE).write_text(json.dumps(before))
    actions = len(lab.actions)
    result = agent._run_procedure(lab, PATH, max_steps=1)
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'reference_response'
    assert len(lab.actions) == actions
    state = _state(agent)
    assert state['tests'] == before['tests']
    assert state['response_context']['number'] == 5
    assert state['response_context']['responses'] == [
        [r['crystal'], r['frequency'], r['activated']] for r in state['response_settings']]
    assert not state.get('expanded_features')


def test_recovered_response_target_does_not_confirm_its_refuted_requested_point():
    state = {'target_numbers': [4, 5], 'completed': [4], 'tests': [
        {'number': 4, 'family': 'attainable_target_search', 'status': 'refuted',
         'response_question': 'reference_response_4_8'},
        {'number': 5, 'family': 'signal_cubic_5', 'status': 'refuted'}]}
    before = deepcopy(state)
    assert recipe._point_needs_response(state, 5)
    assert state == before
    state['completed'] = []
    assert not recipe._point_needs_response(state, 5)


def test_recovered_point_target_routes_response_without_confirming_either_point():
    state = {'target_numbers': [4, 5], 'completed': [4], 'tests': [
        {'number': 4, 'family': 'density_square', 'status': 'refuted'},
        {'number': 5, 'family': 'signal_cubic_5', 'status': 'refuted'}]}
    before = deepcopy(state)
    assert recipe._point_needs_response(state, 5)
    assert state == before
    for change in ({'status': 'supported'}, {'authored_experiment': True},
                   {'response_question': 'reference_response_5_1'}):
        altered = deepcopy(state)
        altered['tests'][1].update(change)
        assert not recipe._point_needs_response(altered, 5)
    for completed in ([], [1], [5]):
        state['completed'] = completed
        assert not recipe._point_needs_response(state, 5)


def test_actual_recovered_point_activation_routes_next_failure_before_more_point_trials(tmp_path):
    agent, _ = prepared(tmp_path, intercept=100.3)
    lab = BoundedReferenceLab(intercept=100.3)
    agent.lab = lab
    # Tuning to the fitted request (260.3) actually passes an activating 270.
    # The point stays refuted; the existing recovery restores that paid setting.
    lab.required[4] = 270.3
    lab.density[5] = 395.5 ** .5
    lab.required[5] = 4055.3
    _submit_point_prefix(agent, lab, square('density', 'density_square'))
    before = deepcopy(_state(agent))
    assert before['completed'] == [4] and lab.active(4) and not lab.active(5)
    assert [(test['number'], test['status']) for test in before['tests']] == [
        (4, 'refuted'), (5, 'refuted')]
    assert all(not test.get('response_question') for test in before['tests'])
    assert before['setting_recoveries'][-1]['restored']
    questions = deepcopy(_questions(agent))
    actions = len(lab.actions)

    result = agent._run_procedure(lab, PATH, max_steps=1)

    assert result['steps'][0]['proposal']['arguments']['operator'] == 'reference_response'
    assert len(lab.actions) == actions
    state = _state(agent)
    assert state['tests'] == before['tests']
    assert state['response_context']['number'] == 5
    assert not state.get('expanded_features')
    for question_id in ('reference_test_4_density_square', 'reference_test_5_density_square'):
        assert _questions(agent)[question_id] == questions[question_id]
        assert questions[question_id]['status'] == 'refuted'


def test_response_progress_requires_a_target_outcome_and_a_new_point_failure():
    state = {'target_numbers': [4, 5], 'tests': [
        {'number': 4, 'family': 'attainable_target_search', 'status': 'supported',
         'response_question': 'reference_response_4_6'},
        {'number': 5, 'family': 'signal_cubic_5', 'status': 'refuted'}]}
    assert recipe._point_needs_response(state, 5)
    for change in ({'number': 1}, {'authored_experiment': True},
                   {'restore_frequency': 100.}, {'status': 'inconclusive'}):
        altered = deepcopy(state)
        altered['tests'][0].update(change)
        assert not recipe._point_needs_response(altered, 5)
    for change in ({'status': 'supported'}, {'authored_experiment': True},
                   {'response_question': 'reference_response_5_1'}):
        altered = deepcopy(state)
        altered['tests'][1].update(change)
        assert not recipe._point_needs_response(altered, 5)
    state['tests'][0].pop('response_question')
    assert not recipe._point_needs_response(state, 5)


def test_authored_settings_are_not_a_shared_predictive_family():
    # The public response_experiment interface uses this same operational label
    # for all authored settings; matching labels do not identify a shared model.
    state = {'target_numbers': [4, 5], 'completed': [4], 'tests': [
        {'number': 4, 'family': 'authored_setting', 'status': 'supported',
         'authored_experiment': True},
        {'number': 5, 'family': 'authored_setting', 'status': 'refuted',
         'authored_experiment': True}]}
    assert not recipe._point_needs_response(state, 5)


def test_reference_restoration_is_not_support_on_another_target():
    state = {'target_numbers': [4, 5], 'tests': [
        {'number': 1, 'family': 'shape', 'status': 'supported', 'restore_frequency': 100.},
        {'number': 5, 'family': 'shape', 'status': 'refuted'}]}
    assert not recipe._point_needs_response(state, 5)
    state['tests'].insert(1, {'number': 4, 'family': 'shape', 'status': 'supported'})
    assert recipe._point_needs_response(state, 5)


def test_parameter_response_research_reopens_shape_and_confirms_only_actual_success(tmp_path):
    agent, lab, old = _quantized_extrapolation(tmp_path)
    _exhausted_grammar(agent)
    source = (agent.workspace / PATH).read_text()
    before = len(agent.observations)
    negative_settings = {item["frequency"] for item in _state(agent)["response_settings"]
                         if item["crystal"] == 5 and not item["activated"]}
    result = agent._run_procedure(lab, PATH, max_steps=256)

    assert result["status"] == "completed" and lab.success
    assert lab.frequencies[5] == 4055
    # Response failures may now extend the finite grammar. Existing declarations
    # and the controller's functions must survive that supported extension.
    before_tree = ast.parse(source)
    after_tree = ast.parse((agent.workspace / PATH).read_text())
    def declarations_and_controller(tree):
        declaration = next(node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'ESTIMATORS'
                                   for target in node.targets))
        return ast.literal_eval(declaration.value), [ast.dump(node) for node in tree.body
                                                     if node is not declaration]
    old_declarations, old_controller = declarations_and_controller(before_tree)
    new_declarations, new_controller = declarations_and_controller(after_tree)
    assert all(declaration in new_declarations for declaration in old_declarations)
    assert new_controller == old_controller
    questions = _questions(agent)
    assert questions[old["id"]] == old
    responses = [q for q in agent.questions if q["id"].startswith("reference_response_")]
    proposals = [q for q in responses if not q["id"].endswith("_physical")]
    physical = [q for q in responses if q["id"].endswith("_physical")]
    assert proposals and all(q["status"] in {"inconclusive", "reopened"} for q in proposals)
    assert any(q["status"] == "supported" and q["answer"]["activated"] for q in physical)
    assert all(q["parent_id"] in {p["id"] for p in proposals} for q in physical)
    assert all(q["answer"]["actual_frequency"] not in negative_settings for q in physical)
    assert all(all(index >= before for index in q["evidence"]) for q in physical)
    assert agent.costs["model_calls"] == 0


def test_response_proposal_alone_spends_no_environment_actions_and_cannot_confirm(tmp_path):
    agent, lab, old = _quantized_extrapolation(tmp_path)
    # The acquired prefix is fixed by the fixture; this operator only proposes.
    assert "response_research_history" not in _state(agent)
    before_actions, before_observations = len(lab.actions), len(agent.observations)
    result = agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)

    selected = result["selection"]["selected_experiment"]
    assert selected in result["experiments"]
    assert result["selected_setting"] == selected["prediction"]
    assert result["selected_number"] == selected["number"]
    assert 0 <= selected["prediction"] <= 10000
    assert not any(item["crystal"] == selected["number"]
                   and item["frequency"] == selected["prediction"]
                   for item in _state(agent)["response_settings"])
    assert len(lab.actions) == before_actions and len(agent.observations) == before_observations
    assert not lab.success
    question = _questions(agent)[result["question_id"]]
    assert question["status"] == "inconclusive"
    assert question["answer"]["scope"].endswith("no law confirmation")
    assert _questions(agent)[old["id"]] == old
    assert not any(q["id"].endswith("_physical") for q in agent.questions)


def test_failed_point_fit_trials_stop_when_joint_refit_adds_no_model(tmp_path, monkeypatch):
    from hegel_machine.cognition import (world_reference_response, world_response_constraints,
                                        world_response_hypotheses)
    # Isolate the retained finite point-fit branch. A real joint refit may now
    # return additional untested coefficient values; full-family consequences
    # can also select restored controls after the point-fit batch is exhausted.
    monkeypatch.setattr(world_response_constraints, 'fit_response_constraints',
        lambda *args: {'models': [], 'search': [], 'compute_cost': 0})
    def point_members(state, estimators, path, paid_checks=None):
        # Preserve the full proposer's result contract without asserting any
        # new family proof in this point-only branch.
        result = world_response_hypotheses._propose_response_members(state, estimators, path)
        result['family_history_checks'] = deepcopy(state.get('family_history_checks', {}))
        return result

    monkeypatch.setattr(world_reference_response, 'propose_response_research', point_members)
    # This target violates the shared quadratic relation. The controller should
    # retain the relation as a declaration, reject its measured predictions and
    # stop when this finite response search has no untried supported setting.
    agent, lab, old = _quantized_extrapolation(tmp_path, target_setting=4077.3)
    _exhausted_grammar(agent)
    source = (agent.workspace / PATH).read_text()
    initial_tests = deepcopy(_state(agent)["tests"])
    negative_settings = {item["frequency"] for item in _state(agent)["response_settings"]
                         if item["crystal"] == 5 and not item["activated"]}
    result = agent._run_procedure(lab, PATH, max_steps=256)

    assert result["status"] == "method_stopped" and not lab.success, result['steps'][-1].get('feedback')
    state = _state(agent)
    history = state["response_research_history"]
    assert len(history) > 1
    assert history[-1]["status"] == "no_untried_supported_setting"
    assert 0 < history[-1]["compatible_count"] < history[0]["compatible_count"]
    assert {item["candidate_count"] for item in history} == {history[0]["candidate_count"]}
    assert state["tests"][:len(initial_tests)] == initial_tests
    assert (agent.workspace / PATH).read_text() == source
    assert _questions(agent)[old["id"]] == old
    physical = [q for q in agent.questions if q["id"].startswith("reference_response_")
                and q["id"].endswith("_physical")]
    assert len(physical) > 1 and all(q["status"] == "refuted" for q in physical)
    settings = [q["answer"]["actual_frequency"] for q in physical]
    assert len(settings) == len(set(settings)) and not set(settings) & negative_settings
    assert {q["answer"]["family"] for q in physical} == {"density_square"}
    assert all(q["evidence"] and not q["answer"]["activated"] for q in physical)

    before = len(lab.actions), len(agent.questions), len(history)
    repeat = agent._run_procedure(lab, PATH, max_steps=256)
    assert repeat["status"] == "method_stopped"
    assert (len(lab.actions), len(agent.questions),
            len(_state(agent)["response_research_history"])) == before
    assert agent.costs["model_calls"] == 0


def test_intermediate_completion_records_actual_response_without_confirming_requested_setting(tmp_path, monkeypatch):
    from hegel_machine.cognition import world_reference_response, world_response_hypotheses

    # Isolate execution of an existing 4055 proposal: the environment can finish
    # at 4056 while the controller is still tuning toward that declared setting.
    # Choosing the research experiment is covered by the unrestricted success
    # test above; a new selector need not take this particular intermediate path.
    def choose_intermediate_path(state, estimators, path, paid_checks=None):
        result = world_response_hypotheses._propose_response_members(state, estimators, path)
        result['family_history_checks'] = deepcopy(state.get('family_history_checks', {}))
        selected = next(item for item in result["experiments"]
                        if item["number"] == 5 and item["prediction"] == 4055)
        result["selection"]["selected_experiment"] = deepcopy(selected)
        result.update(selected_number=5, selected_setting=4055)
        return result

    monkeypatch.setattr(world_reference_response, "propose_response_research", choose_intermediate_path)
    agent, lab, old = _quantized_extrapolation(
        tmp_path, target_setting=4056.3, immediate=True)
    _exhausted_grammar(agent)
    before = len(agent.observations)
    result = agent._run_procedure(lab, PATH, max_steps=256)

    assert result["status"] == "completed" and lab.success, result['steps'][-1].get('feedback')
    assert lab.frequencies[5] == 4056
    physical = [q for q in agent.questions if q["id"].startswith("reference_response_")
                and q["id"].endswith("_physical")]
    assert len(physical) == 1 and physical[0]["status"] == "inconclusive"
    feedback = physical[0]["answer"]
    assert feedback["requested_setting"] == 4055
    assert feedback == _state(agent)["response_feedback"]
    positive = [item for item in feedback["observed_responses"] if item["activated"]]
    assert [item["frequency"] for item in positive] == [4056]
    assert all(index >= before for item in positive for index in item["evidence"])
    assert set(positive[0]["evidence"]) <= set(physical[0]["evidence"])
    assert set(feedback["refuted_hypotheses"]) == {
        item["id"] for item in _state(agent)["response_research"]["hypotheses"]}
    assert feedback["remaining_hypotheses"] == []
    # No extra action continues from the actual successful setting to the
    # selected setting after the environment has already completed.
    assert len(lab.actions) - before == 4  # TALK, +10, -1, -1
    assert not any(item["frequency"] == 4055 for item in feedback["observed_responses"])
    assert _questions(agent)[old["id"]] == old
    assert agent.costs["model_calls"] == 0
