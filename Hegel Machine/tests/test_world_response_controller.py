"""Revisit refuted point predictions through shared response hypotheses.

These integration fixtures test controller behavior using the existing public
laboratory, not model-authored discovery or a DiscoveryWorld benchmark.
"""

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


def _quantized_extrapolation(tmp_path, *, target_setting=4055.3, immediate=False):
    """One successful rounded label moves the fitted extrapolation too far."""
    agent, _ = prepared(tmp_path, intercept=100.3)
    lab = (ImmediateCompletionLab if immediate else BoundedReferenceLab)(intercept=100.3)
    agent.lab = lab
    lab.density[5] = 395.5 ** .5
    lab.required[5] = target_setting
    first = submit(agent, lab, square("density", "density_square"))
    assert not lab.success and first["status"] != "completed"
    old = _questions(agent)["reference_test_5_density_square"]
    assert old["status"] == "refuted"
    assert lab.frequencies[4] == 260 and lab.frequencies[5] != 4055
    return agent, lab, deepcopy(old)


def _exhausted_grammar(agent):
    state = _state(agent)
    state["expanded_features"] = ["density", "gap"]
    (agent.workspace / recipe.STATE).write_text(json.dumps(state))


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
    assert (agent.workspace / PATH).read_text() == source
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
    # The isolated declaration path does not silently launch response research.
    assert "response_research_history" not in _state(agent)
    before_actions, before_observations = len(lab.actions), len(agent.observations)
    result = agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)

    assert result["selected_setting"] == 4055
    assert len(lab.actions) == before_actions and len(agent.observations) == before_observations
    assert not lab.success
    question = _questions(agent)[result["question_id"]]
    assert question["status"] == "inconclusive"
    assert question["answer"]["scope"].endswith("no law confirmation")
    assert _questions(agent)[old["id"]] == old
    assert not any(q["id"].endswith("_physical") for q in agent.questions)


def test_failed_point_fit_trials_stop_when_joint_refit_adds_no_model(tmp_path, monkeypatch):
    from hegel_machine.cognition import world_response_constraints
    # Isolate the retained finite point-fit branch. A real joint refit may now
    # return additional untested coefficient values instead of stopping here.
    monkeypatch.setattr(world_response_constraints, 'fit_response_constraints',
        lambda *args: {'models': [], 'search': [], 'compute_cost': 0})
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

    assert result["status"] == "method_stopped" and not lab.success
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


def test_intermediate_completion_records_actual_response_without_confirming_requested_setting(tmp_path):
    agent, lab, old = _quantized_extrapolation(
        tmp_path, target_setting=4056.3, immediate=True)
    _exhausted_grammar(agent)
    before = len(agent.observations)
    result = agent._run_procedure(lab, PATH, max_steps=256)

    assert result["status"] == "completed" and lab.success
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
