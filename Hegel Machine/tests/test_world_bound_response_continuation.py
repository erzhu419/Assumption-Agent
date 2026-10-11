"""A new question name must not reset research on the same computed binding."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_reference_response as response
from hegel_machine.cognition import world_reference_recalibration as recipe
from test_world_bound_response import BINDING, BIND_CODE, _compare, _ready, _state
from test_world_reference_hypothesis import PATH


def _arguments(identifier, binding_path=BINDING):
    return {"operator": "reference_response", "path": PATH, "binding_path": binding_path,
            "question": {"id": identifier, "parent_id": "task",
                         "question": "Which response premise explains the measured data?"}}


def _remember_stagnation(agent, previous, *, author):
    previous = deepcopy(previous)
    previous["family_research"]["no_progress_cycles"] = 2
    if author:
        # The proposal writer may have compared a later draft without formally
        # adopting it. Its latest result must beat the older saved question.
        agent.llm_inquiries["proposal:" + BINDING] = {
            "question_id": previous["question_id"], "notes": "Compare the revised representation.",
            "events": [], "proposal": {"comparison_result": previous}}
    else:
        state = _state(agent)
        state["bound_response_research"][previous["question_id"]]["result"] = previous
        (agent.workspace / recipe.STATE).write_text(json.dumps(state))
    return previous


@pytest.mark.parametrize("author", [False, True], ids=["saved-question", "latest-author"])
def test_renamed_question_preserves_paid_stagnation_without_counting_same_evidence_twice(tmp_path, author):
    agent, lab = _ready(tmp_path)
    previous = _remember_stagnation(agent, _compare(agent, lab), author=author)
    observations, actions = len(agent.observations), deepcopy(lab.actions)

    for identifier in ("renamed_comparison", "another_comparison"):
        result = agent._execute("run", _arguments(identifier), lab)

        assert result["question_id"] == identifier
        assert result["response_count"] == previous["response_count"]
        assert result["family_research"]["no_progress_cycles"] == 2
        assert not result["family_research"]["new_response_received"]
        assert result["new_responses"] == []
        if author:
            session = agent.llm_inquiries["proposal:" + BINDING]
            assert session["proposal"]["comparison_result"] == result
            returned = session["events"][-1]
            assert returned["action"] == "response_return"
            assert returned["feedback"]["response_comparison"] == result
            assert "physical_test" not in returned["feedback"]
    assert len(agent.observations) == observations
    assert lab.actions == actions
    assert agent.costs["model_calls"] == 0


@pytest.mark.parametrize("change", ["basis", "features"])
def test_actual_binding_revision_resets_old_stagnation_even_after_question_rename(tmp_path, change):
    agent, lab = _ready(tmp_path)
    previous = _remember_stagnation(agent, _compare(agent, lab), author=True)
    if change == "basis":
        revised = BIND_CODE.replace(
            "[{'op': 'feature', 'name': 'log_density'}]",
            "[{'op': 'feature', 'name': 'log_density'}, "
            "{'op': 'mul', 'children': [{'op': 'feature', 'name': 'log_density'}, "
            "{'op': 'feature', 'name': 'log_density'}]}]")
    else:
        revised = BIND_CODE.replace("math.log(value)", "math.sqrt(value)")
    agent._execute("write", {"path": BINDING, "code": revised}, lab)

    result = agent._execute("run", _arguments("revised_comparison"), lab)

    old, current = previous["family_research"], result["family_research"]
    assert current["no_progress_cycles"] == 0
    assert current["epoch"] == old["epoch"] + 1
    assert current["response_count"] == old["response_count"]
    assert current["nonlocal_progress"] == 0
    assert current["excluded_prior_signatures"] == []
    assert "research_obligation" not in current
    assert (current["bases"] != old["bases"]) == (change == "basis")
    assert (current["features"] != old["features"]) == (change == "features")
    assert agent.llm_inquiries["proposal:" + BINDING]["proposal"]["comparison_result"] == result


def test_saved_comparisons_follow_latest_same_binding_and_keep_distinct_binding_separate(tmp_path, monkeypatch):
    agent, lab = _ready(tmp_path)
    template = _compare(agent, lab)
    state = _state(agent)
    old = {**deepcopy(template), "marker": "old"}
    latest = {**deepcopy(template), "marker": "latest"}
    unrelated = {**deepcopy(template), "binding_path": "shared/unrelated.py", "marker": "other"}
    state["bound_response_research"] = {
        "computed": {"binding_path": BINDING, "result": old},
        "latest": {"binding_path": BINDING, "result": latest},
        "other": {"binding_path": "shared/unrelated.py", "result": unrelated}}
    state_path = agent.workspace / recipe.STATE
    state_path.write_text(json.dumps(state))
    received = []

    def compare(agent, arguments, state, previous):
        received.append(deepcopy(previous))
        return {**deepcopy(template), "question_id": arguments["question"]["id"],
                "binding_path": arguments["binding_path"], "marker": len(received)}

    monkeypatch.setattr(response, "_compare_bound_response", compare)

    result = response._execute_bound_response(agent, _arguments("computed"), state, state_path)
    assert received[-1] == latest
    assert list(state["bound_response_research"])[-1] == "computed"
    response._execute_bound_response(agent, _arguments("new_name"), state, state_path)
    assert received[-1] == result

    # Reusing a question ID for a different binding does not let that binding
    # inherit unrelated progress, even when it has no author session yet.
    response._execute_bound_response(agent, _arguments("computed", "shared/fresh.py"), state, state_path)
    assert received[-1] == {}
    response._execute_bound_response(agent, _arguments("other_name", "shared/unrelated.py"), state, state_path)
    assert received[-1] == unrelated


def test_bound_comparison_keeps_own_programs_without_inheriting_same_named_raw_history(tmp_path, monkeypatch):
    agent, lab = _ready(tmp_path)
    previous = _compare(agent, lab)
    assert previous['candidates']
    state = _state(agent)
    raw_program = {'op': 'add', 'children': [
        {'op': 'const', 'value': 500.}, {'op': 'feature', 'name': 'log_density'}]}
    # The same coordinate and family names do not give a raw program the
    # authored binding's meaning. Its own previous candidates remain available.
    state.setdefault('response_research_history', []).append({
        'question_id': 'raw_coordinate_history', 'evidence': [0],
        'hypotheses': [{'family': 'computed_log', 'program': raw_program,
                        'lower': 0., 'upper': 1.}]})
    before, received = deepcopy(state), []
    original = response.propose_response_research

    def compare(working, estimators, path, paid_checks=None):
        received.append(deepcopy(working))
        return original(working, estimators, path, paid_checks=paid_checks)

    monkeypatch.setattr(response, 'propose_response_research', compare)
    result = response._compare_bound_response(agent, _arguments('next_bound_comparison'), state, previous)

    assert len(received) == 1
    assert received[0]['response_research_history'] == []
    assert received[0]['response_research'] == previous
    assert received[0]['response_research']['candidates'] == previous['candidates']
    assert result['binding_path'] == BINDING
    assert state == before
    assert state['response_research_history'][-1]['hypotheses'][0]['program'] == raw_program


def test_explicit_experiment_with_top_level_binding_runs_trial_then_returns_updated_comparison(tmp_path):
    agent, lab = _ready(tmp_path)
    compared = _compare(agent, lab)
    agent.llm_inquiries["proposal:" + BINDING] = {
        "question_id": compared["question_id"], "notes": "Use the actual response to reconsider this family.",
        "events": [], "proposal": {"comparison_result": deepcopy(compared)}}
    selected = deepcopy(compared["experiment_action"])
    # A live caller retained binding_path while adding the explicit experiment.
    # Its comparison question is correctly nested in response_comparison.
    selected["arguments"]["binding_path"] = BINDING
    assert "question" not in selected["arguments"]
    before = len(agent.observations)
    old_source = (agent.workspace / BINDING).read_text()

    trial = agent._execute(selected["action"], selected["arguments"], lab)

    assert trial["requested_setting_reached"]
    assert trial["prediction_status"] == "refuted"
    assert len(agent.observations) > before
    assert trial["new_observation_indices"] == list(range(before, len(agent.observations)))
    requested = selected["arguments"]["experiment"]
    assert any(row["frequency"] == requested["setting"] and not row["activated"]
               for row in trial["observed_responses"])
    updated = trial["response_comparison"]
    assert updated["question_id"] == compared["question_id"]
    assert updated["response_count"] > compared["response_count"]
    assert set(trial["new_observation_indices"]) & set(updated["evidence"])
    session = agent.llm_inquiries["proposal:" + BINDING]
    assert session["proposal"]["comparison_result"] == updated
    returned = session["events"][-1]["feedback"]
    assert returned["response_comparison"] == updated
    assert returned["physical_test"]["new_observation_indices"] == trial["new_observation_indices"]
    assert returned["physical_test"]["selected_setting"] == requested["setting"]
    assert (agent.workspace / BINDING).read_text() == old_source
    assert agent.costs["model_calls"] == 0


def test_returned_research_assignment_remains_parent_of_explicit_task_level_point_test(tmp_path):
    from test_discoveryworld_llm_inquiry import Provider

    agent, lab = _ready(tmp_path)
    compared = _compare(agent, lab)
    declaration = {"id": "explain_response", "parent_id": "task", "depends_on": [],
                   "question": "Which shared response premise explains the target?"}
    parent = agent._question(declaration)
    agent._record_answer(parent, {"status": "inconclusive", "evidence": [0],
        "answer": {"scope": "The measured reference alone does not settle the shared response premise."}})
    agent.provider = Provider([("finish", {"lesson": "The shared premise remains unresolved."})])
    returned = agent._execute("run", {"operator": "inquire", "question": declaration,
                                     "max_model_calls": 1}, lab)
    assert returned["status"] == "inquiry_finished" and agent._inquiry_stack == []
    old_parent = deepcopy(parent)
    before, calls = len(agent.observations), agent.costs["model_calls"]
    selected = compared["experiment_action"]["arguments"]["experiment"]

    trial = agent._execute("run", {"operator": "reference_response", "path": PATH,
        "experiment": deepcopy(selected), "question": deepcopy(declaration)}, lab)

    assert trial["requested_setting_reached"]
    assert trial["new_observation_indices"] and len(agent.observations) > before
    assert trial["prediction_status"] == "refuted"
    assert trial["question_id"] != parent["id"]
    physical = next(q for q in agent.questions if q["id"] == trial["question_id"])
    assert physical["parent_id"] == parent["id"]
    assert physical["evidence"] and min(physical["evidence"]) >= before
    assert physical["answer"]["prediction"] == selected["setting"]
    assert parent["status"] == "reopened"
    for field in ("question", "parent_id", "answer", "evidence"):
        assert parent[field] == old_parent[field]
    assert agent.costs["model_calls"] == calls
