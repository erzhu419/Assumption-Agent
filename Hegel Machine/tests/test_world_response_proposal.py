"""Response drafts face paid counterexamples inside the existing proposal loop."""

from copy import deepcopy

import pytest

from test_discoveryworld_llm_inquiry import Provider
from test_world_bound_response import BIND_CODE, _families, _ready, _state
from test_world_reference_hypothesis import PATH


DRAFT = "shared/proposed_response.py"
QUESTION = {"id": "computed_proposal", "parent_id": "task", "depends_on": [],
            "question": "Which computed representation explains the paid response evidence?"}


def _propose(agent, lab, actions):
    provider = agent.provider = Provider(actions)
    result = agent._execute("run", {"operator": "propose_explanations", "path": DRAFT,
        "response_path": PATH, "question": QUESTION, "max_model_calls": len(actions)}, lab)
    return result, provider


def _parent(agent, lab):
    return {"state": _state(agent), "questions": deepcopy(agent.questions),
            "observations": deepcopy(agent.observations), "actions": deepcopy(lab.actions),
            "controller": (agent.workspace / PATH).read_text()}


def test_written_response_draft_returns_paid_comparison_before_parent_adoption(tmp_path):
    agent, lab = _ready(tmp_path)
    before = _parent(agent, lab)
    compute = agent.costs["research_compute_units"]
    python_time = agent.costs["python_execution_seconds"]

    result, provider = _propose(agent, lab, [
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("finish", {"lesson": "The family survives; return its proposed experiment to the parent."}),
    ])

    comparison = provider.requests[1]["events"][-1]["feedback"]["response_comparison"]
    assert _families(comparison) == {"computed_log": "SAT_WITNESS"}
    assert comparison["comparison_status"] == "inconclusive"
    assert comparison["new_observation_indices"] == []
    assert comparison["paid_comparison_cost"] > 0
    assert result["proposal"]["comparison_result"] == comparison
    assert result["proposal"]["experiment_action"] == comparison["experiment_action"]
    assert comparison["experiment_action"] is not None
    assert _parent(agent, lab) == before
    assert agent.costs["research_compute_units"] - compute == comparison["compute_cost"]
    assert agent.costs["python_execution_seconds"] > python_time
    assert result["model_calls"] == len(provider.requests) == 2
    assert agent.costs["model_calls"] == 2
    action = result["proposal"]["comparison_action"]
    assert action == {"action": "run", "arguments": {"operator": "reference_response",
        "path": PATH, "binding_path": DRAFT, "question": QUESTION}}

    adopted = agent._execute(action["action"], action["arguments"], lab)

    assert _families(adopted) == _families(comparison)
    assert len(agent.observations) == len(before["observations"])
    assert lab.actions == before["actions"]
    assert next(q for q in agent.questions if q["id"] == QUESTION["id"])["status"] == "inconclusive"
    assert _state(agent)["bound_response_research"][QUESTION["id"]]["result"]["binding_path"] == DRAFT
    assert _state(agent)["response_research"] == before["state"]["response_research"]


def test_paid_refutation_reaches_next_draft_and_changed_representation_is_recomputed(tmp_path):
    agent, lab = _ready(tmp_path)
    trial = agent._execute("run", {"operator": "reference_response", "path": PATH,
        "experiment": {"number": 1, "setting": 103.}}, lab)
    assert trial["prediction_status"] == "refuted"
    before = _parent(agent, lab)
    compute = agent.costs["research_compute_units"]
    # Gap readings distinguish crystals whose density is identical. A quadratic
    # basis can explain this fixture's three references and their paid negatives.
    revised = BIND_CODE.replace("!= 91", "!= 92").replace(
        "density is approximately", "quantum gap is estimated to be").replace(
        "math.log(value)", "value").replace("log_density", "gap").replace(
        "computed_log", "computed_gap").replace(
        "[{'op': 'feature', 'name': 'gap'}]", "[{'op': 'feature', 'name': 'gap'}, "
        "{'op': 'mul', 'children': [{'op': 'feature', 'name': 'gap'}, "
        "{'op': 'feature', 'name': 'gap'}]}]")

    result, provider = _propose(agent, lab, [
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("write", {"path": DRAFT, "code": revised}),
        ("finish", {"lesson": "The revised representation removes the paid contradiction."}),
    ])

    refuted = provider.requests[1]["events"][-1]["feedback"]["response_comparison"]
    refreshed = provider.requests[2]["events"][-1]["feedback"]["response_comparison"]
    assert _families(refuted) == {"computed_log": "UNSAT_VERIFIED"}
    assert refuted["comparison_status"] == "refuted"
    assert refuted["experiment_action"] is None
    assert _families(refreshed) == {"computed_gap": "SAT_WITNESS"}
    assert refreshed["comparison_status"] == "inconclusive"
    assert set(trial["new_observation_indices"]) & set(refreshed["evidence"])
    assert result["proposal"]["comparison_result"] == refreshed
    assert (agent.workspace / DRAFT).read_text() == revised
    assert [event["arguments"]["code"] for event in result["steps"] if event["action"] == "write"] == [BIND_CODE, revised]
    assert _parent(agent, lab) == before
    assert agent.costs["research_compute_units"] - compute == (
        refuted["compute_cost"] + refreshed["compute_cost"])
    assert result["model_calls"] == 3


def test_invalid_computed_binding_is_repaired_in_the_same_proposal_session(tmp_path):
    agent, lab = _ready(tmp_path)
    before = _parent(agent, lab)
    broken = BIND_CODE.replace("    return {'rows':", "    rows[1]['evidence'] = [99999]\n    return {'rows':")

    result, provider = _propose(agent, lab, [
        ("write", {"path": DRAFT, "code": broken}),
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("finish", {"lesson": "Return the repaired and compared draft."}),
    ])

    failed = provider.requests[1]["events"][-1]["feedback"]
    assert "paid observation" in failed["error"]
    comparison = provider.requests[2]["events"][-1]["feedback"]["response_comparison"]
    assert _families(comparison) == {"computed_log": "SAT_WITNESS"}
    assert result["proposal"]["comparison_result"] == comparison
    assert result["status"] == "inquiry_finished"
    assert result["model_calls"] == 3
    assert agent.llm_inquiries["proposal:" + DRAFT]["model_calls"] == 3
    assert _parent(agent, lab) == before
    assert (agent.workspace / DRAFT).read_text() == BIND_CODE


def test_physical_counterexample_returns_to_author_and_resume_refreshes_before_revision(tmp_path):
    agent, lab = _ready(tmp_path)
    result, _ = _propose(agent, lab, [
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("finish", {"lesson": "Return the surviving family and its experiment."}),
    ])
    original = result["proposal"]["comparison_result"]
    action = result["proposal"]["experiment_action"]
    assert action["arguments"]["response_comparison"] == {
        "binding_path": DRAFT, "question": QUESTION}
    calls = agent.costs["model_calls"]

    trial = agent._execute(action["action"], action["arguments"], lab)

    # A failed point is not a refutation of every compatible coefficient and
    # band. Return both answers, preserving that distinction for the author.
    returned = trial["response_comparison"]
    assert trial["requested_setting_reached"]
    assert trial["prediction_status"] == "refuted"
    assert _families(returned) == {"computed_log": "SAT_WITNESS"}
    assert returned["comparison_status"] == "inconclusive"
    assert returned["response_count"] > original["response_count"]
    assert returned["new_observation_indices"] == []
    assert set(trial["new_observation_indices"]) & set(returned["evidence"])
    assert agent.costs["model_calls"] == calls
    session = agent.llm_inquiries["proposal:" + DRAFT]
    assert session["proposal"]["comparison_result"] == returned
    assert session["proposal"]["revision_action"] == trial["revision_action"]
    event = session["events"][-1]
    assert event["action"] == "response_return"
    assert event["feedback"]["response_comparison"] == returned
    assert event["feedback"]["physical_test"]["question_id"] == trial["question_id"]

    # More evidence may arrive between the physical return and the author's
    # next invocation. This cross-crystal contrast actually excludes the family.
    contrast = agent._execute("run", {"operator": "reference_response", "path": PATH,
        "experiment": {"number": 1, "setting": 103.}}, lab)
    assert contrast["prediction_status"] == "refuted"
    before_revision = _parent(agent, lab)
    revised = BIND_CODE.replace("!= 91", "!= 92").replace(
        "density is approximately", "quantum gap is estimated to be").replace(
        "math.log(value)", "value").replace("log_density", "gap").replace(
        "computed_log", "computed_gap").replace(
        "[{'op': 'feature', 'name': 'gap'}]", "[{'op': 'feature', 'name': 'gap'}, "
        "{'op': 'mul', 'children': [{'op': 'feature', 'name': 'gap'}, "
        "{'op': 'feature', 'name': 'gap'}]}]")
    provider = agent.provider = Provider([
        ("write", {"path": DRAFT, "code": revised}),
        ("finish", {"lesson": "The new measured contrast required a different representation."}),
    ])
    resume = deepcopy(trial["revision_action"])
    resume["arguments"]["max_model_calls"] = 2

    repaired = agent._execute(resume["action"], resume["arguments"], lab)

    refreshed = provider.requests[0]["events"][-1]["feedback"]["response_comparison"]
    assert _families(refreshed) == {"computed_log": "UNSAT_VERIFIED"}
    assert refreshed["comparison_status"] == "refuted"
    assert refreshed["response_count"] > returned["response_count"]
    assert set(contrast["new_observation_indices"]) & set(refreshed["evidence"])
    rewritten = provider.requests[1]["events"][-1]["feedback"]["response_comparison"]
    assert _families(rewritten) == {"computed_gap": "SAT_WITNESS"}
    assert repaired["proposal"]["comparison_result"] == rewritten
    assert repaired["model_calls"] == 2
    assert session["invocations"] == 2
    assert _parent(agent, lab) == before_revision


def test_interrupted_proposed_trial_retains_author_return_without_redeclaring_binding(tmp_path):
    agent, lab = _ready(tmp_path)
    result, _ = _propose(agent, lab, [
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("finish", {"lesson": "Return the family for an actual experiment."}),
    ])
    action = deepcopy(result["proposal"]["experiment_action"])
    action["arguments"]["max_steps"] = 4
    session = agent.llm_inquiries["proposal:" + DRAFT]
    calls = agent.costs["model_calls"]
    events = len(session["events"])

    interrupted = agent._execute(action["action"], action["arguments"], lab)

    assert interrupted["status"] == "method_step_limit"
    assert not interrupted["requested_setting_reached"]
    assert "response_comparison" not in interrupted
    assert len(session["events"]) == events
    assert _state(agent)["test"]["response_comparison"] == action["arguments"]["response_comparison"]

    resumed = agent._execute("run", {"operator": "reference_response", "path": PATH,
        "experiment": action["arguments"]["experiment"]}, lab)

    assert resumed["requested_setting_reached"]
    assert resumed["question_id"] == interrupted["question_id"]
    assert resumed["response_comparison"]["question_id"] == QUESTION["id"]
    assert resumed["response_comparison"]["binding_path"] == DRAFT
    assert set(interrupted["new_observation_indices"]) & set(resumed["response_comparison"]["evidence"])
    assert set(resumed["new_observation_indices"]) & set(resumed["response_comparison"]["evidence"])
    assert session["events"][-1]["action"] == "response_return"
    assert len(session["events"]) == events + 1
    assert session["proposal"]["comparison_result"] == resumed["response_comparison"]
    assert agent.costs["model_calls"] == calls


def test_parent_trial_with_fresh_comparison_id_retains_distinct_point_refutation(tmp_path):
    agent, lab = _ready(tmp_path)
    proposed, _ = _propose(agent, lab, [
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("finish", {"lesson": "Return the compared representation and its trial."}),
    ])
    action = deepcopy(proposed["proposal"]["experiment_action"])
    declaration = {"id": "fresh_parent_response_comparison", "parent_id": "task",
        "depends_on": [], "question": "Does this target setting activate, and how "
            "does its outcome update the retained response family?"}
    # The live parent supplied the same fresh ID for its point request and
    # source comparison. No active inquiry or previously registered ID can
    # disambiguate them; the physical verdict still needs its own child.
    action["arguments"]["research_question"] = deepcopy(declaration)
    action["arguments"]["response_comparison"]["question"] = deepcopy(declaration)
    assert agent._inquiry_stack == []
    assert not any(q["id"] == declaration["id"] for q in agent.questions)
    calls = agent.costs["model_calls"]

    trial = agent._execute(action["action"], action["arguments"], lab)

    assert trial["requested_setting_reached"]
    assert trial["prediction_status"] == "refuted"
    assert trial["question_id"] != declaration["id"]
    point = next(q for q in agent.questions if q["id"] == trial["question_id"])
    assert point["parent_id"] == declaration["id"]
    assert point["status"] == "refuted"
    assert point["answer"]["activated"] is False
    assert point["evidence"]
    compared = trial["response_comparison"]
    assert compared["question_id"] == declaration["id"]
    assert compared["comparison_status"] == "inconclusive"
    assert _families(compared) == {"computed_log": "SAT_WITNESS"}
    family = next(q for q in agent.questions if q["id"] == declaration["id"])
    assert family["status"] == "inconclusive"
    assert family["answer"]["binding_path"] == DRAFT
    returned = agent.llm_inquiries["proposal:" + DRAFT]["events"][-1]["feedback"]
    assert returned["physical_test"]["question_id"] == point["id"]
    assert returned["response_comparison"]["question_id"] == family["id"]
    assert agent.costs["model_calls"] == calls


@pytest.mark.parametrize("explicit_parent", [False, True], ids=["inherited", "explicit-parent"])
def test_active_assignment_receives_distinct_physical_children_not_point_answers(tmp_path, explicit_parent):
    agent, lab = _ready(tmp_path)
    parent = {"id": "explain_response", "parent_id": "task", "depends_on": [],
              "question": "Which measured variable and response premise explain the target?"}
    agent._question(parent)
    family = {**QUESTION, "parent_id": parent["id"]}
    agent.provider = Provider([
        ("write", {"path": DRAFT, "code": BIND_CODE}),
        ("finish", {"lesson": "Return the computed family's proposed experiment."}),
    ])
    proposed = agent._execute("run", {"operator": "propose_explanations", "path": DRAFT,
        "response_path": PATH, "question": family, "max_model_calls": 2}, lab)
    action = deepcopy(proposed["proposal"]["experiment_action"])
    # Live providers carry the enclosing assignment on each action, and may
    # repeat it explicitly. Neither form declares the individual point test.
    action["arguments"]["research_question"] = deepcopy(parent)
    if explicit_parent:
        action["arguments"]["question"] = deepcopy(parent)
    next_action = deepcopy(action)
    next_action["arguments"]["experiment"]["setting"] += 1.
    provider = agent.provider = Provider([
        (action["action"], action["arguments"]),
        (next_action["action"], next_action["arguments"]),
        ("finish", {"lesson": "Both points failed, but the explanatory question remains open."}),
    ])

    result = agent._execute("run", {"operator": "inquire", "question": parent,
        "max_model_calls": 3}, lab)

    assert result["status"] == "inquiry_finished"
    assert result["model_calls"] == len(provider.requests) == 3
    trials = [step["feedback"] for step in result["steps"][:2]]
    assert len({trial["question_id"] for trial in trials}) == 2
    assert not set(trials[0]["new_observation_indices"]) & set(trials[1]["new_observation_indices"])
    returns = [event["feedback"] for event in agent.llm_inquiries["proposal:" + DRAFT]["events"]
               if event["action"] == "response_return"]
    assert len(returns) == 2
    # The second paid point excludes this fixture's family. That still cannot
    # answer the broader question of which alternative premise explains it.
    assert [returned["response_comparison"]["comparison_status"] for returned in returns] == [
        "inconclusive", "refuted"]
    for physical, returned in zip(trials, returns):
        assert physical["requested_setting_reached"]
        assert physical["prediction_status"] == "refuted"
        assert physical["question_id"] != parent["id"]
        assert physical["new_observation_indices"]
        child = next(q for q in agent.questions if q["id"] == physical["question_id"])
        assert child["parent_id"] == parent["id"]
        assert child["status"] == "refuted" and child["evidence"]
        assert returned["physical_test"]["question_id"] == child["id"]
        assert returned["response_comparison"]["question_id"] == family["id"]
    assert result["question"]["status"] not in ("supported", "refuted")
    assert result["question"]["question"] == parent["question"]
    assert not result["question"].get("answer")


def test_response_author_can_test_its_draft_without_adopting_over_supported_answer(tmp_path):
    from test_world_condition_practice import ChoosingAuthor, finish

    agent, lab = _ready(tmp_path)
    enclosing = agent._question(deepcopy(QUESTION))
    agent._record_answer(enclosing, {
        'status': 'supported', 'answer': {'scope': 'Previously supported explanation'},
        'evidence': [0]})
    retained = deepcopy(enclosing)
    adopted = deepcopy(_state(agent).get('bound_response_research', {}))
    paid = len(agent.observations)

    def select_computed_trial(request):
        # Writing still only previews. Select the actual returned action, as
        # the live authors did before the proposal sandbox deferred it.
        assert enclosing == retained
        assert len(agent.observations) == paid
        return deepcopy(request['events'][-1]['feedback']['response_comparison']['experiment_action'])

    provider = agent.provider = ChoosingAuthor(agent, [
        lambda request: {'action': 'write', 'arguments': {'path': DRAFT, 'code': BIND_CODE}},
        select_computed_trial, finish])

    result = agent._execute('run', {'operator': 'propose_explanations', 'path': DRAFT,
        'response_path': PATH, 'question': QUESTION, 'max_model_calls': 3}, lab)

    actual = provider.requests[2]['events'][-1]['feedback']
    assert actual['requested_setting_reached']
    assert actual['prediction_status'] == 'refuted'
    assert actual['question_id'] != QUESTION['id']
    physical = next(q for q in agent.questions if q['id'] == actual['question_id'])
    assert physical['parent_id'] == QUESTION['id']
    assert physical['status'] == 'refuted' and physical['evidence']
    compared = actual['response_comparison']
    assert compared['binding_path'] == DRAFT
    assert _families(compared) == {'computed_log': 'SAT_WITNESS'}
    assert set(actual['new_observation_indices']) & set(compared['evidence'])
    assert result['proposal']['comparison_result'] == compared
    assert result['proposal']['deferred_actions'] == []
    # The real child can reopen a question; it cannot overwrite its old answer
    # with the trial point or the draft family's compatibility judgment.
    assert enclosing['status'] == 'reopened'
    assert enclosing['answer'] == retained['answer']
    assert enclosing['evidence'] == retained['evidence']
    assert _state(agent).get('bound_response_research', {}) == adopted
    assert result['status'] == 'inquiry_finished' and result['model_calls'] == 3
    assert all(frame['stage'] == 'proposal' for frame in provider.frames)
    assert (agent.workspace / DRAFT).read_text() == BIND_CODE
