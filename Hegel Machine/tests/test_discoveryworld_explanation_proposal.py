"""Proposal work computes candidates; the parent chooses whether to buy evidence."""

from copy import deepcopy
import json

from test_discoveryworld_contrast import QUESTION, SOURCE, prepared as physical_agent
from test_discoveryworld_llm_inquiry import CHILD, Provider, inquire, prepared
from test_discoveryworld_recovery import USAGE
from test_discoveryworld_research import BINDING, RelationLab


PATH = "shared/new_design.py"
SESSION = "proposal:" + PATH
RELATION_QUESTION = {"id": "response_law", "question": "Which response law explains the measured conditions?",
                     "parent_id": "task", "depends_on": []}


def propose(agent, lab, *, question=QUESTION, calls=8):
    return agent._execute("run", {"operator": "propose_explanations", "question": question,
                                  "path": PATH, "max_model_calls": calls}, lab)


def scientific_question(agent, qid):
    return next(question for question in agent.questions if question["id"] == qid)


def test_proposal_defers_experiments_and_scientific_mutations_without_pending_child_takeover(tmp_path):
    actions = [
        ("query", {"question": QUESTION, "actions": [{"action": "TALK", "arg1": 7}]}),
        ("evaluate", {"question_id": QUESTION["id"], "path": PATH, "code": SOURCE}),
        ("run", {"operator": "contrast", "path": PATH, "question": QUESTION}),
        ("run", {"operator": "question", **QUESTION}),
        ("run", {"operator": "inquire", "question": QUESTION, "max_model_calls": 2}),
        ("write", {"path": PATH, "code": SOURCE,
                   "research_question": QUESTION, "notes": "The candidate still needs a physical test."}),
        ("finish", {"lesson": "Return the candidate for the parent's decision."}),
    ]
    agent, provider, lab = prepared(tmp_path, actions)
    agent._question(CHILD)
    outer_focus = {"kind": "parent_research", "stage": "review", "question": {"id": "task"},
                   "path": "shared/retained_parent.py", "purpose": "Review the caller's earlier stop."}
    agent.research_focus = deepcopy(outer_focus)
    questions = deepcopy(agent.questions)
    notes = (tmp_path / "research_notes.md").read_text()

    result = propose(agent, lab)

    assert result["operator"] == "propose_explanations"
    assert result["model_calls"] == len(provider.requests) == 7
    assert agent.costs["model_calls"] == 7
    assert agent.costs["reported_tokens"] == 7 * USAGE["total_tokens"]
    assert agent.costs["cached_input_tokens"] == 7 * USAGE["cached_input_tokens"]
    assert agent.questions == questions and not lab.actions and not agent.observations
    assert agent.status == "running" and not agent.lesson
    assert agent.research_focus == outer_focus
    assert (tmp_path / "research_notes.md").read_text() == notes
    assert (tmp_path / PATH).read_text() == SOURCE
    assert set(agent.llm_inquiries) == {SESSION}
    assert agent.llm_inquiries[SESSION]["model_calls"] == 7
    for request in provider.requests:
        assert "research_root" not in request and "research_focus" not in request
        assert request["llm_inquiry"]["stage"] == "proposal"
        assert request["llm_inquiry"]["proposal_path"] == PATH
        assert request["enclosing_research_notes"] == notes
        assert "contrast" in request["execution_interface"] and "evaluate" in request["execution_interface"]
    deferred = result["proposal"]["deferred_actions"]
    assert len(deferred) == 5
    assert [(item["action"], item["arguments"]) for item in deferred] == actions[:5]
    assert all(step["feedback"]["status"] == "proposal_action_deferred" for step in result["steps"][:5])
    assert result["proposal"]["path"] == PATH
    assert result["proposal"]["contrast_action"]["arguments"]["question"]["id"] == QUESTION["id"]
    agent._prepare_program_state()
    summary = json.loads((tmp_path / "method_decision.json").read_text())
    retained, = summary["explanation_proposals"]
    assert retained["session_id"] == SESSION
    assert retained["notes"] == result["notes"] == "The candidate still needs a physical test."
    assert retained["contrast_action"] == result["proposal"]["contrast_action"]
    assert retained["deferred_actions"] == deferred


def test_existing_supported_answer_does_not_end_or_get_rewritten_by_proposal_work(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": SOURCE}),
        ("finish", {"lesson": "The supported old answer remains separate from this new proposal."}),
    ])
    child = agent._question(CHILD)
    agent._execute("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    agent._record_answer(child, {"status": "supported", "answer": {"observed_active": True}, "evidence": [0]})
    before = deepcopy(agent.questions)
    revised_question = {**CHILD, "question": "Does a changed condition require a different explanation?"}

    result = propose(agent, lab, question=revised_question)

    assert result["model_calls"] == len(provider.requests) == 2
    assert agent.questions == before and child["status"] == "supported"
    assert len(lab.actions) == 1
    returned = result["proposal"]["contrast_action"]["arguments"]["question"]
    assert returned["question"] == revised_question["question"]
    assert returned["parent_id"] == CHILD["parent_id"]
    assert returned["depends_on"] == CHILD["depends_on"]
    assert returned["analysis_path"] == PATH


def test_ordinary_python_failure_can_be_repaired_inside_the_same_proposal_notebook(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"code": "raise ValueError('The prospective calculation has the wrong shape')",
                 "notes": "Proposal: compute the candidates before designing their test."}),
        ("run", {"code": "print('Repaired candidate calculation')",
                 "notes": "Proposal: the calculation is repaired; write the resulting design."}),
        ("write", {"path": PATH, "code": SOURCE}),
        ("finish", {"lesson": "Return the executable proposal without making a physical claim."}),
    ])
    questions = deepcopy(agent.questions)
    parent_notes = (tmp_path / "research_notes.md").read_text()

    result = propose(agent, lab)

    failure = provider.requests[1]["events"][-1]["feedback"]
    assert failure["returncode"] != 0 and "wrong shape" in failure["stderr"]
    assert provider.requests[1]["research_notes"] == "Proposal: compute the candidates before designing their test."
    assert result["steps"][1]["feedback"]["stdout"].strip() == "Repaired candidate calculation"
    assert result["model_calls"] == 4 and agent.llm_inquiries[SESSION]["model_calls"] == 4
    assert result["proposal"]["contrast_action"]["arguments"]["path"] == PATH
    assert result["proposal"]["deferred_actions"] == []
    assert agent.questions == questions and not lab.actions
    assert (tmp_path / "research_notes.md").read_text() == parent_notes


def test_parent_adoption_registers_the_proposed_question_and_selects_the_discriminating_experiment(tmp_path):
    agent, lab = physical_agent(tmp_path)
    parent_path = "shared/retained_parent.py"
    parent_source = "def select_actions(spec, observations, memory):\n    return {'stop': 'Compare explanations first'}\n"
    agent._execute("write", {"path": parent_path, "code": parent_source}, lab)
    agent._question({"id": "task", "parent_program_path": parent_path})
    child = agent._question(QUESTION)
    old_analysis_path = "shared/independent_answer.py"
    old_analysis = ("def answer(spec, observations, memory):\n"
                    "    return {'status': 'inconclusive', 'answer': 'Only the old menu was observed', 'evidence': [0]}\n")
    agent._execute("evaluate", {"question_id": child["id"], "path": old_analysis_path,
                               "code": old_analysis}, lab)
    agent._request()
    questions = deepcopy(agent.questions)
    existing_actions = deepcopy(lab.actions)
    agent.provider = Provider([
        ("write", {"path": PATH, "code": SOURCE}),
        ("finish", {"lesson": "The parent should decide whether to adopt this proposed contrast."}),
    ])
    proposed_question = {**QUESTION, "question": "Can this physical intervention separate the two explanations?"}

    proposed = propose(agent, lab, question=proposed_question)

    assert agent.questions == questions and lab.actions == existing_actions
    assert child["analysis_path"] == old_analysis_path
    assert (tmp_path / parent_path).read_text() == parent_source
    assert (tmp_path / old_analysis_path).read_text() == old_analysis
    action = proposed["proposal"]["contrast_action"]
    assert action["arguments"]["question"] == {**proposed_question, "analysis_path": PATH}

    actual = agent._execute(action["action"], action["arguments"], lab)

    assert actual["selected_experiment"]["id"] == "physical"
    assert lab.actions == [*existing_actions, {"action": "USE", "arg1": 8, "arg2": 7}]
    assert child["question"] == proposed_question["question"] and child["analysis_path"] == PATH
    assert child["status"] == "refuted" and child["evidence"] == [1]
    assert child["answer"] == {"observed_outcome": False, "tested_prediction": True}
    assert (tmp_path / parent_path).read_text() == parent_source
    assert (tmp_path / old_analysis_path).read_text() == old_analysis


def test_nested_proposal_shares_call_ceiling_and_returns_one_decision_to_the_parent(tmp_path):
    proposed_question = {**QUESTION, "parent_id": CHILD["id"], "depends_on": []}
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "propose_explanations", "question": proposed_question,
                 "path": PATH, "max_model_calls": 20, "notes": "Parent: compare the proposal on return."}),
        ("write", {"path": PATH, "code": SOURCE}),
        ("run", {"code": "print('This draft still needs a parent decision')"}),
        ("finish", {"lesson": "Return the draft without testing it yet."}),
    ])

    result = inquire(agent, lab, calls=4)

    assert result["model_calls"] == agent.costs["model_calls"] == len(provider.requests) == 4
    assert [r.get("llm_inquiry", {}).get("stage") == "proposal" for r in provider.requests] == [False, True, True, False]
    assert agent.llm_inquiries[SESSION]["model_calls"] == 2
    assert agent.llm_inquiries[CHILD["id"]]["model_calls"] == 2
    assert provider.requests[-1]["research_notes"] == "Parent: compare the proposal on return."
    assert agent.costs["reported_tokens"] == 4 * USAGE["total_tokens"]
    assert {q["id"] for q in agent.questions} == {"task", CHILD["id"]}
    assert result["status"] == "inquiry_finished" and not lab.actions


def test_top_level_proposal_leaves_the_parent_its_last_shared_model_decision(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": SOURCE}),
        ("run", {"code": "print('The proposed design remains untested')"}),
        ("finish", {"lesson": "The parent retains this proposal without testing it yet."}),
    ], limits={"max_model_calls": 3})

    proposal = propose(agent, lab, calls=20)

    assert proposal["model_calls"] == agent.costs["model_calls"] == 2
    assert len(provider.requests) == 2 and agent.status == "running"
    assert proposal["proposal"]["contrast_action"]["arguments"]["path"] == PATH
    result = agent.run(lab)
    assert len(provider.requests) == result["costs"]["model_calls"] == 3
    assert "llm_inquiry" not in provider.requests[-1]
    assert result["status"] == "agent_finished"
    assert result["lesson"] == "The parent retains this proposal without testing it yet."
    assert result["costs"]["reported_tokens"] == 3 * USAGE["total_tokens"]
    assert agent.llm_inquiries[SESSION]["model_calls"] == 2 and not lab.actions


def relation_proposal_agent(tmp_path, steps, conditions):
    agent, provider, _ = prepared(tmp_path, steps)
    lab = RelationLab(square=True)
    agent.lab = lab
    agent._execute("query", {"question": {"id": "task"}, "actions": [
        {"action": "READ", "arg1": x} for x in conditions]}, lab)
    return agent, provider, lab


def test_bind_proposal_compares_one_model_across_paid_conditions_before_buying_evidence(tmp_path):
    agent, provider, lab = relation_proposal_agent(tmp_path, [
        ("write", {"path": PATH, "code": BINDING}),
        ("run", {"operator": "research_update", "path": PATH, "question": RELATION_QUESTION}),
        ("finish", {"lesson": "Compare the whole predicted response with the existing measurements."}),
    ], conditions=[0, 1, 2])
    questions, observations = deepcopy(agent.questions), deepcopy(agent.observations)
    executions = agent.program_executions

    result = propose(agent, lab, question=RELATION_QUESTION)

    assert agent.questions == questions and agent.observations == observations
    assert not agent.relation_research and agent.program_executions == executions
    assert len(lab.actions) == 3 and result["model_calls"] == len(provider.requests) == 3
    assert result["steps"][1]["feedback"]["status"] == "proposal_action_deferred"
    assert result["proposal"]["contrast_action"] is None
    action = result["proposal"]["comparison_action"]
    assert action == {"action": "run", "arguments": {"operator": "research_update",
        "path": PATH, "question": RELATION_QUESTION}}
    assert result["proposal"]["research_action"]["arguments"]["operator"] == "research"
    assert "bind" in provider.requests[0]["task_interface"]
    assert "Program" in provider.requests[0]["task_interface"]

    parent_request = agent._request()
    retained, = json.loads(parent_request["files"]["method_decision.json"])["explanation_proposals"]
    assert retained["comparison_action"] == action
    assert retained["research_action"] == result["proposal"]["research_action"]
    assert parent_request["evidence"]["observation_count"] == 3

    comparison = agent._execute(action["action"], action["arguments"], lab)

    assert len(lab.actions) == 3 and agent.observations == observations
    assert comparison["fit_evidence"] == [0, 1, 2]
    candidates = {item["program"]["op"]: item for item in comparison["candidates"]}
    assert candidates["mul"]["fit_mse"] == 0 and candidates["mul"]["status"] == "plausible"
    assert candidates["feature"]["fit_mse"] > 0 and candidates["feature"]["status"] == "refuted"
    assert comparison["status"] == "inconclusive" and comparison["diagnosis"] == "needs_confirmation"
    assert comparison["selected_probe"]["actions"] == [{"action": "READ", "arg1": 3}]
    assert comparison["selected_probe"]["predictions"] == [9.0]
    assert scientific_question(agent, RELATION_QUESTION["id"])["status"] != "supported"


def test_same_question_bind_adoption_keeps_frozen_candidates_for_new_condition(tmp_path):
    agent, _, lab = relation_proposal_agent(tmp_path, [
        ("write", {"path": PATH, "code": BINDING}),
        ("finish", {"lesson": "The two explanations agree on the first measurements."}),
        ("finish", {"lesson": "Compare the retained explanations with the additional condition."}),
    ], conditions=[0, 1])
    first = propose(agent, lab, question=RELATION_QUESTION)["proposal"]["comparison_action"]
    initial = agent._execute(first["action"], first["arguments"], lab)
    assert len(initial["candidates"]) == 2
    assert all(item["status"] == "plausible" for item in initial["candidates"])
    assert initial["selected_probe"]["predictions"] == [2.0, 4.0]
    assert len(lab.actions) == 2

    # The parent chooses an actual measurement before resuming the same draft.
    agent._execute("query", {"question": {"id": RELATION_QUESTION["id"]},
                            "actions": initial["selected_probe"]["actions"]}, lab)
    retained = deepcopy(agent.relation_research[RELATION_QUESTION["id"]])
    second = propose(agent, lab, question=RELATION_QUESTION)["proposal"]["comparison_action"]
    assert agent.relation_research[RELATION_QUESTION["id"]] == retained
    assert len(lab.actions) == 3

    comparison = agent._execute(second["action"], second["arguments"], lab)

    assert comparison["fit_rows"] == initial["fit_rows"]
    assert comparison["fit_evidence"] == [0, 1] and comparison["validation_evidence"] == [2]
    candidates = {item["program"]["op"]: item for item in comparison["candidates"]}
    assert {item["program"]["op"] for item in initial["candidates"]} == set(candidates)
    assert candidates["feature"]["fit_mse"] == 0 and candidates["feature"]["validation_mse"] == 4
    assert candidates["feature"]["status"] == "refuted"
    assert candidates["mul"]["validation_mse"] == 0 and candidates["mul"]["status"] == "plausible"
    assert comparison["status"] == "inconclusive" and comparison["diagnosis"] == "needs_confirmation"
    assert len(lab.actions) == 3 and agent.llm_inquiries[SESSION]["invocations"] == 2


def test_combined_bind_and_design_draft_returns_both_parent_choices_without_execution(tmp_path):
    agent, _, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": BINDING + "\n" + SOURCE}),
        ("finish", {"lesson": "Return both the numerical model and the proposed physical comparison."}),
    ])
    questions = deepcopy(agent.questions)

    result = propose(agent, lab, question=RELATION_QUESTION)

    proposal = result["proposal"]
    assert proposal["comparison_action"]["arguments"]["operator"] == "research_update"
    assert proposal["research_action"]["arguments"]["operator"] == "research"
    assert proposal["contrast_action"]["arguments"]["operator"] == "contrast"
    assert proposal["contrast_action"]["arguments"]["question"]["analysis_path"] == PATH
    assert all(proposal[key]["arguments"]["path"] == PATH
               for key in ("comparison_action", "research_action", "contrast_action"))
    assert agent.questions == questions and not agent.relation_research
    assert not lab.actions and not agent.observations and agent.program_executions == 0
