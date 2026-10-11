"""Draft authors receive computed contrast feedback before parent adoption."""

from copy import deepcopy
import json

from test_discoveryworld_explanation_proposal import (
    BINDING,
    CHILD,
    PATH,
    QUESTION,
    SESSION,
    SOURCE,
    prepared,
    propose,
)


# This proposal tests the design contract itself, without a separate operation
# prerequisite. The parent's actual world experiment remains unexecuted.
DESIGN = SOURCE.replace(
    "    assert memory['answers']['operation']['status'] == 'supported'\n", ""
)
DESIGN_QUESTION = {**QUESTION, "depends_on": []}


def scientific_state(agent, lab):
    return deepcopy((agent.questions, agent.observations, agent.relation_research, lab.actions))


def test_written_design_returns_computed_choice_to_author_without_scientific_adoption(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": DESIGN}),
        ("finish", {"lesson": "Return the computed contrast for the parent's decision."}),
    ])
    child = agent._question(CHILD)
    agent._execute("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    agent._record_answer(child, {
        "status": "supported", "answer": {"observed_active": True}, "evidence": [0]
    })
    before = scientific_state(agent, lab)
    compute = agent.costs["research_compute_units"]
    python_seconds = agent.costs["python_execution_seconds"]

    result = propose(agent, lab, question=DESIGN_QUESTION)

    comparison = provider.requests[1]["events"][-1]["feedback"]["design_comparison"]
    assert comparison["diagnosis"] == "discriminating_experiment_available"
    assert comparison["selected_experiment"]["id"] == "physical"
    assert comparison["selected_experiment"]["predictions"] == {"extra": True, "law": False}
    assert comparison["compute_cost"] > 0
    assert agent.costs["research_compute_units"] - compute == comparison["compute_cost"]
    assert agent.costs["python_execution_seconds"] > python_seconds
    assert scientific_state(agent, lab) == before
    assert result["proposal"]["design_comparison"] == comparison
    assert result["proposal"]["deliverable_status"] == "design_computed"
    assert result["proposal"]["contrast_action"]["arguments"]["path"] == PATH
    assert result["model_calls"] == len(provider.requests) == 2


def test_invalid_replacement_does_not_return_previous_valid_design_as_current(tmp_path):
    invalid = DESIGN.replace("'law': False", "'law': None")
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": DESIGN}),
        ("write", {"path": PATH, "code": invalid}),
        ("finish", {"lesson": "The revised predictions are not ready for adoption."}),
    ])
    before = scientific_state(agent, lab)

    result = propose(agent, lab, question=DESIGN_QUESTION)

    first = provider.requests[1]["events"][-1]["feedback"]["design_comparison"]
    failure = provider.requests[2]["events"][-1]["feedback"]["design_comparison"]
    assert first["selected_experiment"]["id"] == "physical"
    assert failure["diagnosis"] == "design_error"
    assert "Boolean" in failure["error"]
    assert failure.get("selected_experiment") is None
    current_proposals = json.loads(provider.requests[2]["files"]["method_decision.json"])[
        "explanation_proposals"
    ]
    current = next(row for row in current_proposals if row["session_id"] == SESSION)
    assert current["design_comparison"] == failure
    assert current["contrast_action"] is None
    assert result["proposal"]["design_comparison"] == failure
    assert result["proposal"]["contrast_action"] is None
    assert agent.llm_inquiries[SESSION]["proposal"]["contrast_action"] is None
    assert scientific_state(agent, lab) == before


def test_author_can_repair_failed_design_execution_using_its_actual_feedback(tmp_path):
    broken = DESIGN.replace(
        "def design(spec, observations, memory):\n",
        "def design(spec, observations, memory):\n    raise ValueError('candidate calculation failed')\n",
    )
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": broken}),
        ("write", {"path": PATH, "code": DESIGN}),
        ("finish", {"lesson": "The repaired design now computes a distinguishing prediction."}),
    ])
    before = scientific_state(agent, lab)

    result = propose(agent, lab, question=DESIGN_QUESTION)

    failure = provider.requests[1]["events"][-1]["feedback"]["design_comparison"]
    repaired = provider.requests[2]["events"][-1]["feedback"]["design_comparison"]
    assert failure["diagnosis"] == "design_error"
    assert "candidate calculation failed" in failure["error"]
    assert repaired["diagnosis"] == "discriminating_experiment_available"
    assert repaired["selected_experiment"]["id"] == "physical"
    assert result["proposal"]["design_comparison"] == repaired
    assert result["proposal"]["contrast_action"] is not None
    assert result["proposal"]["deliverable_status"] == "design_computed"
    assert scientific_state(agent, lab) == before


def test_finish_without_draft_reports_missing_explanation_instead_of_computed_ambiguity(tmp_path):
    agent, _, lab = prepared(tmp_path, [
        ("finish", {"lesson": "No distinguishing experiment seems available."}),
    ])
    before = scientific_state(agent, lab)

    result = propose(agent, lab, question=DESIGN_QUESTION)

    assert result["proposal"]["deliverable_status"] == "NO_EXECUTABLE_EXPLANATION_PROPOSED"
    assert result["proposal"]["contrast_action"] is None
    assert result["proposal"].get("design_comparison") is None
    assert not (tmp_path / PATH).exists()
    assert scientific_state(agent, lab) == before


def test_bind_only_draft_retains_existing_numerical_adoption_route(tmp_path):
    agent, _, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": BINDING}),
        ("finish", {"lesson": "Return the relation binding for paid-data comparison."}),
    ])
    before = scientific_state(agent, lab)

    result = propose(agent, lab, question=DESIGN_QUESTION)

    assert result["proposal"]["deliverable_status"] == "binding_proposed"
    assert result["proposal"]["contrast_action"] is None
    assert result["proposal"].get("design_comparison") is None
    assert result["proposal"]["comparison_action"]["arguments"]["operator"] == "research_update"
    assert result["proposal"]["research_action"]["arguments"]["operator"] == "research"
    assert scientific_state(agent, lab) == before


def test_resume_recomputes_retained_design_on_current_paid_observations(tmp_path):
    # A new paid observation changes this authored contrast's forecasts. A
    # resumed notebook must show its current agreement, not the old separation.
    adaptive = DESIGN.replace("'law': False", "'law': bool(observations)")
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": adaptive}),
        ("finish", {"lesson": "Return the initial contrast."}),
        ("finish", {"lesson": "The retained draft no longer separates these claims."}),
    ])
    initial = propose(agent, lab, question=DESIGN_QUESTION)
    assert initial["proposal"]["design_comparison"]["selected_experiment"]["id"] == "physical"
    agent._execute("query", {
        "question": {"id": "task"}, "actions": [{"action": "TALK", "arg1": 7}]
    }, lab)
    before = scientific_state(agent, lab)
    compute = agent.costs["research_compute_units"]

    resumed = propose(agent, lab, question=DESIGN_QUESTION)

    returned = provider.requests[2]["events"][-1]
    assert returned["action"] == "design_return"
    current = returned["feedback"]["design_comparison"]
    assert current["diagnosis"] == "observational_ambiguity"
    assert current["selected_experiment"] is None
    assert current["predicted_separation"] == 0
    assert current["experiment_count"] == 2
    assert current["comparison_scope"] == "authored_experiments_only"
    retained_proposals = json.loads(provider.requests[2]["files"]["method_decision.json"])[
        "explanation_proposals"
    ]
    retained = next(row for row in retained_proposals if row["session_id"] == SESSION)
    assert retained["design_comparison"] == current
    assert retained["design_comparison"] != initial["proposal"]["design_comparison"]
    assert resumed["proposal"]["design_comparison"] == current
    assert resumed["proposal"]["deliverable_status"] == "design_computed"
    assert agent.costs["research_compute_units"] - compute == current["compute_cost"]
    assert agent.llm_inquiries[SESSION]["invocations"] == 2
    assert scientific_state(agent, lab) == before


def test_ordinary_python_can_write_the_initial_design_and_receive_computed_feedback(tmp_path):
    writer = (
        "from pathlib import Path\n"
        f"path = Path({PATH!r})\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        f"path.write_text({DESIGN!r})\n"
    )
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"code": writer}),
        ("finish", {"lesson": "Return the design produced by the local calculation."}),
    ])
    before = scientific_state(agent, lab)

    result = propose(agent, lab, question=DESIGN_QUESTION)

    feedback = provider.requests[1]["events"][-1]["feedback"]
    assert feedback["returncode"] == 0
    assert feedback["design_comparison"]["selected_experiment"]["id"] == "physical"
    assert result["proposal"]["design_comparison"] == feedback["design_comparison"]
    assert result["proposal"]["deliverable_status"] == "design_computed"
    assert result["proposal"]["contrast_action"] is not None
    assert (tmp_path / PATH).read_text() == DESIGN
    assert agent.program_executions == 1
    assert scientific_state(agent, lab) == before


def test_python_repair_refreshes_failed_design_but_unrelated_calculation_does_not_rerun_it(tmp_path):
    invalid = DESIGN.replace("'law': False", "'law': None")
    repair = (
        "from pathlib import Path\n"
        f"path = Path({PATH!r})\n"
        "path.write_text(path.read_text().replace(\"'law': None\", \"'law': False\"))\n"
    )
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": invalid}),
        ("run", {"code": repair}),
        ("run", {"code": "print(sum([2, 3]))"}),
        ("finish", {"lesson": "Retain the corrected design after checking another calculation."}),
    ])
    before = scientific_state(agent, lab)
    compute = agent.costs["research_compute_units"]

    result = propose(agent, lab, question=DESIGN_QUESTION)

    failed = provider.requests[1]["events"][-1]["feedback"]["design_comparison"]
    repaired = provider.requests[2]["events"][-1]["feedback"]["design_comparison"]
    unrelated = provider.requests[3]["events"][-1]["feedback"]
    assert failed["diagnosis"] == "design_error"
    assert repaired["diagnosis"] == "discriminating_experiment_available"
    assert unrelated["returncode"] == 0 and unrelated["stdout"].strip() == "5"
    assert "design_comparison" not in unrelated
    assert result["proposal"]["design_comparison"] == repaired
    assert result["proposal"]["contrast_action"] is not None
    assert agent.program_executions == 2
    assert agent.costs["research_compute_units"] - compute == repaired["compute_cost"]
    assert scientific_state(agent, lab) == before


def test_design_without_return_is_a_repairable_author_error(tmp_path):
    no_return = DESIGN.replace("    return {'hypotheses':", "    unused = {'hypotheses':", 1)
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": no_return}),
        ("write", {"path": PATH, "code": DESIGN}),
        ("finish", {"lesson": "The design now returns its hypotheses and experiments."}),
    ])
    before = scientific_state(agent, lab)

    result = propose(agent, lab, question=DESIGN_QUESTION)

    failed = provider.requests[1]["events"][-1]["feedback"]["design_comparison"]
    corrected = provider.requests[2]["events"][-1]["feedback"]["design_comparison"]
    assert failed["diagnosis"] == "design_error"
    assert "dictionary" in failed["error"]
    assert failed.get("selected_experiment") is None
    assert corrected["selected_experiment"]["id"] == "physical"
    assert result["proposal"]["design_comparison"] == corrected
    assert result["proposal"]["deliverable_status"] == "design_computed"
    assert result["proposal"]["contrast_action"] is not None
    assert scientific_state(agent, lab) == before


def test_empty_experiment_design_is_missing_proposals_not_computed_ambiguity(tmp_path):
    empty = """def design(spec, observations, memory):
    return {'hypotheses': [
        {'id': 'shared', 'claim': 'The response is shared across objects'},
        {'id': 'local', 'claim': 'The response differs between objects'}],
        'experiments': []}

def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'No experiment has been specified', 'evidence': []}
"""
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": PATH, "code": empty}),
        ("finish", {"lesson": "The hypotheses still require an executable experimental proposal."}),
    ])
    before = scientific_state(agent, lab)
    compute = agent.costs["research_compute_units"]

    result = propose(agent, lab, question=DESIGN_QUESTION)

    comparison = provider.requests[1]["events"][-1]["feedback"]["design_comparison"]
    assert comparison["diagnosis"] == "no_experiments_proposed"
    assert comparison["experiment_count"] == 0
    assert comparison["comparison_scope"] == "authored_experiments_only"
    assert comparison["selected_experiment"] is None
    assert comparison["compute_cost"] == 0
    assert agent.costs["research_compute_units"] == compute
    assert result["proposal"]["design_comparison"] == comparison
    assert result["proposal"]["deliverable_status"] == "no_experiments_proposed"
    assert result["proposal"]["contrast_action"] is None
    assert scientific_state(agent, lab) == before
