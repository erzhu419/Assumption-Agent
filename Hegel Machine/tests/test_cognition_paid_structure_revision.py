"""Existing structural revisions must consume paid trajectories before new probes."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import _Fitter, _entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_structure import propose_structures
from test_cognition_structural_neighbor_cells import paid_response


def simple_scene():
    schema = {"objects": ["object"], "fields": {"object": {
        "q": "q", "u": "u", "x": "x", "f": "f"}}}
    rows = [Observation({"q": value, "u": 0., "x": value, "f": 1.}, value)
            for value in (-2., -1., 1., 3.)]
    spec = TaskSpec("paid_structure", "revise a paid explanation", ("q", "u", "x", "f"), (), (),
                    context={"object_schema": json.dumps(schema)})
    return spec, rows, schema


def claim(name, program, schema):
    return {"candidate": Candidate(name, program, "representation"),
            "kind": "structural:local_exchange", "family_id": "T08", "fallback": 0.,
            "training_loss": 0., "validation_loss": 0.,
            "structure": {"schema": schema, "fits": [{"family": "local_exchange", "training_loss": 0.,
                                                      "radius": .5}],
                          "radius_interval": [0., 1.]}}


def shifted_x(amount):
    return Program("add", children=(Program("feature", name="x"), Program("const", value=amount)))


def test_paid_revision_preserves_better_old_program_and_same_id_versions(monkeypatch):
    from hegel_machine.cognition import unified_structure
    spec, rows, schema = simple_scene()
    parent = claim("parent", Program("const", value=0.), schema)
    better_old = claim("reused_id", shifted_x(.2), schema)
    new = claim("reused_id", shifted_x(.5), schema)
    original = deepcopy([parent, better_old])
    received = []

    def propose(spec, training, budget, *, parent=None, **kwargs):
        received.append((list(training), parent))
        assert budget.spend(7)
        return [new]

    monkeypatch.setattr(unified_structure, "propose_structures", propose)
    work = SearchBudget(300)
    result = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))._revise_paid_structure(
        spec, parent, rows, work, [], finalists=[parent, better_old])
    assert received[0][0] == rows and received[0][1]["structure"] == parent["structure"]
    assert result["full_comparison"] is True
    assert result["selected"]["candidate"].program == better_old["candidate"].program
    assert {item["candidate"].program for item in result["claims"]} >= {
        parent["candidate"].program, better_old["candidate"].program, new["candidate"].program}
    assert [parent, better_old] == original
    assert work.spent == result["compute_cost"] and work.spent > 7


def test_exact_paid_scores_are_reused_instead_of_rescoring_old_programs(monkeypatch):
    from hegel_machine.cognition import unified_research, unified_structure
    spec, rows, schema = simple_scene()
    parent = claim("parent", Program("const", value=0.), schema)
    old = claim("old", shifted_x(1.), schema)
    proposal = claim("new", shifted_x(.1), schema)
    known = {entry["candidate"].program: loss(entry["candidate"].program, rows, 0., .5)
             for entry in (parent, old)}
    original_predict = unified_research.predict
    scored = []

    def score(program, observation, fallback, **kwargs):
        assert program not in known, "unchanged program has a same-cohort paid score"
        scored.append(program)
        return original_predict(program, observation, fallback, **kwargs)

    monkeypatch.setattr(unified_research, "predict", score)
    monkeypatch.setattr(unified_structure, "propose_structures", lambda *args, **kwargs: [proposal])
    work = SearchBudget(100)
    result = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))._revise_paid_structure(
        spec, parent, rows, work, [], finalists=[parent, old], paid_scores=known)
    assert result["full_comparison"] is True and result["status"] == "improved"
    assert scored == [proposal["candidate"].program] * len(rows)
    assert work.spent == len(rows)
    assert result["selected"]["candidate"].program == proposal["candidate"].program


def test_unfunded_complete_comparison_never_truncates_the_pool_or_starts_search(monkeypatch):
    from hegel_machine.cognition import unified_structure
    spec, rows, schema = simple_scene()
    parent = claim("parent", Program("const", value=0.), schema)
    retained = [parent, claim("other", shifted_x(1.), schema)]

    def forbidden(*args, **kwargs):
        raise AssertionError("unfunded full comparison must not start another structural fit")

    monkeypatch.setattr(unified_structure, "propose_structures", forbidden)
    work = SearchBudget(len(rows) - 1)
    result = UnifiedResearchAgent(ResearchConfig())._revise_paid_structure(
        spec, parent, rows, work, [], finalists=retained)
    assert result["status"] == "compute_unresolved" and result["full_comparison"] is False
    assert result["selected"] is parent
    assert {item["candidate"].program for item in result["claims"]} == {
        item["candidate"].program for item in retained}
    assert not result["new_claims"]


@pytest.mark.parametrize("research_action", ["structural_research", "response_basis_revision"])
def test_already_explained_paid_response_does_not_repeat_structural_fitting(monkeypatch, research_action):
    from hegel_machine.cognition import unified_structure
    spec, rows, schema = simple_scene()
    parent = claim("correct", Program("feature", name="x"), schema)

    def forbidden(*args, **kwargs):
        raise AssertionError("a passing paid model does not need another fit")

    monkeypatch.setattr(unified_structure, "propose_structures", forbidden)
    monkeypatch.setattr(unified_structure, "propose_response_revisions", forbidden)
    result = UnifiedResearchAgent(ResearchConfig())._revise_paid_structure(
        spec, parent, rows, SearchBudget(100), [], research_action=research_action)
    assert result["status"] == "already_explained"
    assert result["selected"]["candidate"] == parent["candidate"]
    assert not result["new_claims"]


def test_real_neighbor_revision_uses_complete_paid_trajectories_without_a_probe():
    rows, schema, response = paid_response()
    spec = TaskSpec("paid_neighbor", "existing trajectory counterexamples", schema.feature_names, (), (),
                    context={"object_schema": json.dumps(schema.to_dict())})
    setup = SearchBudget(43234)
    fit = _Fitter(rows[:34], setup, .5).fit(response)
    partial = {**_entry(fit, [], 34, setup.spent), "partial_object_knowledge": True}
    original_pool = propose_structures(spec, rows[:34], setup, parent=partial, tolerance=1e-7)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-10))
    parent = agent._best([{**item, "validation_loss": loss(item["candidate"].program,
                         rows[34:58], 0., .5), "fallback": 0.} for item in original_pool])
    assert loss(parent["candidate"].program, rows, 0., .5) > agent.config.minimum_improvement
    original = deepcopy(parent)
    budget = SearchBudget(12000)
    result = agent._revise_paid_structure(spec, parent, rows, budget, [], finalists=original_pool)
    assert result["full_comparison"] is True and result["status"] == "improved"
    winner = result["selected"]
    assert loss(winner["candidate"].program, rows, 0., .5) < 1e-16
    assert max((winner["candidate"].program.predict(row) - row.target) ** 2 for row in rows) < 1e-16
    assert winner["structure_search"] in {"boundary_revision", "family_revision"}
    assert parent == original and budget.spent <= budget.limit
    assert {item["candidate"].program for item in result["claims"]} >= {
        item["candidate"].program for item in original_pool}


def test_native_paid_structure_revision_precedes_new_experiment_and_preserves_pool(monkeypatch):
    from dataclasses import replace
    from hegel_machine.cognition import unified_structure
    from hegel_machine.cognition.object_continuation import encode_entry
    from test_cognition_object_complete_confirmation_budget import (
        ConstantObjectResponseLab, object_agent, paid_object_scene)

    laboratory = ConstantObjectResponseLab()
    previous = paid_object_scene(laboratory)
    schema = json.loads(laboratory.spec_for(("fine",)).context["object_schema"])
    parent = claim("retained_structure", Program("const", value=0.), schema)
    other = claim("retained_competitor", Program("const", value=3.), schema)
    new = claim("revised_structure", Program("const", value=1.), schema)
    for entry in (parent, other, new):
        entry["structure"]["hypotheses"] = []
    previous.update(selected_kind=parent["kind"], selected_structure=parent["structure"],
        object_continuation_state={"selected": encode_entry(parent),
            "structure_comparison_pending": {"status": "compute_unresolved", "rows": 12},
            "finalists": [encode_entry(parent), encode_entry(other)],
            "schemas": [{"channels": ["fine"], "schema": schema}]})
    agent = object_agent()
    agent.config = replace(agent.config, active_probes=True, risk_learning=False)
    fitted = []

    def proposal(spec, training, budget, *, parent=None, **kwargs):
        fitted.append(list(training))
        assert budget.spend(8)
        return [new]

    def forbidden_experiment(*args, **kwargs):
        pytest.fail("paid counterexamples must be consumed before buying another structural experiment")

    monkeypatch.setattr(unified_structure, "propose_structures", proposal)
    monkeypatch.setattr(unified_structure, "structural_experiments", forbidden_experiment)
    report = agent.run(laboratory, previous=previous)

    assert report["status"] == "completed" and len(fitted) == 1
    assert report["object_continuation_state"]["structure_comparison_pending"] is None
    assert len(fitted[0]) == 12 and all(row.target == 1. for row in fitted[0])
    actions = report["actions"]
    revised = next(i for i, event in enumerate(actions) if event["action"] == "revise_paid_structure")
    acquired = next(i for i, event in enumerate(actions) if event["action"] == "acquire")
    assert revised < acquired and actions[acquired]["purpose"] == "confirm_frozen_program"
    pool = report["object_continuation_state"]["finalists"]
    programs = [item["candidate"]["program"] for item in pool]
    assert all(item["candidate"].program.to_dict() in programs for item in (parent, other, new))


def test_native_incomplete_paid_structure_comparison_blocks_fresh_confirmation(monkeypatch):
    from dataclasses import replace
    from hegel_machine.cognition import unified_structure
    from hegel_machine.cognition.object_continuation import encode_entry
    from test_cognition_object_complete_confirmation_budget import (
        ConstantObjectResponseLab, object_agent, paid_object_scene)

    laboratory = ConstantObjectResponseLab()
    previous = paid_object_scene(laboratory)
    schema = json.loads(laboratory.spec_for(("fine",)).context["object_schema"])
    parent = claim("retained_structure", Program("const", value=0.), schema)
    other = claim("retained_competitor", Program("const", value=3.), schema)
    for entry in (parent, other):
        entry["structure"]["hypotheses"] = []
    previous.update(selected_kind=parent["kind"], selected_structure=parent["structure"],
        object_continuation_state={"selected": encode_entry(parent),
            "finalists": [encode_entry(parent), encode_entry(other)],
            "schemas": [{"channels": ["fine"], "schema": schema}]})
    agent = object_agent(compute=200)
    agent.config = replace(agent.config, active_probes=True, risk_learning=False)

    def forbidden(*args, **kwargs):
        pytest.fail("unfunded paid revision must not start another fit or experiment")

    monkeypatch.setattr(unified_structure, "propose_structures", forbidden)
    monkeypatch.setattr(unified_structure, "structural_experiments", forbidden)
    report = agent.run(laboratory, previous=previous)

    assert report["status"] == "unresolved"
    pending = report["object_continuation_state"]["structure_comparison_pending"]
    assert pending["status"] == "compute_unresolved" and pending["rows"] == 12
    assert any(event["action"] == "defer_frozen_confirmation"
               and event.get("reason") == "paid_structure_comparison_pending" for event in report["actions"])
    assert not any(event["action"] == "acquire" for event in report["actions"])
    assert report["costs"]["observation_cost"] == previous["costs"]["observation_cost"]
    assert previous["costs"]["compute_cost"] <= report["costs"]["compute_cost"] <= 200
    pool = report["object_continuation_state"]["finalists"]
    programs = [entry["candidate"]["program"] for entry in pool]
    assert all(entry["candidate"].program.to_dict() in programs for entry in (parent, other))


def test_declared_response_research_uses_existing_solver_with_its_original_cap(monkeypatch):
    from hegel_machine.cognition import unified_structure
    from hegel_machine.cognition.structural_discovery import discover_structure
    from hegel_machine.cognition.structure_lab import StructureLab
    from test_cognition_response_basis_reopening import paid_trajectories, old_window_drive

    laboratory = StructureLab(readout="total_event", seed=71)
    rows = paid_trajectories(laboratory)
    schema = laboratory.object_schema
    setup = SearchBudget(45000)
    old = discover_structure(rows, schema, budget=setup, families=("joint_window_event",),
                             response_terms=old_window_drive(schema))
    prior_pool = unified_structure._entries(old, "family_revision", laboratory.spec, rows, setup, 1e-7)
    parent = prior_pool[0]
    original = deepcopy(prior_pool)
    source_solver = unified_structure.propose_response_revisions
    calls = []

    def real_response(spec, observations, budget, **kwargs):
        calls.append({"limit": budget.limit, "rows": list(observations), "parent": kwargs["parent"]})
        return source_solver(spec, observations, budget, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("a declared response revision must not silently execute structural research")

    monkeypatch.setattr(unified_structure, "propose_structures", forbidden)
    monkeypatch.setattr(unified_structure, "propose_response_revisions", real_response)
    work = SearchBudget(50000)
    result = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))._revise_paid_structure(
        laboratory.spec, parent, rows, work, [], finalists=prior_pool,
        research_action="response_basis_revision")

    assert len(calls) == 1 and calls[0]["limit"] == 32000
    assert calls[0]["rows"] == rows and calls[0]["parent"] is parent
    assert result["action"]["research_action"] == "response_basis_revision"
    assert result["action"]["new_scoring_reserve"] == 8 * len(rows)
    assert result["full_comparison"] and 0 < len(result["new_claims"]) <= 8
    assert all(entry["structure_search"] == "response_revision" for entry in result["new_claims"])
    assert {entry["candidate"].program for entry in result["claims"]} >= {
        entry["candidate"].program for entry in prior_pool}
    assert result["selected"]["validation_loss"] < result["action"]["prior_loss"]
    assert prior_pool == original and work.spent == result["compute_cost"] <= 50000


def test_unknown_research_action_is_not_replaced_by_a_default():
    spec, rows, schema = simple_scene()
    parent = claim("already_correct", Program("feature", name="x"), schema)
    budget, actions = SearchBudget(100), []
    with pytest.raises(ValueError, match="Unsupported paid structure research action"):
        UnifiedResearchAgent(ResearchConfig())._revise_paid_structure(
            spec, parent, rows, budget, actions, research_action="structural-research")
    assert budget.spent == 0 and actions == []
