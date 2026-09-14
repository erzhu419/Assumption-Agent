"""Persistent research must resume computation, not merely retain a tree."""

import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, ResearchQuestion, SearchBudget
from hegel_machine.cognition.planning import execute_revision_plan
from hegel_machine.cognition.research_graph import ResearchGraph


def question(*operations):
    return ResearchQuestion("Explain the remaining error", {"squared_residual": 1.0}, operations)


def restore(graph):
    return ResearchGraph.from_dict(json.loads(json.dumps(graph.to_dict())))


def descendants(program):
    yield program
    for child in program.children:
        yield from descendants(child)


def test_budget_interruption_resumes_the_next_step_with_real_intermediates():
    rows = [Observation({"x": float(x), "z": float(z)}, float(x or z))
            for x in (0, 1) for z in (0, 1)]
    graph = ResearchGraph().update(question("relation"), evidence_count=len(rows))
    plan = graph.pending_plans()[0]
    incumbent = Program("const", value=0.75)
    first = execute_revision_plan(plan, rows, incumbent=incumbent,
                                  budget=SearchBudget(3000), risk_quantile=0.5, max_steps=1)
    assert first and plan.next_step == 1 and plan.status == "paused"
    saved_frontier = list(plan.frontier)
    graph.record(plan)

    graph = restore(graph)
    graph.update(question("relation"), evidence_count=len(rows))
    plan = graph.pending_plans()[0]
    assert execute_revision_plan(plan, rows, incumbent=incumbent,
                                 budget=SearchBudget(0), risk_quantile=0.5) == []
    assert plan.next_step == 1 and plan.frontier == saved_frontier
    assert plan.steps[-1]["status"] == "budget_exhausted"
    graph.record(plan)

    graph = restore(graph)
    plan = graph.pending_plans()[0]
    next_event = len(plan.steps)
    second = execute_revision_plan(plan, rows, incumbent=incumbent,
                                   budget=SearchBudget(3000), risk_quantile=0.5)
    assert {candidate.operation for candidate in second} == {"composition"}
    winner = min(second, key=lambda candidate: loss(candidate.program, rows, 0.0, 0.5))
    assert loss(winner.program, rows, 0.0, 0.5) == 0.0
    assert any(program in tuple(descendants(winner.program)) for program in saved_frontier)
    assert [step["step"] for step in plan.steps[next_event:]] == [1]
    graph.record(plan)
    assert not graph.pending_plans()


def test_new_evidence_resolves_and_reopens_a_previously_completed_branch():
    graph = ResearchGraph().update(question("refit"), evidence_count=4)
    plan = graph.pending_plans()[0]
    rows = [Observation({"x": float(x)}, float(2 * x)) for x in range(4)]
    execute_revision_plan(plan, rows, incumbent=Program("const", value=0),
                          budget=SearchBudget(300), risk_quantile=0.5)
    graph.record(plan)
    assert plan.status == "completed"
    graph.update(question("refit"), evidence_count=4)
    assert graph.pending_plans() == []

    graph.update(question(), evidence_count=8)
    assert graph.branches["refit"]["plan"].status == "resolved"
    assert graph.pending_plans() == []
    graph = restore(graph)
    graph.update(question("refit", "relation"), evidence_count=12)
    assert {plan.plan_id for plan in graph.pending_plans()} == {"refit", "relation_then_composition"}
    assert graph.branches["refit"]["generation"] == 1
    assert graph.branches["refit"]["plan"].next_step == 0
    assert graph.branches["refit"]["history"][0]["frontier"]


def test_more_evidence_reopens_unresolved_work_but_preserves_a_paused_dependency():
    rows = [Observation({"x": float(x)}, float(x * x)) for x in range(4)]
    graph = ResearchGraph().update(question("refit", "representation"), evidence_count=4)
    for plan in graph.pending_plans():
        execute_revision_plan(plan, rows, incumbent=Program("const", value=0),
                              budget=SearchBudget(400), risk_quantile=0.5, max_steps=1)
        graph.record(plan)
    graph = restore(graph)
    graph.update(question("refit", "representation"), evidence_count=8)
    plans = graph.pending_plans()
    assert plans[0].plan_id == "representation_then_relation"
    assert plans[0].next_step == 1
    assert plans[1].plan_id == "refit" and plans[1].next_step == 0
    assert graph.branches["refit"]["generation"] == 1
    assert graph.branches["representation_then_relation"]["generation"] == 0


def test_changed_dependent_operations_replace_the_saved_branch():
    graph = ResearchGraph().update(question("representation"), evidence_count=4, depth=1)
    graph.update(question("representation"), evidence_count=8, depth=2)
    plan = graph.pending_plans()[0]
    assert plan.operations == ("representation", "composition")
    assert graph.branches[plan.plan_id]["generation"] == 1
    assert graph.events[-1]["event"] == "replaced"
