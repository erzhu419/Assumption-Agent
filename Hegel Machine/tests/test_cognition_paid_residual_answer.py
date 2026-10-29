"""Paid child answers must change the parent without another synthesis run."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def claim(name, program):
    return {"candidate": Candidate(name, program, "retained_response"),
            "kind": "retained_response", "training_loss": 0., "validation_loss": 0., "fallback": 0.}


def support(baseline, correction):
    return {"reason": "predictable_residual", "counterexample_baseline_program": baseline.to_dict(),
            "structural_counterexample": {"kind": "measured_correction", "gain": 10.,
                                          "program": correction.to_dict()}}


@pytest.fixture
def agent(monkeypatch):
    from hegel_machine.cognition import research_graph

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("consuming an already executable answer must not fit again")

    monkeypatch.setattr(research_graph, "synthesize", unexpected_fit)
    return UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))


def test_paid_correction_keeps_its_original_baseline_and_entire_pool(agent):
    x = Program("feature", name="x")
    correction = Program("add", children=(x, Program("const", value=1.)))
    current = claim("later_parent", Program("mul", children=(Program("const", value=2.), x)))
    other_view = claim("other_instrument", Program("feature", name="unobserved"))
    original = deepcopy([current, other_view])
    rows = [Observation({"x": value}, 2 * value + 1) for value in (-3., -1., 2., 4.)]
    # Repeated measurements are separate paid observations, not deduplicated data.
    rows.append(rows[0])
    spec = TaskSpec("paid", "measured correction", ("x",), (), ())
    graph = ResearchGraph()
    graph.add_question("unrelated", "This pending question must not execute now")
    budget, actions = SearchBudget(25), []
    result = agent._consume_residual_answer(spec, support(x, correction), rows,
        budget, actions, graph, parent=current, finalists=[current, other_view])

    assert result["status"] == "improved"
    assert loss(result["selected"]["candidate"].program, rows, 0., .5) == 0.
    assert result["action"]["baseline_matches_parent"] is False
    assert result["action"]["requires_fresh_confirmation"] is True
    assert result["action"]["uncompared_candidate_ids"] == ["other_instrument"]
    assert budget.spent == result["compute_cost"] == 25
    assert budget.completed_fits == 0
    assert graph.nodes["unrelated"].status == "pending"
    assert len(result["compared_claims"]) == 2
    assert any(item is other_view for item in result["claims"])
    assert [current, other_view] == original
    assert actions == [result["action"]]


def test_paid_answer_compares_all_rows_instead_of_the_old_favorable_fold(agent):
    baseline = Program("const", value=0.)
    current = claim("broader_evidence_parent", Program("const", value=5.))
    rows = [Observation({"x": float(i)}, target) for i, target in enumerate((0., 0., 10., 10.))]
    spec = TaskSpec("paid", "different observed trajectories", ("x",), (), ())
    result = agent._consume_residual_answer(spec, support(baseline, baseline), rows,
        SearchBudget(20), [], ResearchGraph(), parent=current)
    assert result["status"] == "no_improvement"
    assert result["selected"]["candidate"] == current["candidate"]
    assert result["action"]["prior_loss"] == 25.
    assert result["action"]["answer_loss"] == 50.
    assert result["action"]["requires_fresh_confirmation"] is False


def test_paid_answer_keeps_a_truly_better_old_candidate(agent):
    x = Program("feature", name="x")
    current = claim("exact_old", x)
    rows = [Observation({"x": value}, value) for value in (-1., 1.)]
    result = agent._consume_residual_answer(
        TaskSpec("paid", "preserve correct explanation", ("x",), (), ()),
        support(x, Program("const", value=.5)), rows, SearchBudget(10), [], ResearchGraph(), parent=current)
    assert result["status"] == "no_improvement"
    assert result["selected"]["candidate"] == current["candidate"]
    assert result["action"]["answer_loss"] == .25


def test_unfunded_paid_answer_preserves_pool_and_graph_without_partial_work(agent):
    x = Program("feature", name="x")
    current = claim("old", x)
    rows = [Observation({"x": 1.}, 2.), Observation({"x": 2.}, 3.)]
    graph, budget = ResearchGraph(), SearchBudget(12, reserve=3)
    result = agent._consume_residual_answer(
        TaskSpec("paid", "reserve independent confirmation", ("x",), (), ()),
        support(x, Program("const", value=1.)), rows, budget, [], graph, parent=current)
    assert result["status"] == "compute_unresolved"
    assert result["action"]["compute_required"] == 10
    assert budget.spent == 0 and budget.reserve == 3
    assert not graph.nodes
    assert result["selected"] is current and result["claims"] == [current]


def test_unowned_historical_correction_is_not_rebound_to_current_parent(agent):
    x = Program("feature", name="x")
    current = claim("later", x)
    evidence = support(x, Program("const", value=1.))
    evidence.pop("counterexample_baseline_program")
    budget = SearchBudget(100)
    result = agent._consume_residual_answer(
        TaskSpec("paid", "old counterexample without owner", ("x",), (), ()), evidence,
        [Observation({"x": 1.}, 2.)], budget, [], ResearchGraph(), parent=current)
    assert result["status"] == "unavailable"
    assert result["selected"] is current and budget.spent == 0


def test_nonrecursive_ablation_does_not_execute_the_paid_child():
    agent = UnifiedResearchAgent(ResearchConfig(max_question_depth=0))
    x = Program("feature", name="x")
    parent, graph, budget = claim("old", x), ResearchGraph(), SearchBudget(100)
    result = agent._consume_residual_answer(
        TaskSpec("paid", "nonrecursive control", ("x",), (), ()),
        support(x, Program("const", value=1.)), [Observation({"x": 1.}, 2.)],
        budget, [], graph, parent=parent)
    assert result["status"] == "recursion_disabled"
    assert result["selected"] is parent and budget.spent == 0 and not graph.nodes
