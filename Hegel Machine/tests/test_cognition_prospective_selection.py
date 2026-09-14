"""Fresh observations and executable structures must change research decisions."""

from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.planning import RevisionPlan, execute_revision_plan
from hegel_machine.cognition.selection import (
    PlanAllocation, rank_candidates, representative_probe, warm_start_candidates,
)
from hegel_machine.cognition.sparse_search import sparse_candidates


def test_identical_program_origins_share_work_without_excluding_a_competitor():
    x = Program("feature", name="x")
    candidates = [Candidate("source_a", x, "abstract_reuse", memory_id="a"),
                  Candidate("source_b", x, "abstract_reuse", memory_id="b"),
                  Candidate("competitor", Program("const", value=0.0), "refit")]
    rows = [Observation({"x": x}, x) for x in (-1.0, 0.0, 1.0)]
    budget = SearchBudget(6)
    assessed = rank_candidates(candidates, rows, [], fallback=0.0, q=0.5,
                               budget=budget, complexity_weight=0.0)
    assert len(assessed) == 2 and budget.spent == 6
    assert {origin.memory_id for origin in assessed[0].origins} == {"a", "b"}


def test_representative_controls_separate_explanations_hidden_by_collinear_evidence():
    x, z = Program("feature", name="x"), Program("feature", name="z")
    candidates = [Candidate("x", x, "refit"), Candidate("z", z, "refit")]
    training = [Observation({"x": value, "z": value}, value) for value in (-1.0, 0.0, 1.0)]
    assert loss(x, training, 0.0, 0.5) == loss(z, training, 0.0, 0.5) == 0.0
    spec = TaskSpec("observed", "Predict a scalar", ("x", "z"), (), (),
                    feature_bounds={"x": (-1.5, 1.5), "z": (-1.5, 1.5)})
    probe = representative_probe(spec, seed=19, index=0, remaining=8)
    assert probe is not None and probe.cost == 8
    fresh = [Observation(point, point["x"]) for point in probe.sequence]
    # Each control covers the domain while their independent permutations
    # expose which of the formerly identical explanations predicts new data.
    for name in spec.feature_names:
        strata = {int((row.features[name] + 1.5) / 3.0 * 8) for row in fresh}
        assert strata == set(range(8))
    ranked = rank_candidates(candidates, training, [], fallback=0.0, q=0.5,
                             budget=SearchBudget(1000), complexity_weight=0.0, fresh=fresh)
    assert ranked[0].candidate.candidate_id == "x"
    assert ranked[0].fresh_loss == 0.0 < ranked[1].fresh_loss


def test_learned_startup_uses_current_signals_until_mature_history_is_available():
    x = Program("feature", name="x")
    mature = Program("add", children=(
        Program("mul", children=(Program("const", value=0.3), x)),
        Program("lag", name="x", lag=2)))
    candidate = Candidate("historical", mature, "representation")
    training = []
    for sequence in ((-2.0, -1.0, 0.5), (0.0, 1.0, -0.5), (2.0, 3.0, 1.5)):
        history = []
        for value in sequence:
            target = 2.0 * value + 1.0 if len(history) < 2 else 0.3 * value + history[-2]["x"]
            training.append(Observation({"x": value}, target, tuple(history)))
            history.append({"x": value, "__target__": target})
    budget = SearchBudget(2000)
    generated = warm_start_candidates([candidate], training, budget=budget,
                                      q=0.5, complexity_weight=0.0)
    assert generated and 0 < budget.spent <= budget.limit
    winner = min(generated, key=lambda c: loss(c.program, training, 0.0, 0.5)).program
    assert winner.minimum_history_depth == 0 and winner.history_depth == 2
    # Direct AST execution proves the guard selects its learned startup branch;
    # a global MissingHistory fallback could not produce these predictions.
    for history in ((), ({"x": -0.75, "__target__": -0.5},)):
        assert winner.predict(Observation({"x": 0.25}, 0.0, history)) == pytest.approx(1.5)
    mature_row = Observation({"x": 0.25}, 0.0,
                             ({"x": -0.75, "__target__": -0.5}, {"x": 4.0, "__target__": 9.0}))
    assert winner.predict(mature_row) == pytest.approx(mature.predict(mature_row))
    assert winner.predict(mature_row) != pytest.approx(1.5)


def _nonlinear_history_rows(seed, count):
    rng, history, rows = Random(seed), [], []
    for _ in range(count + 2):
        x, z = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        past = history[-2]["x"] if len(history) >= 2 else 0.0
        target = 0.6 * x + 0.9 * x * x - 0.5 * x * z + 1.1 * past
        if len(history) >= 2:
            rows.append(Observation({"x": x, "z": z}, target, tuple(history)))
        history.append({"x": x, "z": z, "__target__": target})
    return rows


def test_sparse_composition_generalizes_joint_history_nonlinearity_and_interaction():
    training = _nonlinear_history_rows(11, 96)
    fresh = _nonlinear_history_rows(37, 48)
    budget = SearchBudget(6000)
    candidates = sparse_candidates(training, budget=budget, max_history=2, max_terms=4)
    assert candidates and 0 < budget.spent <= budget.limit
    # Choose using training evidence alone, then score a different trajectory.
    winner = min(candidates, key=lambda c: loss(c.program, training, 0.0, 0.5)).program
    assert loss(winner, fresh, 0.0, 0.5) < 1e-12
    current_only = sparse_candidates(training, budget=SearchBudget(6000), max_history=0, max_terms=4)
    current_winner = min(current_only, key=lambda c: loss(c.program, training, 0.0, 0.5)).program
    assert loss(current_winner, fresh, 0.0, 0.5) > 0.2


def test_measured_fresh_gain_per_actual_plan_cost_changes_next_allocation():
    training = [Observation({"x": value}, value ** 2) for value in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    fresh = [Observation({"x": value}, value ** 2) for value in (-1.25, -0.75, 0.75, 1.25)]
    incumbent = Program("const", value=2.0)
    plans = [RevisionPlan("linear", "Refit a linear explanation", ("refit",)),
             RevisionPlan("curved", "Compose a nonlinear explanation", ("composition",))]
    allocation = PlanAllocation()
    assert allocation.allocate(plans, 1000) == [500, 500]
    candidates = []
    for plan in plans:
        candidates.extend(execute_revision_plan(plan, training, incumbent=incumbent,
                                                budget=SearchBudget(1000), risk_quantile=0.5))
        assert sum(step["compute_cost"] for step in plan.steps) > 0
    assessed = rank_candidates(candidates, training, [], fallback=2.0, q=0.5,
                               budget=SearchBudget(1000), complexity_weight=0.0, fresh=fresh)
    baseline = loss(incumbent, fresh, 2.0, 0.5)
    allocation.update(plans, assessed, baseline)
    revised = allocation.allocate(plans, 1000)
    assert revised[1] > revised[0] and sum(revised) <= 1000
    assert allocation.experience["curved"] > allocation.experience["linear"]
