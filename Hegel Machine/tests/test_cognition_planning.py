"""Behavioral checks for dependent revision and informative new observations."""

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.planning import RevisionPlan, execute_revision_plan
from hegel_machine.cognition.probing import choose_probe, construct_probes, observed_classes


def descendants(program):
    yield program
    for child in program.children:
        yield from descendants(child)


def test_later_stage_uses_learned_coordinate_and_improves_new_predictions():
    # A linear fit to |x| supplies a useful coordinate. Squaring that learned
    # coordinate improves cubic data beyond an independent raw x/x^2 search.
    rows = [Observation({"x": float(x)}, float(abs(x) ** 3))
            for x in (-3, -2, -1, 0, 1, 2, 3)]
    incumbent = Program("const", value=sum(row.target for row in rows) / len(rows))
    plan = RevisionPlan("dependent", "Learn a coordinate and then combine it",
                        ("representation", "composition"))
    budget = SearchBudget(6000)
    candidates = execute_revision_plan(plan, rows, incumbent=incumbent, budget=budget,
                                       risk_quantile=0.5)
    first_stage = [c for c in candidates if c.operation == "representation"]
    second_stage = [c for c in candidates if c.operation == "composition"]
    intermediate = min(first_stage, key=lambda c: loss(c.program, rows, 0.0, 0.5)).program
    winner = min(second_stage, key=lambda c: loss(c.program, rows, 0.0, 0.5)).program
    assert intermediate != winner and intermediate in tuple(descendants(winner))

    independent = execute_revision_plan(
        RevisionPlan("independent", "Combine raw observations", ("composition",)),
        rows, incumbent=incumbent, budget=SearchBudget(6000), risk_quantile=0.5)
    independent_winner = min(independent, key=lambda c: loss(c.program, rows, 0.0, 0.5)).program
    held_out = [Observation({"x": x}, abs(x) ** 3) for x in (-2.5, -0.5, 0.5, 2.5)]
    assert loss(winner, held_out, 0.0, 0.5) < loss(independent_winner, held_out, 0.0, 0.5)
    assert loss(winner, rows, 0.0, 0.5) < loss(intermediate, rows, 0.0, 0.5)
    assert 0 < budget.spent <= budget.limit


def test_worse_intermediate_relations_survive_until_their_combination_succeeds():
    # Each one-variable predicate is worse than the fitted constant. Their OR
    # is exact, so a rule requiring immediate improvement would stop too early.
    rows = [Observation({"x": float(x), "z": float(z)}, float(x or z))
            for x in (0, 1) for z in (0, 1)]
    incumbent = Program("const", value=0.75)
    plan = RevisionPlan("relations", "Retain predicates to test their composition",
                        ("relation", "composition"))
    candidates = execute_revision_plan(plan, rows, incumbent=incumbent,
                                       budget=SearchBudget(3000), risk_quantile=0.5)
    first_stage = [c for c in candidates if c.operation == "relation"]
    second_stage = [c for c in candidates if c.operation == "composition"]
    assert first_stage and second_stage
    assert min(loss(c.program, rows, 0.0, 0.5) for c in first_stage) > loss(incumbent, rows, 0.0, 0.5)
    winner = min(second_stage, key=lambda c: loss(c.program, rows, 0.0, 0.5)).program
    assert loss(winner, rows, 0.0, 0.5) == 0.0
    assert any(c.program in tuple(descendants(winner)) for c in first_stage)


def test_constructed_probe_splits_previously_equal_x_and_x_squared():
    linear = Program("feature", name="x")
    square = Program("mul", children=(linear, linear))
    candidates = [Candidate("linear", linear, "representation"),
                  Candidate("square", square, "composition")]
    observed = [Observation({"x": x}, x * x) for x in (0.0, 1.0)]
    budget = SearchBudget(2000)
    before = observed_classes(candidates, observed, 0.0, budget)
    assert before == [["linear", "square"]]

    spec = TaskSpec("queryable", "Predict an observed scalar", ("x",), (), (),
                    feature_bounds={"x": (0.0, 2.0)}, max_probe_length=4)
    constructed = construct_probes(spec, candidates, round_number=0, seed=7, max_history=1)
    probe, disagreement = choose_probe(constructed, candidates, 0.0, budget)
    assert probe is not None and disagreement > 0.0
    fresh = [Observation(dict(point), point["x"] ** 2) for point in probe.sequence]
    after = observed_classes(candidates, observed + fresh, 0.0, budget)
    assert sorted(after) == [["linear"], ["square"]]
    assert any(point["x"] not in (0.0, 1.0) for point in probe.sequence)
    assert len(candidates) == 2
