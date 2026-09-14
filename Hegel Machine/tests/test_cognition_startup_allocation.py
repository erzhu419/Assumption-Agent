"""A late deep-history explanation must receive its startup fit budget."""

from itertools import combinations
from random import Random

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.selection import warm_start_candidates


def _sum_programs(programs):
    result = programs[0]
    for program in programs[1:]:
        result = Program("add", children=(result, program))
    return result


def _scale(coefficient, program):
    return Program("mul", children=(Program("const", value=coefficient), program))


def _observations(sequences):
    result = []
    for sequence in sequences:
        history = []
        for x, z in sequence:
            target = 0.2 * x - 0.87 * x * x
            if len(history) >= 4:
                target -= 0.79 * history[-4]["x"]
            result.append(Observation({"x": x, "z": z}, target, tuple(history)))
            history.append({"x": x, "z": z, "__target__": target})
    return result


def test_shallow_candidate_pool_cannot_starve_correct_four_step_startup_fit():
    # The acquired f08 failure had three constant 3-row initial sequences,
    # one independent 8-row sequence, and one 6-row single-control sequence.
    # Keep that compact observation pattern with the same kind of causal law.
    independent = [(-0.917, 1.739), (1.762, 0.809), (1.378, -1.368), (0.283, 0.558),
                   (-2.095, -0.180), (-1.427, -0.734), (0.927, 1.538), (-0.091, -1.702)]
    training = _observations([
        *[[(value, value)] * 3 for value in (-0.75, 0.25, 0.9)],
        independent, [(value, 0.0) for value in (-2.25, -1.35, -0.45, 0.45, 1.35, 2.25)],
    ])
    assert len(training) == 23
    assert sum(len(row.history) < 4 for row in training) == 17

    x, z = Program("feature", name="x"), Program("feature", name="z")
    square = Program("mul", children=(x, x))
    basis = (x, z, square, Program("mul", children=(z, z)),
             Program("mul", children=(x, z)), Program("abs", children=(x,)))
    pool = [Candidate("initial", Program("const", value=0.0), "refit")]
    # Several distinct current-coordinate supports at each shallow depth
    # precede the correct candidate. Generic shallow searches used to consume
    # all 500 units before the fourth-depth model's 17-row startup fit.
    for depth in (1, 2, 3):
        for index, terms in enumerate(combinations(basis, 2)):
            program = _sum_programs([*terms, Program("lag", name="x", lag=depth)])
            pool.append(Candidate(f"shallow:{depth}:{index}", program, "representation"))
    mature = _sum_programs([_scale(0.2, x), _scale(-0.87, square),
                            _scale(-0.79, Program("lag", name="x", lag=4))])
    pool.append(Candidate("correct_deep", mature, "composition"))

    budget = SearchBudget(500)
    generated = warm_start_candidates(pool, training, budget=budget, q=0.5, complexity_weight=0.001)
    complete = [candidate.program for candidate in generated
                if candidate.candidate_id == "startup:correct_deep"]
    assert complete and 0 < budget.spent <= 500
    winner = min(complete, key=lambda program: loss(program, training, 0.0, 0.5))
    assert winner.minimum_history_depth == 0 and winner.history_depth == 4
    assert winner.children[1] == mature
    assert loss(winner, training, 0.0, 0.5) < 1e-12
    assert loss(mature, training, 0.0, 0.5) > 0.1

    rng = Random(47)
    fresh = _observations([[(rng.uniform(-2.25, 2.25), rng.uniform(-2.25, 2.25))
                            for _ in range(12)] for _ in range(3)])
    # Both learned startup predictions and the unchanged mature relation must
    # execute on independent input histories; existence of a guard is not enough.
    assert loss(winner, [row for row in fresh if len(row.history) < 4], 0.0, 0.5) < 1e-12
    assert loss(winner, [row for row in fresh if len(row.history) >= 4], 0.0, 0.5) < 1e-12
    assert {1, 2, 3, 4} <= {candidate.program.children[0].lag for candidate in generated}
