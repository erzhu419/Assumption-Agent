"""Screening keeps acquired contradictions alongside representative rows."""

from random import Random

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.selection import screen_candidates


def task_rows():
    count = 60
    sampled = set(Random(count + 9301).sample(range(count), 12))
    # The public response is x². The already acquired x=2 region falsifies
    # the cheap x and abs(x) explanations, but this random sample misses it.
    return [Observation({"x": float(index % 2) if index in sampled else 2.},
                        float(index % 2)**2 if index in sampled else 4.) for index in range(count)]


def hypotheses():
    x = Program("feature", name="x")
    return [Candidate("current", x, "refit"),
            Candidate("equally_cheap", Program("abs", children=(x,)), "refit"),
            Candidate("explains_counterexamples", Program("mul", children=(x, x)), "composition")]


def test_existing_counterexamples_keep_the_explanation_that_the_random_screen_misses():
    rows, candidates = task_rows(), hypotheses()
    ordinary_budget = SearchBudget(120)
    ordinary = screen_candidates(candidates, rows, [], fallback=0., q=.5, budget=ordinary_budget,
                                 complexity_weight=.001, maximum=2)
    assert all(item.candidate.candidate_id != "explains_counterexamples" for item in ordinary)
    budget = SearchBudget(120)
    retained = screen_candidates(candidates, rows, [], fallback=0., q=.5, budget=budget,
        complexity_weight=.001, maximum=2, incumbent=candidates[0].program)
    assert retained[0].candidate.candidate_id == "explains_counterexamples"
    assert retained[0].evidence_loss == 0.
    # One real incumbent pass plus the unchanged twelve-row sample per model.
    assert budget.spent == len(rows) + ordinary_budget.spent <= budget.limit
    # Four witnesses coexist with eight ordinary sampled points: the current
    # model's reported sample loss is below its all-hard-case loss of four.
    current = next(item for item in retained if item.candidate.candidate_id == "current")
    assert 0. < current.evidence_loss < 4.


def test_small_budget_keeps_candidate_comparisons_and_full_samples_do_not_need_an_extra_scan():
    rows, candidates = task_rows(), hypotheses()
    budget = SearchBudget(30)
    retained = screen_candidates(candidates, rows, [], fallback=0., q=.5, budget=budget,
        complexity_weight=.001, maximum=2, incumbent=candidates[0].program)
    assert retained and budget.spent <= 30
    small_rows = rows[:3]
    budget = SearchBudget(30)
    result = screen_candidates(candidates, small_rows, [], fallback=0., q=.5, budget=budget,
        complexity_weight=.001, maximum=2, incumbent=candidates[0].program)
    assert result and budget.spent == len(small_rows) * len(candidates)
