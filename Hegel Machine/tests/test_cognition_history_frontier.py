"""Retain mature alternatives until paid trajectories can distinguish them."""

import json
from pathlib import Path
from random import Random

from hegel_machine.cognition.branch_support import history_branch_support
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import (
    rank_candidates, representative_probe, retain_history_frontier, screen_candidates,
)


CASES = json.loads((Path(__file__).parent / "fixtures" /
                   "cognition_v018_history_retention_learning.json").read_text())["cases"]


def rows(case, key):
    return [Observation(row["features"], row["target"], tuple(row["history"])) for row in case[key]]


def assess(candidates, observed, *, fresh=()):
    budget = SearchBudget(len(candidates) * (len(observed) + len(fresh)))
    result = rank_candidates(candidates, observed, [], fallback=0., q=.5, budget=budget,
                             complexity_weight=.001, fresh=fresh)
    assert budget.spent == budget.limit
    return result


def f07_candidates():
    case = CASES[0]
    current = Program.from_dict(case["first_selected_program"])
    historical = Program.from_dict(case["historical_program"])
    return [Candidate("current_r", current, "refit"),
            Candidate("current_s", current.remap_features({"r": "s"}), "refit"),
            Candidate("mature_r", historical.remap_features({"x": "r"}), "abstract_reuse", memory_id="source"),
            Candidate("mature_s", historical.remap_features({"x": "s"}), "abstract_reuse", memory_id="source")]


def test_paid_f07_startup_keeps_a_complete_explanation_and_lengthens_its_comparison():
    initial, fresh = rows(CASES[0], "initial"), rows(CASES[0], "first_reference")
    ranked = assess(f07_candidates(), initial)
    assert [row.candidate.candidate_id for row in ranked[:2]] == ["current_r", "current_s"]
    assert 0. < ranked[-1].evidence_loss < 1e-12
    selected, details = retain_history_frontier(ranked, initial, maximum=2)
    assert [row.candidate.candidate_id for row in selected] == ["current_r", "mature_r"]
    assert details["history_frontier"][0]["unresolved"][0]["reason"] == "history_not_observed"
    spec = TaskSpec("public_controls", "Predict the response", ("r", "s"), (), (),
                    feature_bounds={"r": (-1.5, 1.5), "s": (-1.5, 1.5)}, max_probe_length=12)
    ordinary = representative_probe(spec, seed=17, index=0, remaining=40,
                                    candidates=[row.candidate for row in ranked[:2]])
    proposed = representative_probe(spec, seed=17, index=0, remaining=40,
                                    candidates=[row.candidate for row in selected])
    assert ordinary.cost == 8 < proposed.cost == 12
    compared = assess([row.candidate for row in selected], initial, fresh=fresh)
    assert compared[0].candidate.candidate_id == "mature_r"
    assert compared[0].fresh_loss < 1e-12 < compared[1].fresh_loss
    assert not history_branch_support(compared[0].candidate.program, [*initial, *fresh])["unresolved"]


def test_full_startup_screen_uses_paid_scores_and_two_slots_are_enough():
    initial, candidates = rows(CASES[0], "initial"), f07_candidates()
    budget = SearchBudget(len(initial) * len(candidates))
    selected = screen_candidates(candidates, initial, [], fallback=0., q=.5, budget=budget,
                                  complexity_weight=.001, maximum=2)
    assert [row.candidate.candidate_id for row in selected] == ["current_r", "mature_r"]
    assert budget.spent == len(initial) * len(candidates)
    # A one-slot budget must still keep the best current explanation.
    ranked = assess(candidates, initial)
    single, details = retain_history_frontier(ranked, initial, maximum=1)
    assert single == ranked[:1] and not details["history_frontier"]


def test_sampled_screen_pays_for_only_the_history_branch_it_can_promote():
    observed, candidates = rows(CASES[0], "initial") * 4, f07_candidates()
    budget = SearchBudget(120)
    selected = screen_candidates(candidates, observed, [], fallback=0., q=.5, budget=budget,
                                  complexity_weight=.001, maximum=2)
    assert [row.candidate.candidate_id for row in selected] == ["current_r", "mature_r"]
    assert budget.spent == 12 * len(candidates) + len(observed)
    poor_budget = SearchBudget(12 * len(candidates))
    limited = screen_candidates(candidates, observed, [], fallback=0., q=.5, budget=poor_budget,
                                 complexity_weight=.001, maximum=2)
    # One full branch check plus a three-row screen still fits this smaller
    # allowance; it cannot borrow the later experiment's comparison budget.
    assert poor_budget.spent == poor_budget.limit
    assert limited[0].candidate.candidate_id == "current_r"


def test_existing_counterexample_blocks_promotion_even_when_a_screen_misses_it():
    count = 36
    sampled = set(Random(count + 9301).sample(range(count), 12))
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    historical = Program("if", children=(Program("available", lag=4),
                                           Program("lag", name="x", lag=4), x))
    observed = [Observation({"x": float(index % 2) if index in sampled else 2.},
                             float(index % 2) if index in sampled else 4.) for index in range(count)]
    candidates = [Candidate("correct", square, "composition"),
                  Candidate("also_correct", Program("abs", children=(square,)), "composition"),
                  Candidate("unseen_mature_wrong_startup", historical, "abstract_reuse", memory_id="source")]
    budget = SearchBudget(72)
    retained = screen_candidates(candidates, observed, [], fallback=0., q=.5, budget=budget,
                                  complexity_weight=.001, maximum=2)
    assert [row.candidate.candidate_id for row in retained] == ["correct", "also_correct"]
    assert budget.spent == 12 * len(candidates) + len(observed)


def test_observed_lags_still_equal_to_current_inputs_need_a_distinguishing_trajectory():
    observed = rows(CASES[0], "initial")
    current = Program.from_dict(CASES[0]["first_selected_program"])
    delayed = current.remap_features({"r": "r"})
    coefficient = current.children[0]
    delayed = Program("if", children=(Program("available", lag=2),
        Program("mul", children=(coefficient, Program("lag", name="r", lag=2))), delayed))
    support = history_branch_support(delayed, observed)
    assert support["required_depth"] == support["observed_depth"] == 2
    assert support["unresolved"][0]["reason"] == "history_equals_current_input"
    assert support["unresolved"][0]["current_aliases"] == ["r", "s"]
    retained, details = retain_history_frontier(assess([
        Candidate("current", current, "refit"), Candidate("delayed", delayed, "representation")], observed),
        observed, maximum=2)
    assert details["history_frontier"][0]["candidate_id"] == "delayed"
    assert len(retained) == 2


def test_paid_n09_rejects_old_history_and_does_not_mistake_startup_guards_for_history():
    case = CASES[1]
    initial, fresh = rows(case, "initial"), rows(case, "first_reference")
    historical = Program.from_dict(case["historical_program"])
    repeated_current = Program.from_dict(case["first_selected_program"])
    assert not history_branch_support(repeated_current, initial)["unresolved"]
    assert history_branch_support(repeated_current, initial)["required_depth"] == 0
    candidates = [Candidate("flat_current", historical.children[2], "refit"),
                  Candidate("guarded_current", repeated_current, "abstract_reuse"),
                  Candidate("old_history", historical, "abstract_reuse", memory_id="old_source")]
    retained, before = retain_history_frontier(assess(candidates, initial), initial, maximum=2)
    assert any(row.candidate.candidate_id == "old_history" for row in retained)
    assert before["history_frontier"][0]["candidate_id"] == "old_history"
    measured = assess(candidates, initial, fresh=fresh)
    rejected = next(row for row in measured if row.candidate.candidate_id == "old_history")
    assert rejected.fresh_loss > 3.
    # Fresh counterexamples remove the privilege even if the caller still
    # holds the initial support description until appending its new rows.
    retained, after = retain_history_frontier(measured, initial, maximum=2)
    assert not after["history_frontier"]
    assert all(row.candidate.candidate_id != "old_history" for row in retained)


def test_callable_history_uses_its_real_guarded_mature_depth_for_the_probe():
    body = Program("if", children=(Program("available", lag=3),
        Program("lag", name="x", lag=1), Program("arg", name="now")))
    definition = Definition("guarded_mechanism", ("now",), body, "source")
    call = Program("call", name="guarded_mechanism", definition=definition,
                   children=(Program("feature", name="x"),))
    observed = [Observation({"x": 1.}, 1., ({"x": 2.}, {"x": 3.}))]
    assert call.history_depth == 1
    support = history_branch_support(call, observed)
    assert support["required_depth"] == 3 > support["observed_depth"]
    spec = TaskSpec("controls", "Predict", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)}, max_probe_length=16)
    probe = representative_probe(spec, seed=19, index=0, remaining=20,
                                  candidates=[Candidate("call", call, "reuse")])
    assert probe.cost == 16
