"""A historical transition learner must deliver control-only closed predictors."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.legacy_research_bridge import InquiryResearchLab
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition import state_search


def _record(values, targets, offset):
    return {"row_indices": list(range(offset, offset + len(values))),
            "rows": [{"features": {"drive": value}, "target": target}
                     for value, target in zip(values, targets)]}


def _transition_records(levels):
    records, offset = [], 11
    for values in ((-1., 0.), (1., 0.), (-1.2, 1.2, .2, -.2, -1.2),
                   (1.2, .4, -.4, -1.2, 0., .8), (0., .8, 0., -.8, 0.),
                   (-.8, .8, .5, -.5, -.8), (.8, .2, -.2, -.8)):
        active, targets = False, []
        for value in values:
            active = value >= .6 or (active and value >= -.6)
            targets.append(levels[int(active)])
        records.append(_record(values, targets, offset))
        offset += len(values)
    return records


@pytest.mark.parametrize("levels", ((0., 1.), (-2., 3.)))
def test_learns_closed_transition_from_records_and_retains_memoryless_competitor(levels):
    records = _transition_records(levels)
    indices = list(range(20, 20 + len(records)))
    budget = SearchBudget(4000)
    claims = state_search.closed_state_claims(records, budget=budget, record_indices=indices)
    winner = min(claims, key=lambda item: (item["training_loss"], item["candidate"].program.size))
    program = winner["candidate"].program
    assert program.op == "state_replay"
    assert winner["training_loss"] == 0.
    assert all(name != "__target__" for name, _ in program.observed_variables())
    assert any(item["candidate"].program.op != "state_replay" for item in claims)
    for reset in (-1., 1.):
        # The model must reproduce different states from controls alone.
        row = Observation({"drive": 0.}, 123., ({"drive": reset},))
        assert program.predict(row) == levels[int(reset > 0)]
    receipt = winner["state_research"]
    assert receipt["fit_record_indices"] == indices
    assert receipt["fit_evidence"] == [index for record in records for index in record["row_indices"]]
    assert receipt["fit_rows"] == sum(len(record["rows"]) for record in records)
    assert winner["research_program"] is None
    assert receipt["independent_confirmation"] is False
    assert budget.spent == receipt["shared_search_compute"] + sum(
        item["state_research"]["candidate_training_compute"] for item in claims)


def test_training_loss_replays_own_predictions_instead_of_observed_past_labels(monkeypatch):
    records = [_record([0., 0., 0.], [1., 1., 1.], 7),
               _record([0., 0.], [0., 0.], 41)]
    body = Program("if", children=(Program("available", lag=1),
        Program("lag", name="__target__", lag=1), Program("const", value=0.)))
    captured = []

    def learn(rows, *, budget, **kwargs):
        captured.extend(rows)
        budget.spend(9)
        return [Candidate("bad_start", body, "composition")]

    monkeypatch.setattr(state_search, "state_candidates", learn)
    budget = SearchBudget(1000)
    claim, = state_search.closed_state_claims(records, budget=budget, max_candidates=1)
    # The historical learner sees paid labels and each true reset boundary.
    assert captured[1].history[0]["__target__"] == 1.
    assert captured[3].history == ()
    # Teacher forcing would miss only the first positive row (1/5). A zero
    # startup persists under closed execution and misses all three (3/5).
    assert claim["training_loss"] == pytest.approx(3 / 5)
    assert claim["state_research"]["replayed_prefix_steps"] == 9
    assert budget.spent == 9 + 5


def test_reserves_complete_candidate_comparison_before_starting_search(monkeypatch):
    records = _transition_records((0., 1.))
    rows = [row for record in records for row in InquiryResearchLab.measured_rows(record)]
    required = state_search.state_search_work(rows) + 8 * len(rows)
    monkeypatch.setattr(state_search, "state_candidates", lambda *args, **kwargs: pytest.fail(
        "an unfunded closed comparison must not start the transition fit"))
    budget = SearchBudget(required - 1)
    assert state_search.closed_state_claims(records, budget=budget) == []
    assert budget.spent == 0
    assert budget.last_denied_request["required_compute"] == required


def test_failed_search_keeps_actual_spending(monkeypatch):
    def fail(rows, *, budget, **kwargs):
        budget.spend(13)
        raise RuntimeError("interrupted fit")

    monkeypatch.setattr(state_search, "state_candidates", fail)
    budget = SearchBudget(4000)
    with pytest.raises(RuntimeError, match="interrupted fit"):
        state_search.closed_state_claims(_transition_records((0., 1.)), budget=budget)
    assert budget.spent == 13


def test_continuous_responses_do_not_enter_discrete_search(monkeypatch):
    monkeypatch.setattr(state_search, "state_candidates", lambda *args, **kwargs: pytest.fail(
        "the discrete learner requires two observed output levels"))
    budget = SearchBudget(1000)
    assert state_search.closed_state_claims([_record([0., 1., 2.], [.1, .2, .3], 0)],
                                           budget=budget) == []
    assert budget.spent == 0
