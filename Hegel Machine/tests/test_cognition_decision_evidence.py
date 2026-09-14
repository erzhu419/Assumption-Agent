"""Additional paid trajectories resolve a frozen complete decision question."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import decision_evidence
from hegel_machine.cognition.decision_evidence import (
    begin_decision_evidence, decision_evidence_quote, extend_decision_evidence,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.uncertainty import select_decision_plan
from hegel_machine.cognition.workspace import observation_from_dict


@pytest.fixture(scope="module")
def noisy_decisions():
    path = Path(__file__).parent / "fixtures" / "cognition_v021_n11_conflicting_decision_evidence.json"
    return json.loads(path.read_text())


def _from_case(case, *, comparison=None):
    return begin_decision_evidence(Program.from_dict(case["previous_program"]), case["previous_model"],
        Program.from_dict(case["candidate_program"]), case["candidate_model"],
        fallback=case["fallback"], costs=case["costs"], comparison=comparison)


def _trajectory(target, *, length=4):
    result, history = [], []
    for index in range(length):
        point = {"x": index / length}
        result.append(Observation(point, target, tuple(history)))
        history.append({**point, "__target__": target})
    return result


def _calibrated_pair():
    def model(offset, calibration):
        return {"kind": "global_interval", "action_rule": "empirical_cost_quantile",
                "action_offset": offset, "residual_interval": [offset, offset],
                "calibration_offset": calibration}
    # Predictions 0 and 2 produce actions .75 and 1.25 under their own models.
    return begin_decision_evidence(Program("const", value=0.), model(.25, .5),
        Program("const", value=2.), model(-.25, -.5), fallback=0., costs=(.9, .1))


def test_actual_n11_conflict_keeps_a_fundable_question_and_both_calibrations(noisy_decisions, monkeypatch):
    def already_paid(*args, **kwargs):
        raise AssertionError("importing paid comparisons must not predict again")

    monkeypatch.setattr(decision_evidence, "action_and_interval", already_paid)
    state = _from_case(noisy_decisions, comparison=noisy_decisions["comparison"])
    compared = state["comparison"]
    assert compared["trajectories"] == 2 and compared["rows"] == 16
    assert compared["gain"] == pytest.approx(.16263895418348953)
    assert compared["support_margin"] == pytest.approx(.3528970301614401)
    assert [trial["gain"] for trial in compared["trajectory_comparisons"]] == pytest.approx(
        [-.01380956089723051, .3390874692642096])
    assert compared["compute_cost"] == 32 and state["last_compute_cost"] == 0
    assert state["status"] == "continue" and not compared["adopted"]
    restored = json.loads(json.dumps(state))
    assert restored["previous_model"] == noisy_decisions["previous_model"]
    assert restored["candidate_model"] == noisy_decisions["candidate_model"]
    assert restored["previous_model"]["calibration_offset"] == pytest.approx(1.5218230537222892)
    assert restored["candidate_model"]["calibration_offset"] == pytest.approx(.6496443045924822)
    quote = decision_evidence_quote(restored, trajectory_length=8,
        observation_remaining=16, compute_remaining=32)
    assert quote["status"] == "continue" and quote["payable"] and quote["needs_evidence"]
    assert (quote["observation_cost"], quote["compute_cost"], quote["probe_lengths"]) == (16, 32, [8, 8])


def test_actual_n11_split_batches_match_original_comparison_without_rescoring(noisy_decisions, monkeypatch):
    rows = [observation_from_dict(row) for row in noisy_decisions["followup_rows"]]
    action = decision_evidence.action_and_interval
    calls = []

    def record_call(program, row, *args, **kwargs):
        calls.append(id(row))
        return action(program, row, *args, **kwargs)

    monkeypatch.setattr(decision_evidence, "action_and_interval", record_call)
    empty = _from_case(noisy_decisions)
    first_work = SearchBudget(16)
    first = extend_decision_evidence(empty, rows[:8], budget=first_work)
    assert first_work.spent == first["last_compute_cost"] == 16
    assert first["comparison"]["trajectories"] == 1 and first["status"] == "continue"
    assert empty["comparison"]["trajectories"] == 0
    paid = deepcopy(first["comparison"]["trajectory_comparisons"])
    calls.clear()
    second_work = SearchBudget(16)
    second = extend_decision_evidence(json.loads(json.dumps(first)), rows[8:], budget=second_work)
    assert calls == [id(row) for row in rows[8:] for _ in range(2)]
    assert second_work.spent == second["last_compute_cost"] == 16
    assert second["comparison"]["compute_cost"] == 32
    assert second["comparison"]["trajectory_comparisons"][:1] == paid
    for field in ("before_loss", "after_loss", "gain", "standard_error", "support_margin"):
        assert second["comparison"][field] == pytest.approx(noisy_decisions["comparison"][field])
    assert second["status"] == "continue" and not second["comparison"]["adopted"]


def test_conflicting_trials_then_actual_positive_evidence_adopt_the_frozen_complete_plan():
    state = _calibrated_pair()
    first_rows = _trajectory(.75) + _trajectory(1.75)
    first = extend_decision_evidence(state, first_rows, budget=SearchBudget(16))
    assert first["status"] == "continue"
    assert [trial["gain"] for trial in first["comparison"]["trajectory_comparisons"]] == pytest.approx([-.05, .45])
    assert first["comparison"]["gain"] < first["comparison"]["support_margin"]
    # A later funded visit resumes these exact programs and calibration models.
    new_rows = _trajectory(1.75) + _trajectory(1.75)
    work = SearchBudget(16)
    final = extend_decision_evidence(json.loads(json.dumps(first)), new_rows, budget=work)
    assert final["status"] == "adopted" and final["comparison"]["selected"] == "candidate"
    assert final["comparison"]["gain"] == pytest.approx(.325)
    assert final["comparison"]["support_margin"] == pytest.approx(.25)
    assert final["last_compute_cost"] == work.spent == 16
    assert final["comparison"]["compute_cost"] == 32
    assert final["previous_model"] == state["previous_model"]
    assert final["candidate_model"] == state["candidate_model"]
    _, model, together = select_decision_plan(Program.from_dict(state["previous_program"]),
        state["previous_model"], Program.from_dict(state["candidate_program"]), state["candidate_model"],
        first_rows + new_rows, fallback=state["fallback"], costs=state["costs"], budget=SearchBudget(32))
    assert model == state["candidate_model"]
    assert final["comparison"]["gain"] == together["gain"]
    assert final["comparison"]["support_margin"] == together["support_margin"]
    quote = decision_evidence_quote(final, trajectory_length=4, observation_remaining=8, compute_remaining=16)
    assert quote["status"] == "adopted" and not quote["needs_evidence"]
    assert quote["observation_cost"] == quote["compute_cost"] == 0


def test_positive_but_different_sized_gains_can_need_more_evidence():
    state = begin_decision_evidence(Program("const", value=0.), {}, Program("const", value=1.), {},
        fallback=0., costs=(.9, .1))
    state = extend_decision_evidence(state, _trajectory(.11) + _trajectory(1.), budget=SearchBudget(16))
    gains = [trial["gain"] for trial in state["comparison"]["trajectory_comparisons"]]
    assert all(gain > .002 for gain in gains)
    assert state["comparison"]["gain"] < state["comparison"]["support_margin"]
    assert state["status"] == "continue"


def test_consistently_inferior_plan_stops_without_buying_another_batch():
    state = extend_decision_evidence(_calibrated_pair(), _trajectory(.75) + _trajectory(.75),
        budget=SearchBudget(16))
    assert state["status"] == "retained"
    assert all(trial["gain"] < 0 for trial in state["comparison"]["trajectory_comparisons"])
    quote = decision_evidence_quote(state, trajectory_length=4, observation_remaining=100, compute_remaining=1000)
    assert not quote["needs_evidence"] and not quote["payable"]
    assert quote["probe_lengths"] == []


def test_different_programs_with_equivalent_actual_actions_finish_the_question():
    previous_model = {"kind": "global_interval", "action_rule": "empirical_cost_quantile",
        "residual_interval": [1., 1.], "action_offset": .25, "calibration_offset": .75}
    state = begin_decision_evidence(Program("const", value=0.), previous_model,
        Program("const", value=1.), {}, fallback=0., costs=(.9, .1))
    state = extend_decision_evidence(state, _trajectory(.5) + _trajectory(2.), budget=SearchBudget(16))
    assert state["status"] == "retained"
    assert state["comparison"]["reason"] == "no_observed_action_difference"
    assert state["comparison"]["gain"] == 0
    assert state["comparison"]["program_changed"]


@pytest.mark.parametrize("observations,compute", [(15, 32), (16, 31)])
def test_insufficient_budget_preserves_conflict_pending(noisy_decisions, observations, compute):
    state = _from_case(noisy_decisions, comparison=noisy_decisions["comparison"])
    before = deepcopy(state)
    quote = decision_evidence_quote(state, trajectory_length=8,
        observation_remaining=observations, compute_remaining=compute)
    assert quote["status"] == "pending" and quote["needs_evidence"] and not quote["payable"]
    assert quote["observation_cost"] == 16 and quote["compute_cost"] == 32
    assert state == before
    funded = decision_evidence_quote(json.loads(json.dumps(state)), trajectory_length=8,
        observation_remaining=16, compute_remaining=32)
    assert funded["status"] == "continue" and funded["payable"]


def test_unfunded_initial_comparison_is_not_imported_as_paid_evidence(noisy_decisions, monkeypatch):
    rows = [observation_from_dict(row) for row in noisy_decisions["followup_rows"]]
    previous, candidate = (Program.from_dict(noisy_decisions[key])
                           for key in ("previous_program", "candidate_program"))
    _, _, not_compared = select_decision_plan(previous, noisy_decisions["previous_model"],
        candidate, noisy_decisions["candidate_model"], rows, fallback=noisy_decisions["fallback"],
        costs=noisy_decisions["costs"], budget=SearchBudget(0))
    state = _from_case(noisy_decisions, comparison=not_compared)
    assert state["comparison"]["rows"] == state["comparison"]["trajectories"] == 0
    assert state["comparison"]["compute_cost"] == 0
    assert state["comparison"]["reason"] == "insufficient_budget"
    assert state["uncompared_rows"] == 16

    def no_unfunded_prediction(*args, **kwargs):
        raise AssertionError("an unfunded batch must not evaluate frozen policies")

    monkeypatch.setattr(decision_evidence, "action_and_interval", no_unfunded_prediction)
    work = SearchBudget(31)
    pending = extend_decision_evidence(state, rows, budget=work)
    assert work.spent == pending["last_compute_cost"] == 0
    assert pending["comparison"]["trajectory_comparisons"] == []
    assert pending["last_update_reason"] == "insufficient_budget"
    assert pending["status"] == "continue"
