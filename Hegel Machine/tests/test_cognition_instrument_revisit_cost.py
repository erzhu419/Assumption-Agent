"""Learn whole paid execution cost without duplicating marginal return trials."""

from copy import deepcopy
import json

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, instrument_cycle_cost_estimate,
    record_condition_followup, record_instrument_revisit_cost,
)
from test_cognition_condition_realized_investment import marginal_policy


def test_paid_root07_tail_is_in_whole_cost_but_screen_is_not_future_work():
    policy, plan = marginal_policy()
    # Recorded root07 p5: first fresh stage includes the sunk 14221 screen;
    # another 13592 is then spent without buying more observations.
    record_condition_followup(policy, before_loss=0., after_loss=0.,
        observation_cost=48, measurement_cost=48, compute_cost=37608)
    record_condition_followup(policy,
        observation_cost=0, measurement_cost=0, compute_cost=13592)
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    before = deepcopy(record)
    cycle = record_instrument_revisit_cost(policy,
        observation_cost=48, measurement_cost=48, compute_cost=51200,
        screening_compute_cost=14221, sampling_context=plan["sampling_context"])
    assert cycle["compute_cost"] == 36979
    assert cycle["compute_cost"] == 37608 - 14221 + 13592
    assert cycle["observation_cost"] == cycle["measurement_cost"] == 48
    assert record == {**before, "instrument_revisits": [cycle]}
    assert record["trials"] == record["unscored_attempts"] == 1
    assert record["expected_gain"] == 0.


def test_adjacent_phases_never_emit_partial_complete_visit_cost_samples():
    policy, _ = marginal_policy()
    record_condition_followup(policy, before_loss=10., after_loss=1.,
        observation_cost=48, measurement_cost=48, compute_cost=4000)
    record_condition_followup(policy, before_loss=1., after_loss=1.,
        observation_cost=48, measurement_cost=48, compute_cost=3000)
    record_condition_followup(policy,
        observation_cost=0, measurement_cost=0, compute_cost=2000)
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    assert record["trials"] == 2 and record["unscored_attempts"] == 1
    assert "instrument_revisits" not in record


def test_whole_visit_cost_survives_json_and_preserves_hard_reserve_and_phase_gains():
    policy, plan = marginal_policy()
    record_condition_followup(policy, before_loss=10., after_loss=1.,
        observation_cost=48, measurement_cost=48, compute_cost=50000)
    before = deepcopy(policy.action_record(CONDITION_FOLLOWUP_ACTION))
    record_instrument_revisit_cost(policy,
        observation_cost=72, measurement_cost=72, compute_cost=80000,
        screening_compute_cost=20000, sampling_context=plan["sampling_context"])
    record = json.loads(json.dumps(policy.action_record(CONDITION_FOLLOWUP_ACTION)))
    reserve = {"observation_cost": 48, "measurement_cost": 48, "compute_cost": 110000}
    assert instrument_cycle_cost_estimate(record, plan, reserve) == {
        "observation_cost": 60, "measurement_cost": 60, "compute_cost": 85000}
    assert reserve == {"observation_cost": 48, "measurement_cost": 48, "compute_cost": 110000}
    assert {key: value for key, value in record.items() if key != "instrument_revisits"} == before


def test_restored_phase_costs_do_not_discount_complete_revisit_quotes():
    policy, plan = marginal_policy()
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    record["instrument_cycles"] = [{"sampling_context": deepcopy(plan["sampling_context"]),
        "observation_cost": 48, "measurement_cost": 48, "compute_cost": 1000,
        "screening_compute_cost": 1000}]
    restored = json.loads(json.dumps(record))
    reserve = {"observation_cost": 48, "measurement_cost": 48, "compute_cost": 110000}
    assert instrument_cycle_cost_estimate(restored, plan, reserve) == reserve
    assert restored == record
    record_instrument_revisit_cost(policy, observation_cost=48, measurement_cost=48,
        compute_cost=40000, screening_compute_cost=1000, sampling_context=plan["sampling_context"])
    assert instrument_cycle_cost_estimate(record, plan, reserve)["compute_cost"] == 74500
    assert record["instrument_cycles"] == restored["instrument_cycles"]
