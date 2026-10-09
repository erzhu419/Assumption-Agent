"""A useful response refit does not make its later boundary trials useful."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, condition_followup_policy,
    record_condition_followup, record_condition_information,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.condition_research_returns import record_condition_research
from hegel_machine.cognition.object_continuation import observations
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from test_cognition_condition_research_returns import decide, plan, state
from test_cognition_method_revisit_opportunity import UnifiedResearchSession, persisted, real_method_stream
from test_cognition_observed_condition_uncertainty import (
    READINGS, entry, observations as observed_responses, response, spec as condition_spec,
)


@pytest.fixture
def root03():
    path = Path(__file__).parent / "fixtures/condition_marginal_root03.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def measured_context(payload):
    before, after = payload["cases"]
    rows = observations(after["observations"])
    channels = after["comparison"]["channels"]
    specification = TaskSpec("measured_followup", "", tuple(rows[0].features), (), (),
        feature_bounds={name: (-1.5, 1.5) for name in rows[0].features if name not in channels})
    selected = {"candidate": Candidate("confirmed_response",
        Program.from_dict(before["comparison"]["program"]), "retained_incumbent"),
        "condition_uncertainty": deepcopy(payload["condition_uncertainty"])}
    return specification, rows, selected


def test_real_method_gain_is_retained_without_crediting_it_to_zero_gain_followup(root03):
    spec, rows, selected = measured_context(root03)
    old, current = (case["comparison"] for case in root03["cases"])
    measured_before = loss(selected["candidate"].program, rows, old["fallback"], .5)
    measured_after = loss(Program.from_dict(current["program"]), rows, current["fallback"], .5)
    assert current["before_loss"] > 100.
    assert measured_before == pytest.approx(measured_after, abs=1e-12)
    whole_context = root03["whole_method_outcome"]["method_context"]
    history = {whole_context: deepcopy(root03["method_experience"])}
    original = deepcopy(history)
    policy = condition_followup_policy(spec, rows, selected, history)
    record_condition_followup(policy, before_loss=measured_before, after_loss=measured_after,
        observation_cost=72, measurement_cost=72, compute_cost=5000)
    marginal = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    assert marginal["trials"] == 1
    assert marginal["last_gain"] == pytest.approx(0., abs=1e-7)
    assert marginal["expected_gain"] < 1e-7
    assert history[whole_context] == original[whole_context]
    assert next(iter(history[whole_context].values()))["expected_gain"] > .99


def test_information_feedback_is_not_an_extra_prediction_trial_or_an_extra_bill(root03):
    spec, rows, selected = measured_context(root03)
    history = {}
    policy = condition_followup_policy(spec, rows, selected, history)
    record_condition_followup(policy, before_loss=1., after_loss=1.,
        observation_cost=72, measurement_cost=72, compute_cost=5000)
    before = deepcopy(policy.action_record(CONDITION_FOLLOWUP_ACTION))
    outcomes = [outcome for allocation in root03["allocations"] for outcome in allocation["outcomes"]]
    assert len(outcomes) == 7 and not any(outcome["informative"] for outcome in outcomes)
    for outcome in outcomes:
        record_condition_information(policy, outcome)
    after = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    for key in ("trials", "expected_gain", "last_gain", "mean_observation_cost",
                "mean_measurement_cost", "mean_compute_cost"):
        assert after[key] == before[key]
    saved = json.loads(json.dumps(history))
    restored = condition_followup_policy(spec, rows, selected, saved)
    frozen = deepcopy(saved)
    restored.action_record(CONDITION_FOLLOWUP_ACTION)
    assert saved == frozen
    # An unfinished tail phase incurred costs, but did not buy a common fresh
    # response on which another prediction gain could be measured.
    record_condition_followup(restored, observation_cost=24, measurement_cost=24, compute_cost=200)
    unconfirmed = restored.action_record(CONDITION_FOLLOWUP_ACTION)
    assert unconfirmed["trials"] == after["trials"]
    assert unconfirmed["expected_gain"] == after["expected_gain"]
    assert unconfirmed["unscored_attempts"] == 1
    assert unconfirmed["unscored_observation_cost"] == 24
    assert unconfirmed["unscored_measurement_cost"] == 24
    assert unconfirmed["unscored_compute_cost"] == 200


def test_unproductive_followup_can_stop_without_blocking_new_paid_evidence(root03):
    spec, rows, selected = measured_context(root03)
    policy = condition_followup_policy(spec, rows, selected, {})
    record_condition_followup(policy, before_loss=1., after_loss=1.,
        observation_cost=72, measurement_cost=72, compute_cost=5000)
    for allocation in root03["allocations"]:
        for outcome in allocation["outcomes"]:
            record_condition_information(policy, outcome)
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    question = state()
    experiment = plan(.01)
    assert not decide(question, experiment, marginal_record=record,
        observation_remaining=96, measurement_remaining=96)["buy"]
    explicit = decide(question, experiment, marginal_record=record,
        observation_remaining=96, measurement_remaining=96, new_counterexample=True)
    assert explicit["buy"] and explicit["reopened"]
    # A measured narrower interval can be repriced even when the historical
    # prediction return was zero, but cannot bypass the remaining resource bill.
    previous = {"last_question": deepcopy(question), "batches": 7}
    narrowed = state(interval=(-.1, .1))
    opened = decide(narrowed, plan(.01, interval=(-.1, .1)), previous,
        marginal_record=record, observation_remaining=96, measurement_remaining=96)
    assert not opened["buy"] and not opened["reopened"]
    funded = decide(narrowed, plan(.01, interval=(-.1, .1)), previous,
        marginal_record=record, observation_remaining=2880, measurement_remaining=2880,
        compute_remaining=2400000)
    assert funded["buy"] and not funded["reopened"]


def test_transfer_of_followup_experience_respects_the_executable_question(root03):
    spec, rows, selected = measured_context(root03)
    history = {}
    first = condition_followup_policy(spec, rows, selected, history)
    record_condition_followup(first, before_loss=1., after_loss=1.,
        observation_cost=72, measurement_cost=72, compute_cost=5000)
    context = first.action_context(CONDITION_FOLLOWUP_ACTION)
    names = {name: f"command_{index}" for index, name in enumerate(spec.feature_bounds)}
    names.update({name: "reading" for name in spec.feature_names if name not in spec.feature_bounds})

    def rename(program):
        return replace(program, name=names.get(program.name, program.name),
                       children=tuple(rename(child) for child in program.children))

    renamed = deepcopy(selected)
    renamed["candidate"] = replace(selected["candidate"], program=rename(selected["candidate"].program))
    for condition in renamed["condition_uncertainty"]["conditions"]:
        condition["condition_field"] = names[condition["condition_field"]]
        condition["control"] = names.get(condition.get("control"), condition.get("control"))
    renamed_rows = [Observation({names[name]: value for name, value in row.features.items()}, row.target,
        tuple({names.get(name, name): value for name, value in past.items()} for past in row.history)) for row in rows]
    renamed_spec = replace(spec, task_id="different_apparatus",
        feature_names=tuple(names[name] for name in spec.feature_names),
        feature_bounds={names[name]: bounds for name, bounds in spec.feature_bounds.items()})
    equivalent = condition_followup_policy(renamed_spec, renamed_rows, renamed, history)
    assert equivalent.action_context(CONDITION_FOLLOWUP_ACTION) == context
    assert equivalent.action_record(CONDITION_FOLLOWUP_ACTION)["trials"] == 1

    def wider(program):
        return replace(program, lag=program.lag + 2 if program.op == "window_sum" else program.lag,
                       children=tuple(wider(child) for child in program.children))

    changed = deepcopy(selected)
    changed["candidate"] = replace(selected["candidate"], program=wider(selected["candidate"].program))
    different = condition_followup_policy(spec, rows, changed, history)
    assert different.action_context(CONDITION_FOLLOWUP_ACTION) != context
    assert different.action_record(CONDITION_FOLLOWUP_ACTION).get("trials", 0) == 0
    risk = condition_followup_policy(replace(spec, risk_quantile=.9), rows, selected, history)
    assert risk.action_context(CONDITION_FOLLOWUP_ACTION) != context
    assert risk.action_record(CONDITION_FOLLOWUP_ACTION).get("trials", 0) == 0


def test_actual_method_feedback_uses_adjacent_frozen_responses_without_duplicate_costs(real_method_stream):
    stream = real_method_stream
    previous = stream["queued"]["results"][0]["research"]
    report = stream["final"]["results"][0]["research"]
    events = report["continuation"]["method_marginal_comparisons"]
    paired = [event for event in events if event["action"] == "compare_method_followup_on_fresh_response"]
    assert paired
    for event in paired:
        batch = next(batch for batch in report["batches"] if batch["probe"]["probe_id"] == event["probe_id"])
        rows = observations(batch["observations"])
        assert event["before_loss"] == pytest.approx(loss(
            Program.from_dict(event["baseline_program"]), rows, event["baseline_fallback"], .5))
        assert event["after_loss"] == pytest.approx(loss(
            Program.from_dict(event["program"]), rows, event["fallback"], .5))
        assert event["observation_rows"] == len(rows)
        assert event["probe_id"] not in {batch["probe"]["probe_id"] for batch in previous["batches"]}
    assert paired[0]["baseline_program"] == paired[0]["program"]
    assert paired[0]["before_loss"] == paired[0]["after_loss"]
    whole = report["continuation"]["method_visit_outcome"]
    assert whole["fresh_comparison"]["after_loss"] < whole["fresh_comparison"]["before_loss"]
    for event in events:
        assert event["costs_included_in_whole_visit"]
        assert all(value >= 0 for value in event["stage_costs"].values())
        if event["action"] == "retain_unconfirmed_method_followup":
            assert event.get("before_loss") is event.get("after_loss") is None
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert sum(event["stage_costs"][resource] for event in events) <= report["visit_costs"][resource]
        actual = sum(item["research"]["costs"][resource] for item in stream["final"]["results"])
        if resource == "compute_cost":
            actual += stream["final"]["screening_compute_cost"]
        assert stream["final"]["costs"][resource] == actual
    restored = UnifiedResearchSession(state=persisted(stream["final"]), lab_factory=stream["factory"])
    before = len(stream["purchases"])
    assert persisted(restored.run()) == stream["final"]
    assert len(stream["purchases"]) == before


def test_refitting_the_new_counterexample_preserves_its_information_without_prediction_credit():
    old = entry(response())
    paid = observed_responses(response(.1), READINGS)
    before = assess_observed_condition_uncertainty(condition_spec(), paid, old, SearchBudget(20000))
    acquired = observed_responses(response(.1), [.05, .15])
    measured = [*paid, *acquired]
    observed = assess_observed_condition_uncertainty(condition_spec(), measured, old,
        SearchBudget(20000), previous=before)
    fitted = entry(response(.1))
    after = assess_observed_condition_uncertainty(condition_spec(), measured, fitted,
        SearchBudget(20000), previous=observed)
    assert before["original_target_loss"] == after["original_target_loss"] == 0.
    assert observed["original_target_loss"] > 1e-7
    history = record_condition_research(None, before, after, plan(),
        observation_cost=len(acquired), measurement_cost=len(acquired), compute_cost=200,
        changed_program=True, observed_after=observed)
    outcome = history["last_outcome"]
    assert outcome["observed_response_counterexample"]
    assert outcome["marginal_information_gain"] > 0.
    assert history["last_question"]["program"] == fitted["candidate"].program.to_dict()
    policy = condition_followup_policy(condition_spec(), paid, old, {})
    record_condition_information(policy, outcome)
    # Refitting these same paid rows is not an independent prediction result.
    record_condition_followup(policy, observation_cost=len(acquired),
        measurement_cost=len(acquired), compute_cost=200)
    credited = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    assert credited.get("trials", 0) == 0
    assert credited.get("expected_gain", 0.) == 0.
    assert credited["unscored_attempts"] == 1
    assert credited["information_gain_sum"] > 0.
