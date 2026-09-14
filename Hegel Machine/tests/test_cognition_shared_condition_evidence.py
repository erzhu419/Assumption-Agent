"""Reuse purchased condition evidence without sharing a model's truth claim."""

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.transfer_condition_research import TransferConditionResearch

from test_cognition_transfer_condition_plans import binary, bounds, call, const, field, purchase


def plan(state, trial, paid=(), *, spec=None, **kwargs):
    return state.plan(spec or bounds(), trial, SearchBudget(200000), paid_rows=paid, **kwargs)


def test_gain_versions_share_repeated_work_but_retain_each_measured_error_and_new_task_question():
    original = call()
    versions = [binary("mul", const(gain), original) for gain in (.5, 1., 2., 3.)]
    truth = versions[0]
    active = {"x": .2312 + 2e-8, "y": -.4117 + 2e-8}
    paid = [Observation(active, 3.5), Observation({"x": 0., "y": 0.}, 0.)] * 16
    state = TransferConditionResearch(versions)
    first = plan(state, truth, paid)
    repeated = plan(state, truth, paid)
    cold = plan(TransferConditionResearch(versions), truth, paid)
    assert first["forecast_status"] == repeated["forecast_status"] == "covered_by_paid_evidence"
    assert repeated["transfer_scope"]["calls"] == cold["transfer_scope"]["calls"]
    assert repeated["compute_cost"] < cold["compute_cost"]
    assert repeated["transfer_scope"]["sharing"]["condition_questions"] == 1
    assert repeated["transfer_scope"]["sharing"]["reused_prediction_evaluations"] > 0
    calls = repeated["transfer_scope"]["calls"]
    assert [item["active_original_call_loss"] for item in calls] == [0., 12.25, 110.25, 306.25]
    assert not calls[0]["original_call_counterexamples"]
    assert all(item["original_call_counterexamples"] for item in calls[1:])

    # New tasks carry executable knowledge; the previous task's active response
    # does not turn its first inactive observations into coverage.
    successor = plan(TransferConditionResearch(versions), truth, [paid[1]], task_id="successor")
    assert successor["forecast_status"] == "predicted_distinction"
    assert not successor["transfer_scope"]["all_conditions_covered"]
    assert any(row.target != 0. for row in purchase(successor, truth))


def test_refitted_trial_and_replaced_response_are_rechecked_on_the_same_controls():
    original = call()
    truth = binary("mul", const(.5), original)
    active = {"x": .2312 + 2e-8, "y": -.4117 + 2e-8}
    state = TransferConditionResearch([original])
    observed = [Observation(active, 3.5)]
    assert plan(state, truth, observed)["forecast_status"] == "covered_by_paid_evidence"
    changed = plan(state, original, observed)
    assert changed["forecast_status"] == "trial_refuted_on_paid_conditions"
    assert changed["transfer_scope"]["trial_counterexamples"][0]["squared_error"] == 12.25
    assert changed["transfer_scope"]["sharing"]["reused_phase_evaluations"] > 0
    # The supported input can be remeasured after a mechanism change. Equality
    # of command coordinates alone cannot reuse its old response verdict.
    remeasured = plan(state, original, [Observation(active, 7.)])
    assert remeasured["forecast_status"] == "covered_by_paid_evidence"
    assert remeasured["transfer_scope"]["calls"][0]["original_call_loss"] == 0.


def test_paid_wrong_binding_does_not_cover_the_unmeasured_correct_binding():
    wrong, correct = call(("y", "x")), call()
    state = TransferConditionResearch([wrong, correct])
    paid = [Observation({"x": 0., "y": 0.}, 0.)]
    first = plan(state, const(0.), paid)
    measured = purchase(first, correct)
    assert all(row.target == 0. for row in measured)
    second = plan(state, const(0.), paid + measured, serial=1)
    assert second["forecast_status"] == "predicted_distinction"
    assert second["transfer_scope"]["sharing"]["condition_questions"] == 2
    assert second["transfer_scope"]["covered_call_ids"] == ["transfer_call_0"]
    assert second["transfer_scope"]["uncovered_call_ids"] == ["transfer_call_1"]
    assert second["transfer_scope"]["calls"][0]["active_original_call_loss"] == 49.
    assert any(row.target == 7. for row in purchase(second, correct))
    assert [item["program"] for item in second["hypotheses"]] == [
        const(0.).to_dict(), wrong.to_dict(), correct.to_dict()]


def test_sensor_recalibration_changes_controls_and_paired_enrichment_rechecks_missing_commands():
    spec = bounds(("knob_a", "knob_b"))
    original = call(("sensor_x", "sensor_y"))
    old_binding = {"sensor_x": field("knob_a"), "sensor_y": field("knob_b")}
    binding = {"sensor_x": binary("mul", const(-1.), field("knob_a")),
               "sensor_y": field("knob_b")}
    truth = bind_measurement_arguments(original.expanded(), binding)
    state = TransferConditionResearch([original])
    first = plan(state, const(0.), spec=spec, binding=old_binding)

    def actual(proposal):
        measured, pairs = [], []
        for row in purchase(proposal, truth):
            sensor = Observation({name: expression.predict(row) for name, expression in binding.items()},
                                 row.target)
            measured.append(sensor)
            pairs.append((row, sensor))
        return measured, pairs

    inactive, inactive_pairs = actual(first)
    assert all(row.target == 0. for row in inactive)
    second = plan(state, const(0.), inactive, spec=spec, binding=binding,
                  paired_rows=inactive_pairs, serial=1)
    assert second["forecast_status"] == "predicted_distinction"
    active, active_pairs = actual(second)
    assert any(row.target == 7. for row in active)
    assert first["probe"].sequence[-1]["knob_a"] > 0.
    assert second["probe"].sequence[-1]["knob_a"] < 0.

    # A command program cannot be validated by sensor-only fields. Supplying
    # the already-paid paired readings must invalidate that missing-data cache.
    measured_only = plan(state, truth, inactive + active, spec=spec, binding=binding)
    assert not measured_only["transfer_scope"]["all_conditions_covered"]
    paired = plan(state, truth, inactive + active, spec=spec, binding=binding,
                  paired_rows=inactive_pairs + active_pairs)
    assert paired["forecast_status"] == "covered_by_paid_evidence"
    assert paired["transfer_scope"]["calls"][0]["active_trial_loss"] == 0.


def test_shared_temporal_predicates_keep_distinct_depths_and_paid_exit_counterexamples():
    original = call()
    immediate = Program("event_delta", lag=1, children=(original,))
    delayed = Program("event_delta", lag=2, children=(original,))
    truth = binary("mul", const(.5), delayed)
    trial = binary("add", binary("mul", const(.25), delayed),
                   binary("mul", const(1. / 28.), binary("mul", delayed, delayed)))
    inactive, active = {"x": 0., "y": 0.}, {"x": .2312 + 2e-8, "y": -.4117 + 2e-8}
    complete = purchase({"probe": Probe("event_cycle", (
        inactive, inactive, active, active, active, inactive, inactive))}, truth)
    prefix = complete[:5]
    assert all(trial.predict(row) == row.target for row in prefix)
    state = TransferConditionResearch([immediate, delayed])
    first = plan(state, trial, prefix)
    assert first["transfer_scope"]["sharing"]["condition_questions"] == 2
    assert all(not item["conditions"][0]["phase_witnesses"]["exit"]
               for item in first["transfer_scope"]["calls"])
    contradicted = plan(state, trial, complete)
    assert contradicted["forecast_status"] == "trial_refuted_on_paid_conditions"
    assert any(item["target"] < 0. for item in contradicted["transfer_scope"]["trial_counterexamples"])
    revised = plan(state, truth, complete)
    assert revised["forecast_status"] == "covered_by_paid_evidence"
    assert all(all(item["conditions"][0]["phase_witnesses"].values())
               for item in revised["transfer_scope"]["calls"])
    first_errors, second_errors = [item["original_call_counterexamples"]
                                  for item in revised["transfer_scope"]["calls"]]
    assert next(item["prediction"] for item in first_errors if item["position"] == 3) == 0.
    assert next(item["prediction"] for item in second_errors if item["position"] == 3) == 7.
