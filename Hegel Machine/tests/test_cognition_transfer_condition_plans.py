"""A constant short fit must not erase an untested learned-call condition."""

from itertools import permutations

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
from hegel_machine.cognition.transfer_condition_research import plan_transfer_condition_probe


def field(name):
    return Program("feature", name=name)


def const(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def bounds(names=("x", "y")):
    return TaskSpec("transfer", "public controls", names, (), (),
                    feature_bounds={name: (-1., 1.) for name in names})


def call(names=("x", "y"), origin=(.2312, -.4117), identifier="window"):
    result = const(7.)
    for role, low in zip(("first", "second"), origin):
        argument = Program("arg", name=role)
        condition = binary("and", binary("ge", argument, const(low)),
                           binary("lt", argument, const(low + 4e-8)))
        result = Program("if", children=(condition, result, const(0.)))
    definition = Definition(identifier, ("first", "second"), result)
    return Program("call", name=identifier, definition=definition, children=tuple(field(name) for name in names))


def purchase(plan, truth):
    history, rows = [], []
    for point in plan["probe"].sequence:
        raw = Observation(point, 0., tuple(history))
        target = truth.predict(raw)
        rows.append(Observation(point, target, tuple(history)))
        history.append({**point, "__target__": target})
    return rows


def test_inactive_paid_data_preserves_every_rebound_call_until_real_probes_distinguish_them():
    # The wrong variable permutation is first, so one proposed call is insufficient.
    wrong, correct = call(("y", "x")), call()
    paid = [Observation({"x": 0., "y": 0.}, 0.)]
    first = plan_transfer_condition_probe(bounds(), const(0.), [wrong, correct],
        SearchBudget(64000), paid_rows=paid, serial=11)
    assert first["forecast_status"] == "predicted_distinction"
    assert len(first["hypotheses"]) == 3
    assert first["transfer_scope"]["uncovered_call_ids"] == ["transfer_call_0", "transfer_call_1"]
    assert all(item["active_paid_rows"] == 0 for item in first["transfer_scope"]["calls"])
    measured = purchase(first, correct)
    assert all(row.target == 0. for row in measured)
    assessment = assess_competing_object_probe(first, [(row, row) for row in measured], SearchBudget(1000))
    assert assessment["status"] == "distinguished"
    paid += measured
    second = plan_transfer_condition_probe(bounds(), const(0.), [wrong, correct],
        SearchBudget(64000), paid_rows=paid, serial=12)
    assert second["forecast_status"] == "predicted_distinction"
    assert second["transfer_scope"]["covered_call_ids"] == ["transfer_call_0"]
    assert second["transfer_scope"]["uncovered_call_ids"] == ["transfer_call_1"]
    assert second["transfer_scope"]["calls"][0]["active_original_call_loss"] == 49.
    assert second["transfer_scope"]["calls"][0]["original_call_counterexamples"]
    assert any(row.target == 7. for row in purchase(second, correct))
    # The frozen hypotheses retain both full original definitions after proposal selection.
    assert [item["program"] for item in second["hypotheses"]] == [
        const(0.).to_dict(), wrong.to_dict(), correct.to_dict()]


def test_paid_activation_covers_a_refitted_gain_without_hiding_original_call_error():
    original = call()
    trial = binary("mul", const(.5), original)
    row = Observation({"x": .2312 + 2e-8, "y": -.4117 + 2e-8}, 3.5)
    result = plan_transfer_condition_probe(bounds(), trial, [original], SearchBudget(64000), paid_rows=[row])
    assert result["forecast_status"] == "covered_by_paid_evidence" and result["probe"] is None
    assert result["transfer_scope"]["all_conditions_covered"]
    evidence = result["transfer_scope"]["calls"][0]
    assert evidence["active_paid_rows"] == 1
    assert evidence["active_trial_loss"] == 0. and evidence["active_original_call_loss"] == 12.25
    assert evidence["original_call_counterexamples"][0]["target"] == 3.5


def test_erased_old_condition_reports_its_measured_failure_even_if_constant_trial_is_right():
    original = call()
    row = Observation({"x": .2312 + 2e-8, "y": -.4117 + 2e-8}, 0.)
    result = plan_transfer_condition_probe(bounds(), const(0.), [original], SearchBudget(64000), paid_rows=[row])
    assert result["probe"] is None
    assert result["transfer_scope"]["calls"][0]["original_call_counterexamples"]
    assert result["transfer_scope"]["calls"][0]["active_original_call_loss"] == 49.
    # The caller can now reopen structure instead of re-purchasing this known old region.
    assert result["transfer_scope"]["calls"][0]["active_trial_loss"] == 0.


def test_observed_trial_failure_requests_revision_without_rebuying_the_counterexample():
    original = call()
    row = Observation({"x": .2312 + 2e-8, "y": -.4117 + 2e-8}, 3.)
    result = plan_transfer_condition_probe(bounds(), const(0.), [original], SearchBudget(64000), paid_rows=[row])
    assert result["forecast_status"] == "trial_refuted_on_paid_conditions"
    assert result["probe"] is None and result["transfer_scope"]["trial_counterexamples"]


def test_missing_calibration_requests_actual_paired_readings_and_known_binding_can_activate():
    commands = bounds(("knob_a", "knob_b"))
    original = call(("sensor_x", "sensor_y"))
    unknown = plan_transfer_condition_probe(commands, const(0.), [original], SearchBudget(64000))
    assert unknown["forecast_status"] == "requires_paired_measurements"
    assert unknown["probe"] is not None and unknown["predicted_separation"] is None
    assert set(unknown["missing_fields"]) == {"sensor_x", "sensor_y"}
    binding = {"sensor_x": binary("mul", const(-1.), field("knob_a")), "sensor_y": field("knob_b")}
    known = plan_transfer_condition_probe(commands, const(0.), [original], SearchBudget(64000), binding=binding)
    assert known["forecast_status"] == "predicted_distinction"
    final = known["probe"].sequence[-1]
    command = Observation(final, 7.)
    measured = Observation({name: program.predict(command) for name, program in binding.items()}, 7.)
    command_trial = binary("mul", const(7.), binary("and",
        binary("ge", field("knob_a"), const(-.232)), binary("lt", field("knob_a"), const(-.231))))
    covered = plan_transfer_condition_probe(commands, command_trial, [original], SearchBudget(64000),
        paid_rows=[measured], paired_rows=[(command, measured)])
    assert covered["forecast_status"] == "covered_by_paid_evidence"
    assert covered["transfer_scope"]["evidence_rows"] == 1


def test_one_covered_branch_does_not_hide_another_unmeasured_branch():
    first, second = call(identifier="old_first"), call(origin=(-.7112, .6173), identifier="old_second")
    original = binary("add", first, second)
    paid = [Observation({"x": .2312 + 2e-8, "y": -.4117 + 2e-8}, 0.)]
    plan = plan_transfer_condition_probe(bounds(), const(0.), [original], SearchBudget(64000), paid_rows=paid)
    assert plan["forecast_status"] == "predicted_distinction"
    assert any(item["covered"] for item in plan["transfer_scope"]["calls"][0]["conditions"])
    measured = purchase(plan, second)
    assert any(row.target == 7. for row in measured)
    assert all(first.predict(row) == 0. for row in measured)
    assert plan["hypotheses"][1]["program"] == original.to_dict()


def test_small_budget_and_unreachable_conditions_remain_unknown_while_numeric_calls_are_cheap():
    empty = SearchBudget(0)
    numeric = plan_transfer_condition_probe(bounds(), const(0.), [field("x")], empty)
    assert numeric["forecast_status"] == "no_condition" and empty.spent == 0
    exhausted = plan_transfer_condition_probe(bounds(), const(0.), [call()], SearchBudget(0))
    assert exhausted["forecast_status"] == "compute_exhausted"
    inaccessible = plan_transfer_condition_probe(bounds(), const(0.), [call(origin=(2., 3.))],
                                                  SearchBudget(64000))
    assert inaccessible["probe"] is None and inaccessible["forecast_status"] == "no_predicted_distinction"
    assert not inaccessible["transfer_scope"]["all_conditions_covered"]


def test_all_frozen_programs_can_be_refuted_by_purchased_measurements():
    plan = plan_transfer_condition_probe(bounds(), const(0.), [call()], SearchBudget(64000))
    measured = purchase(plan, const(3.))
    result = assess_competing_object_probe(plan, [(row, row) for row in measured], SearchBudget(1000))
    assert result["status"] == "all_refuted"
    assert not plan["transfer_scope"]["all_conditions_covered"]


def test_sixteen_distinct_current_bindings_remain_distinct_frozen_questions():
    names = ("a", "b", "c", "d", "e")
    retained = [call(pair) for pair in list(permutations(names, 2))[:16]]
    plan = plan_transfer_condition_probe(bounds(names), const(0.), retained, SearchBudget(64000))
    assert plan["forecast_status"] == "predicted_distinction"
    assert len(plan["hypotheses"]) == 17
    assert len(plan["transfer_scope"]["uncovered_call_ids"]) == 16
    assert [item["program"] for item in plan["hypotheses"][1:]] == [program.to_dict() for program in retained]
    assert all(item["predictions"] is not None for item in plan["hypotheses"])


def test_unchanged_call_is_activated_against_a_constant_alternative_supported_by_paid_rows():
    original = call()
    paid = [Observation({"x": 0., "y": 0.}, 0.), Observation({"x": .5, "y": .5}, 0.)]
    plan = plan_transfer_condition_probe(bounds(), original, [original], SearchBudget(64000), paid_rows=paid)
    assert plan["forecast_status"] == "predicted_distinction"
    assert plan["transfer_scope"]["observational_baseline"] == {
        "hypothesis_id": "transfer_observed_constant", "value": 0., "paid_rows": 2, "measured_loss": 0.}
    assert [item["program"] for item in plan["hypotheses"]] == [
        original.to_dict(), original.to_dict(), const(0.).to_dict()]
    assert all(item["predictions"] is not None for item in plan["hypotheses"])
    measured = purchase(plan, original)
    assert any(row.target == 7. for row in measured)
    actual = assess_competing_object_probe(plan, [(row, row) for row in measured], SearchBudget(1000))
    assert actual["status"] == "distinguished"
    assert actual["hypotheses"][-1]["status"] == "counterexample"
    covered = plan_transfer_condition_probe(bounds(), original, [original], SearchBudget(64000),
        paid_rows=paid + measured)
    assert covered["forecast_status"] == "covered_by_paid_evidence"
    assert "observational_baseline" not in covered["transfer_scope"]


def test_no_paid_constant_alternative_is_invented_without_constant_observations():
    original = call()
    for paid in ([], [Observation({"x": 0., "y": 0.}, 0.),
                     Observation({"x": .5, "y": .5}, 1.)]):
        plan = plan_transfer_condition_probe(bounds(), original, [original], SearchBudget(64000), paid_rows=paid)
        assert "observational_baseline" not in plan["transfer_scope"]
        assert len(plan["hypotheses"]) == 2


def test_a_covered_correct_call_cannot_keep_drawing_probes_away_from_an_unknown_binding():
    correct, wrong = call(), call(("y", "x"))
    paid = [Observation({"x": 0., "y": 0.}, 0.)]
    methods = ProgramExperimentMethods()
    first = plan_transfer_condition_probe(bounds(), const(0.), [correct, wrong], SearchBudget(64000),
                                           paid_rows=paid, methods=methods, task_id="source")
    measured = purchase(first, correct)
    assert any(row.target == 7. for row in measured)
    assessment = assess_competing_object_probe(first, [(row, row) for row in measured], SearchBudget(1000))
    assert methods.observe(first, assessment, task_id="source", command_spec=bounds()) is not None
    next_plan = plan_transfer_condition_probe(bounds(), correct, [correct, wrong], SearchBudget(64000),
        paid_rows=paid + measured, methods=methods, task_id="source", serial=1)
    assert next_plan["forecast_status"] == "predicted_distinction"
    assert next_plan["transfer_scope"]["covered_call_ids"] == ["transfer_call_0"]
    assert next_plan["transfer_scope"]["uncovered_call_ids"] == ["transfer_call_1"]
    next_rows = purchase(next_plan, correct)
    assert all(row.target == 0. for row in next_rows)
    assert any(wrong.predict(row) == 7. for row in next_rows)
    assert [item["program"] for item in next_plan["hypotheses"]] == [
        correct.to_dict(), correct.to_dict(), wrong.to_dict()]


def test_available_history_guard_alone_adds_no_condition_research_or_purchases():
    value = Program("arg", name="value")
    body = Program("if", children=(Program("available", lag=1),
        binary("add", value, Program("lag", name="x", lag=1)), value))
    definition = Definition("history_numeric", ("value",), body)
    original = Program("call", name=definition.definition_id, definition=definition, children=(field("x"),))
    budget = SearchBudget(0)
    plan = plan_transfer_condition_probe(bounds(), original, [original], budget,
                                         paid_rows=[Observation({"x": 0., "y": 0.}, 0.)])
    assert plan["forecast_status"] == "no_condition" and plan["probe"] is None
    assert not plan["transfer_scope"]["has_conditional_calls"] and budget.spent == 0
    guarded = Program("if", children=(Program("available", lag=1), call(), const(0.)))
    actual = plan_transfer_condition_probe(bounds(), const(0.), [guarded], SearchBudget(64000))
    assert actual["transfer_scope"]["has_conditional_calls"]
    assert actual["forecast_status"] == "predicted_distinction"


def test_event_exit_is_purchased_before_history_coverage_and_the_cycle_method_rebinds():
    original = Program("event_delta", lag=2, children=(call(),))
    truth = binary("mul", const(.5), original)
    # This fit is exact on positive entries, but predicts zero on negative exits.
    trial = binary("add", binary("mul", const(.25), original),
        binary("mul", const(1. / 28.), binary("mul", original, original)))
    inactive, active = {"x": 0., "y": 0.}, {"x": .2312 + 2e-8, "y": -.4117 + 2e-8}
    prefix = {"probe": Probe("positive_entry", (inactive, active, active, active))}
    paid = purchase(prefix, truth)
    assert all(trial.predict(row) == row.target for row in paid)
    methods = ProgramExperimentMethods()
    plan = plan_transfer_condition_probe(bounds(), trial, [original], SearchBudget(64000),
        paid_rows=paid, methods=methods, task_id="event_source")
    condition = plan["transfer_scope"]["calls"][0]["conditions"][0]
    assert condition["phase_witnesses"]["enter"] and condition["phase_witnesses"]["maintain"]
    assert not condition["phase_witnesses"]["exit"] and not condition["covered"]
    assert plan["forecast_status"] == "predicted_distinction"
    assert plan["proposal"]["sequence_mode"] == "cycle"
    measured = purchase(plan, truth)
    assert any(row.target < 0. for row in measured)
    contradicted = plan_transfer_condition_probe(bounds(), trial, [original], SearchBudget(64000),
        paid_rows=paid + measured)
    assert contradicted["forecast_status"] == "trial_refuted_on_paid_conditions"
    refitted = plan_transfer_condition_probe(bounds(), truth, [original], SearchBudget(64000),
        paid_rows=paid + measured)
    assert refitted["forecast_status"] == "covered_by_paid_evidence"
    assert all(refitted["transfer_scope"]["calls"][0]["conditions"][0]["phase_witnesses"].values())
    assessment = assess_competing_object_probe(plan, [(row, row) for row in measured], SearchBudget(1000))
    assert assessment["status"] == "all_refuted"
    assert methods.observe(plan, assessment, task_id="event_source", command_spec=bounds()) is not None
    renamed = Program("event_delta", lag=2, children=(call(("left", "right")),))
    replay = plan_transfer_condition_probe(bounds(("left", "right")), const(0.), [renamed],
        SearchBudget(64000), methods=methods, task_id="event_successor")
    assert replay["proposal_source"] == "learned_method"
    assert replay["proposal"]["sequence_mode"] == "cycle"
    response = purchase(replay, renamed)
    assert any(row.target > 0. for row in response) and any(row.target < 0. for row in response)


def test_prior_output_state_predicates_are_not_mistaken_for_empty_history_guards():
    condition = binary("ge", Program("lag", name="__target__", lag=1), const(.2))
    original = Program("if", children=(condition, const(1.), const(0.)))
    plan = plan_transfer_condition_probe(bounds(), const(0.), [original], SearchBudget(64000))
    assert plan["transfer_scope"]["has_conditional_calls"]
    assert plan["forecast_status"] != "no_condition"


def test_startup_zero_and_exit_only_data_do_not_cover_a_real_event_entry():
    original = Program("event_delta", lag=3, children=(call(),))
    truth = binary("mul", const(.5), original)
    trial = binary("sub", binary("mul", const(.25), original),
        binary("mul", const(1. / 28.), binary("mul", original, original)))
    inactive, active = {"x": 0., "y": 0.}, {"x": .2312 + 2e-8, "y": -.4117 + 2e-8}
    startup = purchase({"probe": Probe("insufficient_entry", (inactive, active))}, truth)
    exit_only = purchase({"probe": Probe("reset_active", (active, active, active, active,
                                                           inactive, inactive, inactive, inactive))}, truth)
    assert all(trial.predict(row) == row.target for row in startup + exit_only)
    plan = plan_transfer_condition_probe(bounds(), trial, [original], SearchBudget(64000),
                                          paid_rows=startup + exit_only)
    condition = plan["transfer_scope"]["calls"][0]["conditions"][0]
    assert not condition["phase_witnesses"]["enter"]
    assert condition["phase_witnesses"]["maintain"] and condition["phase_witnesses"]["exit"]
    assert plan["forecast_status"] == "predicted_distinction"
    assert any(row.target > 0. for row in purchase(plan, truth))
