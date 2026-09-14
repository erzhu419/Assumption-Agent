"""Shared condition planning must retain binding, history and evidence meaning."""

from copy import deepcopy

from hegel_machine.cognition.competing_object_experiments import _forecast, assess_competing_object_probe
from hegel_machine.cognition.condition_questions import plan_condition_questions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods


def field(name):
    return Program("feature", name=name)


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def window(names=("x", "y"), origin=(.2312, -.4117)):
    conditions = [binary("and", binary("ge", field(name), constant(low)),
                         binary("lt", field(name), constant(low + 4e-8)))
                  for name, low in zip(names, origin)]
    return binary("and", *conditions)


def spec(names=("x", "y")):
    return TaskSpec("questions", "public controls", names, (), (),
                    feature_bounds={name: (-1., 1.) for name in names})


def question(predicate=None, identifier="condition_0", dependencies=("call_0",), depth=0):
    return {"question_id": identifier, "predicate": predicate or window(),
            "history_depth": depth, "uncovered_phases": ["enter", "maintain", "exit"] if depth else ["active"],
            "dependent_hypothesis_ids": list(dependencies)}


def freeze_laws(plan, programs):
    # This is what the runtime must do BEFORE buying/observing the question.
    plan["hypotheses"] = [{"hypothesis_id": f"law_{index}", "candidate_id": f"law_{index}",
        "program": program.to_dict(), "predictions": _forecast(program, plan["probe"].sequence)}
        for index, program in enumerate(programs)]
    return plan


def purchase(plan, truth):
    rows, history = [], []
    for point in plan["probe"].sequence:
        row = Observation(point, 0., tuple(history))
        target = truth.predict(row)
        rows.append(Observation(point, target, tuple(history)))
        history.append({**point, "__target__": target})
    return rows


def test_sixteen_response_versions_share_one_condition_solve_and_forecast():
    single = plan_condition_questions(spec(), [question()], SearchBudget(64000))
    duplicate = plan_condition_questions(spec(), [question(identifier=f"condition_{index}",
        dependencies=(f"gain_{index}",)) for index in range(16)], SearchBudget(64000))
    assert single["forecast_status"] == duplicate["forecast_status"] == "predicted_distinction"
    assert duplicate["question_count"] == duplicate["questions_attempted"] == 1
    assert duplicate["compute_cost"] == single["compute_cost"]
    assert duplicate["generation_cost"] == single["generation_cost"] <= 6000
    assert duplicate["probe"].sequence == single["probe"].sequence
    assert len(duplicate["question"]["dependent_hypothesis_ids"]) == 16
    assert any(window().predict(Observation(point, 0.)) for point in duplicate["probe"].sequence)


def test_different_field_bindings_remain_distinct_condition_questions():
    wrong = question(window(("y", "x")), "wrong", ("wrong_call",))
    correct = question()
    plan = plan_condition_questions(spec(), [wrong, correct], SearchBudget(64000))
    assert plan["question_count"] == 2 and plan["question"]["question_id"] == "wrong"
    assert all(window().predict(Observation(point, 0.)) == 0. for point in plan["probe"].sequence)
    next_plan = plan_condition_questions(spec(), [correct], SearchBudget(64000), serial=1)
    assert any(window().predict(Observation(point, 0.)) for point in next_plan["probe"].sequence)


def test_real_old_method_rebinds_after_extra_response_controls_are_removed_from_question():
    source_spec = spec(("x", "y", "unused_response_field"))
    methods = ProgramExperimentMethods()
    source = plan_condition_questions(source_spec, [question()], SearchBudget(64000), methods=methods)
    truth = binary("mul", constant(7.), window())
    freeze_laws(source, [truth, field("unused_response_field"), constant(0.)])
    rows = purchase(source, truth)
    assessment = assess_competing_object_probe(source, [(row, row) for row in rows], SearchBudget(1000))
    assert assessment["status"] == "distinguished"
    assert methods.observe(source, assessment, task_id="source", command_spec=source_spec)
    saved = deepcopy(methods.records)
    assert len(saved[0]["source_control_roles"]) == 3
    replay = plan_condition_questions(spec(("left", "right")),
        [question(window(("left", "right")))], SearchBudget(64000), methods=methods, task_id="renamed")
    assert replay["forecast_status"] == "predicted_distinction"
    assert replay["proposal_source"] == "learned_method" and replay["fresh_synthesis_skipped"]
    assert replay["proposal"]["method_id"] == saved[0]["method_id"]
    assert methods.records == saved  # Retrieval cannot add measured support.
    assert any(window(("left", "right")).predict(Observation(point, 0.)) for point in replay["probe"].sequence)


def test_temporal_question_selects_full_cycle_without_claiming_target_predictions():
    plan = plan_condition_questions(spec(), [question(depth=3)], SearchBudget(64000))
    assert plan["forecast_status"] == "predicted_distinction"
    assert plan["proposal"]["sequence_mode"] == "cycle"
    assert set(plan["predicted_question_phases"]) == {"enter", "maintain", "exit"}
    assert plan["hypotheses"] == [] and len(plan["question_hypotheses"]) == 2
    values = plan["question_hypotheses"][0]["predictions"]
    assert any(value > 0. for value in values) and any(value < 0. for value in values)
    assert "covered_by_paid_evidence" not in str(plan)
    truth = binary("mul", constant(7.), Program("event_delta", children=(window(),), lag=3))
    freeze_laws(plan, [truth, constant(0.)])
    assert max(plan["hypotheses"][0]["predictions"]) == 7.
    rows = purchase(plan, truth)
    actual = assess_competing_object_probe(plan, [(row, row) for row in rows], SearchBudget(1000))
    assert actual["status"] == "distinguished" and any(row.target < 0. for row in rows)
    methods = ProgramExperimentMethods()
    assert methods.observe(plan, actual, task_id="event_source", command_spec=spec())
    replay = plan_condition_questions(spec(("left", "right")),
        [question(window(("left", "right")), depth=3)], SearchBudget(64000),
        methods=methods, task_id="event_renamed")
    assert replay["proposal_source"] == "learned_method"
    assert replay["proposal"]["sequence_mode"] == "cycle"
    assert set(replay["predicted_question_phases"]) == {"enter", "maintain", "exit"}


def test_sensor_question_requests_paired_measurements_before_using_a_current_binding():
    sensor = question(window(("sensor_a", "sensor_b")))
    commands = spec(("knob_a", "knob_b"))
    missing = plan_condition_questions(commands, [sensor], SearchBudget(64000))
    assert missing["forecast_status"] == "requires_paired_measurements"
    assert missing["question_hypotheses"] == [] and missing["predicted_separation"] is None
    assert set(missing["missing_fields"]) == {"sensor_a", "sensor_b"}
    mapping = {"sensor_a": binary("mul", constant(-1.), field("knob_a")), "sensor_b": field("knob_b")}
    mapped = plan_condition_questions(commands, [sensor], SearchBudget(64000), binding=mapping)
    assert mapped["forecast_status"] == "predicted_distinction"
    for point in mapped["probe"].sequence:
        readings = {name: program.predict(Observation(point, 0.)) for name, program in mapping.items()}
        assert sensor["predicate"].predict(Observation(readings, 0.)) == 1.


def test_rejected_full_forecast_attempts_share_the_same_generation_allowance():
    shared, paid = SearchBudget(6000), SearchBudget(64000)
    first = plan_condition_questions(spec(), [question(window(origin=(2., 3.)))], paid,
                                     generation_budget=shared)
    second = plan_condition_questions(spec(), [question()], paid, serial=1, generation_budget=shared)
    assert first["probe"] is None
    assert shared.spent == paid.spent == first["generation_cost"] + second["generation_cost"] <= 6000
    assert first["compute_cost"] == first["generation_cost"]
    assert second["compute_cost"] == second["generation_cost"]
    empty = plan_condition_questions(spec(), [question()], SearchBudget(0))
    assert empty["probe"] is None and empty["forecast_status"] == "compute_exhausted"


def test_actual_full_response_failure_scope_restricts_only_the_same_laws_and_task():
    methods = ProgramExperimentMethods()
    truth = binary("mul", constant(7.), window())
    source = plan_condition_questions(spec(), [question()], SearchBudget(64000), methods=methods)
    freeze_laws(source, [truth, constant(0.)])
    rows = purchase(source, truth)
    actual = assess_competing_object_probe(source, [(row, row) for row in rows], SearchBudget(1000))
    assert methods.observe(source, actual, task_id="source", command_spec=spec())
    # A real earlier purchase produced no distinction between these two laws.
    # Its full-program failure must survive the compact question representation.
    responses = [truth, truth]
    unresolved = plan_condition_questions(spec(), [question()], SearchBudget(64000),
        methods=methods, task_id="successor", response_programs=responses)
    freeze_laws(unresolved, responses)
    rows = purchase(unresolved, truth)
    actual = assess_competing_object_probe(unresolved, [(row, row) for row in rows], SearchBudget(1000))
    assert actual["status"] == "unresolved"
    assert methods.observe(unresolved, actual, task_id="successor", command_spec=spec())["action"] == \
        "revise_program_experiment_method"
    saved = deepcopy(methods.records)
    replay = plan_condition_questions(spec(), [question()], SearchBudget(64000),
        methods=methods, task_id="successor", response_programs=responses)
    assert replay["forecast_status"] == "predicted_distinction"
    assert replay["proposal_source"] != "learned_method"
    assert replay["method_scope"]["blocked_scopes"][0]["reason"] == "unresolved_under_current_condition"
    new_task = plan_condition_questions(spec(), [question()], SearchBudget(64000),
        methods=methods, task_id="later", response_programs=responses)
    changed = plan_condition_questions(spec(), [question()], SearchBudget(64000),
        methods=methods, task_id="successor", response_programs=[truth, constant(0.)])
    assert new_task["proposal_source"] == changed["proposal_source"] == "learned_method"
    assert methods.records == saved


def test_known_bad_field_needed_only_by_full_responses_still_requests_recalibration():
    commands = spec(("x", "y", "z"))
    methods = ProgramExperimentMethods()
    truth = binary("mul", constant(7.), window())
    source = plan_condition_questions(commands, [question()], SearchBudget(64000), methods=methods)
    freeze_laws(source, [truth, field("z"), constant(0.)])
    rows = purchase(source, truth)
    actual = assess_competing_object_probe(source, [(row, row) for row in rows], SearchBudget(1000))
    assert methods.observe(source, actual, task_id="source", command_spec=commands)
    binding = {"sensor_z": field("z")}
    responses = [truth, field("sensor_z"), constant(0.)]
    drifted = plan_condition_questions(commands, [question()], SearchBudget(64000), methods=methods,
        binding=binding, task_id="drift", response_programs=responses)
    sequence = drifted["probe"].sequence
    drifted["hypotheses"] = [{"hypothesis_id": f"law_{index}", "candidate_id": f"law_{index}",
        "program": program.to_dict(), "predictions": _forecast(executable, sequence)}
        for index, (program, executable) in enumerate(zip(responses, [truth, field("z"), constant(0.)]))]
    drifted["predicted_fields"] = {"sensor_z": [point["z"] for point in sequence]}
    command_rows = purchase(drifted, truth)
    pairs = [(row, Observation({**row.features, "sensor_z": row.features["z"] + 1.}, row.target,
                              tuple({**point, "sensor_z": point["z"] + 1.} for point in row.history)))
             for row in command_rows]
    actual = assess_competing_object_probe(drifted, pairs, SearchBudget(1000))
    assert actual["calibration_status"] == "counterexample"
    assert methods.observe(drifted, actual, task_id="drift", command_spec=commands,
                           binding=binding)["reason"] == "calibration_counterexample"
    saved = deepcopy(methods.records)
    blocked = plan_condition_questions(commands, [question()], SearchBudget(64000), methods=methods,
        binding=binding, task_id="later", response_programs=responses)
    assert blocked["forecast_status"] == "requires_paired_measurements"
    assert blocked["failed_fields"] == ["sensor_z"] and blocked["predicted_separation"] is None
    assert blocked["question_hypotheses"] == [] and blocked["probe"] is not None
    repaired = plan_condition_questions(commands, [question()], SearchBudget(64000), methods=methods,
        binding={"sensor_z": binary("add", field("z"), constant(1.))}, task_id="later", response_programs=responses)
    assert repaired["forecast_status"] == "predicted_distinction"
    assert methods.records == saved
