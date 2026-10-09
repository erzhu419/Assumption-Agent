"""Real temporal experiments teach a reusable operation, not a saved trace."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods

from test_cognition_program_experiment_methods import _measure, _plan


def field(name):
    return Program("feature", name=name)


def const(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def previous(name, delay):
    return Program("if", children=(Program("available", lag=delay),
        Program("lag", name=name, lag=delay), const(0.)))


def window(expression, low):
    return binary("and", binary("ge", expression, const(low)), binary("lt", expression, const(low + 4e-8)))


def case(*, renamed=False, difference=False, compound=False):
    control, sensor = ("new_knob", "new_reading") if renamed else ("u", "reading")
    delay = 3 if renamed else 1
    history = previous(control, delay)
    spec = TaskSpec("temporal", "public commands", (control,), (), (),
                    feature_bounds={control: (-1., 1.)}, max_probe_length=24)
    if compound:
        other = "other_new_reading" if renamed else "other_reading"
        binding = {sensor: binary("add", field(control), binary("mul", const(.5), history)),
                   other: binary("sub", history, field(control))}
        predicate = binary("and", window(field(sensor), .1312 if renamed else .2312),
                            window(field(other), -.2117 if renamed else -.4117))
    else:
        measured = binary("sub", field(control), history) if difference else history
        binding = {sensor: binary("add", const(.2), binary("mul", const(-.7), measured)) if renamed else measured}
        predicate = window(field(sensor), .413217 if renamed else .231234)
    programs = [predicate, binary("mul", const(3. if renamed else 2.), predicate)]
    return spec, programs, binding


def learn(spec, programs, binding, *, mode=None):
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, budget, binding=binding, serial=37)
    assert proposals
    proposal = next(item for item in proposals if mode is None or item["sequence_mode"] == mode)
    assert proposal["boundary"]["temporal_recipe"]
    plan = _plan(proposal, programs, binding)
    assessment = _measure(plan, programs[-1], binding)
    assert assessment["status"] == "distinguished"
    methods = ProgramExperimentMethods()
    action = methods.observe(plan, assessment, task_id="source", command_spec=spec, binding=binding)
    assert action["action"] == "learn_program_experiment_method"
    return ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict()))), proposal


@pytest.mark.parametrize("difference", (False, True))
def test_measured_prefix_recipe_rebinds_names_calibration_delay_and_current_threshold(difference, monkeypatch):
    spec, programs, binding = case(difference=difference)
    methods, source = learn(spec, programs, binding)
    current_spec, current_programs, current_binding = case(renamed=True, difference=difference)
    fresh_budget = SearchBudget(6000)
    fresh = propose_program_disagreement_probes(current_spec, current_programs, fresh_budget,
        binding=current_binding, serial=91)
    assert fresh
    # A recipe must not silently rerun the entire temporal proposal search.
    import hegel_machine.cognition.temporal_control_probes as generation
    monkeypatch.setattr(generation, "propose_temporal_control_probes", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("fresh synthesis used during method recall")))
    before = deepcopy(methods.records)
    budget = SearchBudget(6000)
    replay = methods.propose(current_spec, current_programs, budget, binding=current_binding, serial=91)
    assert replay and 0 < budget.spent < fresh_budget.spent
    assert budget.spent <= 4000
    assert replay[0]["boundary"]["history_depth"] == 3
    assert any(item["lag"] == 3 for group in replay[0]["boundary"]["control_time_dependencies"] for item in group)
    assert replay[0]["probe"].sequence != source["probe"].sequence
    assert all(set(point) == {"new_knob"} for point in replay[0]["probe"].sequence)
    assert methods.records == before
    plan = _plan(replay[0], current_programs, current_binding)
    actual = _measure(plan, current_programs[-1], current_binding)
    assert actual["status"] == "distinguished" and actual["calibration_status"] == "supported_on_probe"
    action = methods.observe(plan, actual, task_id="later", command_spec=current_spec, binding=current_binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]
    assert len(methods.records[0]["support"]) == 2


def test_compound_temporal_recipe_reuses_one_current_joint_solve():
    spec, programs, binding = case(compound=True)
    methods, source = learn(spec, programs, binding)
    recipe = methods.records[0]["temporal_recipe"]
    assert recipe["solver_strategy"] == "compound_conditions"
    assert len(recipe["predicate_patterns"]) == 2
    current_spec, current_programs, current_binding = case(renamed=True, compound=True)
    budget = SearchBudget(4000)
    replay = methods.propose(current_spec, current_programs, budget, binding=current_binding)
    assert replay and replay[0]["boundary"]["solve"]["solver_strategy"] == "compound_conditions"
    plan = _plan(replay[0], current_programs, current_binding)
    assert _measure(plan, current_programs[-1], current_binding)["status"] == "distinguished"
    assert replay[0]["probe"].sequence != source["probe"].sequence


def test_supported_cycle_preparation_expands_to_the_current_delay():
    spec, programs, binding = case()
    methods, source = learn(spec, programs, binding, mode="prefix_cycle")
    current_spec, current_programs, current_binding = case(renamed=True)
    replay = methods.propose(current_spec, current_programs, SearchBudget(4000), binding=current_binding)
    assert replay and replay[0]["sequence_mode"] == "prefix_cycle"
    assert len(replay[0]["probe"].sequence) == 3 * (3 + 1)
    assert len(source["probe"].sequence) == 3 * (1 + 1)
    assert _measure(_plan(replay[0], current_programs, current_binding),
                    current_programs[-1], current_binding)["status"] == "distinguished"


def test_actual_bad_dynamic_calibration_restricts_the_map_then_accepts_a_revised_map():
    spec, programs, binding = case()
    methods, _ = learn(spec, programs, binding)
    proposal = methods.propose(spec, programs, SearchBudget(4000), binding=binding)[0]
    plan = _plan(proposal, programs, binding)
    changed = {"reading": binary("add", const(.1), binding["reading"])}
    actual = _measure(plan, programs[-1], changed)
    assert actual["calibration_status"] == "counterexample"
    action = methods.observe(plan, actual, task_id="drift", command_spec=spec, binding=binding)
    assert action["reason"] == "calibration_counterexample"
    assert methods.propose(spec, programs, SearchBudget(4000), binding=binding, task_id="next") == []
    assert methods.last_proposal["status"] == "blocked_calibration"
    assert methods.propose(spec, programs, SearchBudget(4000), binding=changed, task_id="next")
    assert methods.records[0]["counterexamples"] and len(methods.records[0]["support"]) == 1


def test_actual_no_distinction_restricts_only_the_same_full_program_scope():
    spec, programs, binding = case()
    methods, _ = learn(spec, programs, binding)
    proposal = methods.propose(spec, programs, SearchBudget(4000), binding=binding)[0]
    indistinguishable = [programs[0], programs[0]]
    plan = _plan(proposal, indistinguishable, binding)
    actual = _measure(plan, programs[0], binding)
    assert actual["status"] == "unresolved"
    methods.observe(plan, actual, task_id="same", command_spec=spec, binding=binding)
    assert methods.propose(spec, indistinguishable, SearchBudget(4000), binding=binding, task_id="same") == []
    assert methods.last_proposal["status"] == "blocked_condition"
    assert methods.propose(spec, programs, SearchBudget(4000), binding=binding, task_id="same")


def test_unrelated_predicate_missing_calibration_and_unfunded_search_do_not_reuse():
    spec, programs, binding = case()
    methods, _ = learn(spec, programs, binding)
    unrelated = window(Program("abs", children=(field("reading"),)), .231234)
    assert methods.propose(spec, [unrelated, binary("mul", const(2.), unrelated)],
                           SearchBudget(4000), binding=binding) == []
    assert methods.propose(spec, programs, SearchBudget(4000)) == []
    exhausted = SearchBudget(0)
    assert methods.propose(spec, programs, exhausted, binding=binding) == [] and exhausted.spent == 0


def test_reused_target_strategy_continues_past_a_region_where_full_laws_agree():
    spec, programs, binding = case()
    methods, _ = learn(spec, programs, binding)
    first, second = window(field("reading"), .11), window(field("reading"), .43)
    current = [binary("add", first, second), binary("add", first, binary("mul", const(2.), second))]
    replay = methods.propose(spec, current, SearchBudget(4000), binding=binding)
    assert replay
    assert replay[0]["boundary"]["solve"]["target"] == pytest.approx(.43 + 2e-8)
    assert _measure(_plan(replay[0], current, binding), current[-1], binding)["status"] == "distinguished"


def test_same_sensor_at_different_times_remains_two_constraints_during_recall():
    spec = TaskSpec("different_times", "public command", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)}, max_probe_length=24)
    joint = binary("and", window(field("reading"), .231234),
                   window(Program("lag", name="reading", lag=1), -.41173))
    predicate = Program("if", children=(Program("available", lag=1), joint, const(0.)))
    programs = [predicate, binary("mul", const(2.), predicate)]
    binding = {"reading": previous("u", 1)}
    methods, _ = learn(spec, programs, binding)
    recipe = methods.records[0]["temporal_recipe"]
    assert sorted(recipe["predicate_offsets"]) == [0, 1]
    # Canonical expression order can put the earlier time first. Each offset
    # must still keep its own target interval and actual control dependency.
    constraints = methods.records[0]["source_boundary"]["solve"]["constraints"]
    by_offset = dict(zip(recipe["predicate_offsets"], constraints))
    assert by_offset[0]["lower"] == pytest.approx(.231234)
    assert by_offset[1]["lower"] == pytest.approx(-.41173)
    assert {(offset, tuple((item["control_role"], item["lag"]) for item in dependencies))
            for offset, dependencies in zip(recipe["predicate_offsets"],
                                             recipe["control_time_dependencies"])} == {
                (0, ((0, 1),)), (1, ((0, 2),))}
    replay = methods.propose(spec, programs, SearchBudget(4000), binding=binding)
    assert replay
    assert _measure(_plan(replay[0], programs, binding), programs[-1], binding)["status"] == "distinguished"
    renamed_spec = TaskSpec("renamed_times", "public command", ("new_u",), (), (),
                            feature_bounds={"new_u": (-1., 1.)}, max_probe_length=24)
    renamed_programs = [program.remap_features({"reading": "new_reading"}) for program in programs]
    renamed_binding = {"new_reading": previous("new_u", 3)}
    changed = methods.propose(renamed_spec, renamed_programs, SearchBudget(4000), binding=renamed_binding)
    assert changed and changed[0]["boundary"]["history_depth"] == 4
    assert _measure(_plan(changed[0], renamed_programs, renamed_binding),
                    renamed_programs[-1], renamed_binding)["status"] == "distinguished"
