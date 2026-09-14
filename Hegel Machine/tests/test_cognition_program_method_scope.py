"""A failed measurement map must not erase a useful research procedure."""

import json
from random import Random

from hegel_machine.cognition.continuous_state import _linear_fit, _sum_terms
from hegel_machine.cognition import program_disagreement_probes
from hegel_machine.cognition.competing_object_experiments import plan_competing_object_probe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods

from test_cognition_program_experiment_methods import _case, _c, _f, _measure, _op, _plan


def _learn_source():
    spec, programs, binding = _case()
    source = _plan(propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0],
                   programs, binding)
    methods = ProgramExperimentMethods()
    methods.observe(source, _measure(source, programs[0], binding), task_id="source", command_spec=spec, binding=binding)
    return methods, spec, programs, binding


def _paid_calibration(spec, instruments):
    # Fit field maps from a new purchased calibration batch, then check them
    # against separately acquired points before offering them to the method.
    rng = Random(491)
    points = [Observation({name: rng.uniform(*interval) for name, interval in spec.feature_bounds.items()}, 0.)
              for _ in range(36)]
    readings = [{name: program.predict(point) for name, program in instruments.items()} for point in points]
    controls = list(spec.feature_bounds)
    fitted = {}
    for name in instruments:
        coefficients = _linear_fit([[point.features[control] for point in points[:24]] for control in controls],
                                   [row[name] for row in readings[:24]])
        fitted[name] = _sum_terms([_f(control) for control in controls], coefficients[1:], coefficients[0])
        assert max((fitted[name].predict(point) - row[name]) ** 2
                   for point, row in zip(points[24:], readings[24:])) < 1e-14
    return fitted


def test_measured_calibration_failure_is_retained_until_real_recalibration_then_recipe_recovers():
    methods, spec, programs, binding = _learn_source()
    proposal = methods.propose(spec, programs, SearchBudget(3000), binding=binding, task_id="changed")[0]
    plan = _plan(proposal, programs, binding)
    shifted = {**binding, "left": _op("add", _c(.3), binding["left"])}
    actual = _measure(plan, programs[0], shifted)
    action = methods.observe(plan, actual, task_id="changed", command_spec=spec, binding=binding)
    assert action["reason"] == "calibration_counterexample"
    assert methods.records[0]["status"] == "supported" and len(methods.records[0]["support"]) == 1
    scope = methods.records[0]["failure_scopes"][0]
    assert [field["name"] for field in scope["fields"]] == ["left"]
    restored = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
    assert restored.propose(spec, programs, SearchBudget(3000), binding=binding, task_id="changed") == []
    assert restored.last_proposal["status"] == "blocked_calibration"
    other_change = {**binding, "weight": _op("mul", _c(.8), binding["weight"])}
    assert restored.propose(spec, programs, SearchBudget(3000), binding=other_change, task_id="later") == []
    repaired = _paid_calibration(spec, shifted)
    proposal = restored.propose(spec, programs, SearchBudget(3000), binding=repaired, task_id="changed")[0]
    fixed_plan = _plan(proposal, programs, repaired)
    actual = _measure(fixed_plan, programs[0], shifted)
    assert actual["status"] == "distinguished" and actual["calibration_status"] == "supported_on_probe"
    action = restored.observe(fixed_plan, actual, task_id="changed", command_spec=spec, binding=repaired)
    assert action["action"] == "use_program_experiment_method"
    assert len(restored.records[0]["counterexamples"]) == 1
    new_spec, new_programs, new_binding = _case(renamed=True)
    renamed = restored.propose(new_spec, new_programs, SearchBudget(3000), binding=new_binding, task_id="renamed")[0]
    new_plan = _plan(renamed, new_programs, new_binding)
    action = restored.observe(new_plan, _measure(new_plan, new_programs[0], new_binding),
        task_id="renamed", command_spec=new_spec, binding=new_binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]


def test_actual_unresolved_method_condition_does_not_suppress_another_task():
    methods, spec, programs, binding = _learn_source()
    weak = [_op("mul", _c(.0006), program) for program in programs]
    proposal = methods.propose(spec, weak, SearchBudget(3000), binding=binding, task_id="ambiguous")[0]
    plan = _plan(proposal, weak, binding)
    mixture = _op("mul", _c(.5), _op("add", *weak))
    actual = _measure(plan, mixture, binding)
    assert actual["status"] == "unresolved" and actual["calibration_status"] == "supported_on_probe"
    action = methods.observe(plan, actual, task_id="ambiguous", command_spec=spec, binding=binding)
    assert action["reason"] == "unresolved_under_current_condition"
    restored = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
    assert restored.propose(spec, weak, SearchBudget(3000), binding=binding, task_id="ambiguous") == []
    assert restored.last_proposal["status"] == "blocked_condition"
    new_proposal = restored.propose(spec, weak, SearchBudget(3000), binding=binding, task_id="other_task")[0]
    new_plan = _plan(new_proposal, weak, binding)
    actual = _measure(new_plan, weak[0], binding)
    assert actual["status"] == "distinguished"
    action = restored.observe(new_plan, actual, task_id="other_task", command_spec=spec, binding=binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]


def _two_windows(expressions, origins):
    programs = []
    for width in (2e-8, 3e-8):
        gates = [_op("and", _op("ge", expression, _c(origin)),
                     _op("lt", expression, _c(origin + (4e-8 if index == 0 else width))))
                 for index, (expression, origin) in enumerate(zip(expressions, origins))]
        programs.append(Program("if", children=(gates[0],
            Program("if", children=(gates[1], _c(7.), _c(0.))), _c(0.))))
    return programs


def test_paid_compound_recipe_rebinds_every_condition_with_shared_nonlinear_controls():
    spec = TaskSpec("source", "public controls", ("a", "b"), (), (),
                    feature_bounds={"a": (-1., 1.), "b": (-1., 1.)})
    binding = {"x": _f("a"), "y": _f("b")}
    programs = _two_windows([_f("x"), _f("y")], [.2312, -.4117])
    fresh = propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0]
    assert fresh["boundary"]["solver_strategy"] == "compound_conditions"
    source = _plan(fresh, programs, binding)
    actual = _measure(source, programs[0], binding)
    assert actual["status"] == "distinguished"
    methods = ProgramExperimentMethods()
    methods.observe(source, actual, task_id="source", command_spec=spec, binding=binding)
    assert len(methods.records[0]["condition_patterns"]) == 2
    methods = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
    current_spec = TaskSpec("renamed", "shared public controls", ("delta", "z"), (), (),
                            feature_bounds={"delta": (-2., 2.), "z": (-2., 2.)})
    z = _f("z")
    cubic = _op("mul", z, _op("mul", z, z))
    current_binding = {"new_x": _op("add", _op("mul", _c(.5), cubic), _op("mul", _c(1 / 3), _f("delta"))),
                       "new_y": _op("add", _op("mul", _c(-.5), cubic), _op("mul", _c(1 / 3), _f("delta")))}
    current_programs = _two_windows([_f("new_x"), _f("new_y")], [.13, -.51])
    replay = methods.propose(current_spec, current_programs, SearchBudget(4000),
        binding=current_binding, task_id="renamed")
    assert replay and len(replay[0]["boundary"]["constraints"]) == 2
    constraints = replay[0]["boundary"]["constraints"]
    assert constraints[0]["lower"] == .13 and constraints[1]["lower"] == -.51 + 2e-8
    final = Observation(replay[0]["probe"].sequence[-1], 0.)
    assert all(item["lower"] < Program.from_dict(item["expression"]).predict(final) < item["upper"]
               for item in constraints)
    current = _plan(replay[0], current_programs, current_binding)
    actual = _measure(current, current_programs[0], current_binding)
    assert actual["status"] == "distinguished"
    action = methods.observe(current, actual, task_id="renamed", command_spec=current_spec, binding=current_binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]


def test_compound_rebinding_keeps_variable_identity_shared_between_conditions():
    spec = TaskSpec("source", "three controls", ("a", "b", "c"), (), (),
        feature_bounds={name: (-1., 1.) for name in ("a", "b", "c")})
    binding = dict(zip(("x", "y", "z"), [_f("a"), _f("b"), _f("c")]))
    first = _op("add", _f("x"), _f("y"))
    programs = _two_windows([first, _op("add", _f("y"), _f("z"))], [.2312, -.4117])
    fresh = propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)[0]
    source = _plan(fresh, programs, binding)
    methods = ProgramExperimentMethods()
    methods.observe(source, _measure(source, programs[0], binding), task_id="source", command_spec=spec, binding=binding)
    # Both clauses still look like add(role_0, role_1) separately, but the
    # second clause now shares x with the first instead of y.
    changed = _two_windows([first, _op("add", _f("x"), _f("z"))], [.2312, -.4117])
    assert methods.propose(spec, changed, SearchBudget(4000), binding=binding, task_id="other") == []
    assert methods.last_proposal["status"] == "no_match"


def test_planner_purchases_recalibration_instead_of_reusing_a_refuted_forecast(monkeypatch):
    methods, spec, programs, binding = _learn_source()
    entries = [{"candidate": Candidate(str(index), program, "competing"), "training_loss": 0.}
               for index, program in enumerate(programs)]
    old = methods.propose(spec, programs, SearchBudget(3000), binding=binding, task_id="changed")[0]
    old_plan = _plan(old, programs, binding)
    actual_instruments = {**binding, "left": _op("add", _c(.3), binding["left"])}
    methods.observe(old_plan, _measure(old_plan, programs[0], actual_instruments),
                    task_id="changed", command_spec=spec, binding=binding)

    def unexpected_fresh(*args, **kwargs):
        raise AssertionError("a known failed calibration must be measured before new forecast search")

    monkeypatch.setattr(program_disagreement_probes, "propose_program_disagreement_probes", unexpected_fresh)
    plan = plan_competing_object_probe(spec, entries, SearchBudget(48000), binding=binding,
        methods=methods, task_id="changed", serial=44, length=12)
    assert plan["forecast_status"] == "requires_paired_measurements" and plan["probe"] is not None
    assert plan["proposal_source"] == "calibration" and plan["predicted_separation"] is None
    assert plan["predicted_fields"] == {} and all(item["predictions"] is None for item in plan["hypotheses"])
    assert plan["directed_candidates"] == plan["method_candidates"] == 0
    assert plan["method_scope"]["status"] == "blocked_calibration"
    # Purchase the returned calibration commands, fit an additive correction
    # from their readings, and validate it on separately measured commands.
    bought = [Observation(point, 0.) for point in plan["probe"].sequence]
    new_readings = [actual_instruments["left"].predict(row) for row in bought]
    correction = sum(value - binding["left"].predict(row) for value, row in zip(new_readings, bought)) / len(bought)
    repaired = {**binding, "left": _op("add", _c(correction), binding["left"])}
    validation = [Observation({name: low + fraction * (high - low)
        for name, (low, high) in spec.feature_bounds.items()}, 0.) for fraction in (.17, .43, .81)]
    assert all(abs(repaired["left"].predict(row) - actual_instruments["left"].predict(row)) < 1e-12
               for row in validation)
    recovered = plan_competing_object_probe(spec, entries, SearchBudget(48000), binding=repaired,
        methods=methods, task_id="changed", serial=45)
    assert recovered["proposal_source"] == "learned_method" and recovered["fresh_synthesis_skipped"]
    actual = _measure(recovered, programs[0], actual_instruments)
    assert actual["status"] == "distinguished" and actual["calibration_status"] == "supported_on_probe"
    action = methods.observe(recovered, actual, task_id="changed", command_spec=spec, binding=repaired)
    assert action["action"] == "use_program_experiment_method"
