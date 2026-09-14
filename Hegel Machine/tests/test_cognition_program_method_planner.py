"""Method memory must change the real planner's purchased experiment."""

from hegel_machine.cognition import program_disagreement_probes
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods


def _case(prefix, *, predicate="mul"):
    names = (f"{prefix}_left", f"{prefix}_right")
    controls = (f"{prefix}_a", f"{prefix}_b")
    bounds = {name: (-1., 1.) for name in controls}
    spec = TaskSpec(prefix, "public paired controls", controls, (), (), feature_bounds=bounds)
    binding = {name: Program("feature", name=control) for name, control in zip(names, controls)}
    threshold = .99990001 if predicate == "mul" else 1.99990001
    expression = Program(predicate, children=tuple(Program("feature", name=name) for name in names))
    programs = [Program("ge", children=(expression, Program("const", value=value)))
                for value in (threshold, threshold + 1e-8)]
    entries = [{"candidate": Candidate(str(index), program, "competing"), "training_loss": 0.}
               for index, program in enumerate(programs)]
    return spec, programs, entries, binding


def _acquire(plan, truth, binding):
    commands, measurements, pairs = [], [], []
    for point in plan["probe"].sequence:
        command = Observation(point, 0., tuple(commands))
        fields = {name: program.predict(command) for name, program in binding.items()}
        target = truth.predict(Observation(fields, 0., tuple(measurements)))
        pairs.append((Observation(point, target, tuple(commands)), Observation(fields, target, tuple(measurements))))
        commands.append({**point, "__target__": target})
        measurements.append({**fields, "__target__": target})
    return assess_competing_object_probe(plan, pairs, SearchBudget(1000))


def _learn():
    spec, programs, entries, binding = _case("source")
    plan = plan_competing_object_probe(spec, entries, SearchBudget(48000), binding=binding, serial=11)
    assert plan["proposal_source"] == "program_disagreement"
    actual = _acquire(plan, programs[0], binding)
    assert actual["status"] == "distinguished"
    methods = ProgramExperimentMethods()
    action = methods.observe(plan, actual, task_id="source", command_spec=spec, binding=binding)
    assert action["action"] == "learn_program_experiment_method"
    return methods


def test_real_planner_selects_paid_method_and_does_not_call_fresh_synthesis(monkeypatch):
    methods = _learn()
    spec, programs, entries, binding = _case("renamed_successor")

    def unexpected_fresh(*args, **kwargs):
        raise AssertionError("a prospectively useful learned method should skip fresh synthesis")

    monkeypatch.setattr(program_disagreement_probes, "propose_program_disagreement_probes", unexpected_fresh)
    budget = SearchBudget(48000)
    plan = plan_competing_object_probe(spec, entries, budget, binding=binding, serial=41, methods=methods)
    assert plan["proposal_source"] == "learned_method"
    assert plan["forecast_status"] == "predicted_distinction" and plan["predicted_separation"] > 1e-7
    assert plan["fresh_synthesis_skipped"] and plan["directed_candidates"] == 0
    assert plan["method_candidates"] == 1 and 0 < plan["generation_cost"] <= 6000
    assert plan["generation_cost"] == plan["method_search_cost"]
    assert len(methods.records[0]["support"]) == 1
    actual = _acquire(plan, programs[0], binding)
    assert actual["status"] == "distinguished"
    action = methods.observe(plan, actual, task_id="renamed_successor", command_spec=spec, binding=binding)
    assert action["action"] == "use_program_experiment_method" and action["cross_task"]


def test_nonmatching_paid_method_shares_generation_budget_and_uses_fresh_plan(monkeypatch):
    methods = _learn()
    spec, programs, entries, binding = _case("different_predicate", predicate="add")
    fresh = program_disagreement_probes.propose_program_disagreement_probes
    calls = []

    def record_fresh(command_spec, current_programs, budget, **kwargs):
        calls.append((budget.limit, budget.spent))
        return fresh(command_spec, current_programs, budget, **kwargs)

    monkeypatch.setattr(program_disagreement_probes, "propose_program_disagreement_probes", record_fresh)
    state = methods.to_dict()
    plan = plan_competing_object_probe(spec, entries, SearchBudget(48000), binding=binding, serial=23, methods=methods)
    assert calls and calls[0][0] == 6000 and calls[0][1] > 0
    assert not plan["fresh_synthesis_skipped"] and plan["method_candidates"] == 0
    assert plan["proposal_source"] == "program_disagreement" and "method_id" not in plan["proposal"]
    assert 0 < plan["method_search_cost"] < plan["generation_cost"] <= 6000
    actual = _acquire(plan, programs[0], binding)
    assert actual["status"] == "distinguished"
    action = methods.observe(plan, actual, task_id="different_predicate", command_spec=spec, binding=binding)
    assert action["action"] == "learn_program_experiment_method"
    assert methods.records[0] == state[0]
