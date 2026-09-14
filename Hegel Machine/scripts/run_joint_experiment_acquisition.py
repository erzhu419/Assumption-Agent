#!/usr/bin/env python3
"""Acquire a joint threshold experiment, then rebind its measured method.

The supplied hypothesis family is a threshold of a measured sum or product.
Thresholds and instrument coefficients are inferred from purchased readings.
"""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods


ROOT = Path(__file__).resolve().parents[1]


def feature(name):
    return Program("feature", name=name)


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


class ThresholdApparatus:
    """An actual bounded two-control fixture with opaque measured channels."""

    def __init__(self, controls, sensors, scales, threshold, operator):
        self.spec = TaskSpec("joint_threshold", "Predict a measured binary response.",
            tuple(sensors), (), (), feature_bounds={name: (0., 1.) for name in controls})
        self._controls, self._sensors = controls, sensors
        self._scales, self._threshold, self._operator = scales, threshold, operator
        self.costs = {"observation_cost": 0, "measurement_cost": 0}

    def acquire(self, probe):
        rows = []
        for point in probe.sequence:
            if set(point) != set(self._controls) or any(not 0 <= point[name] <= 1 for name in point):
                raise ValueError("public controls must be in [0, 1]")
            values = [point[name] * scale for name, scale in zip(self._controls, self._scales)]
            raw = values[0] + values[1] if self._operator == "add" else values[0] * values[1]
            target = float(raw >= self._threshold)
            rows.append((Observation(dict(point), target),
                         Observation(dict(zip(self._sensors, values)), target)))
        self.costs["observation_cost"] += len(rows)
        self.costs["measurement_cost"] += len(rows) * 2
        return rows


def paid_initial_rows(lab, expression):
    """Thirty paid diagonal bisections leave a narrow unresolved interval."""
    controls = list(lab.spec.feature_bounds)
    low, high = 0., 1.
    rows = lab.acquire(Probe("initial_extrema", tuple(dict.fromkeys(controls, value) for value in (low, high))))
    assert rows[0][1].target == 0. and rows[1][1].target == 1.
    for index in range(30):
        middle = (low + high) / 2
        observed = lab.acquire(Probe(f"initial_bracket:{index}", (dict.fromkeys(controls, middle),)))
        rows.extend(observed)
        if observed[0][1].target:
            high = middle
        else:
            low = middle
    return rows


def fit_instruments(spec, pairs):
    controls = list(spec.feature_bounds)
    # Each supplied measured role has a linear one-control prior. Two actual
    # extreme readings identify its coefficient; later readings verify it.
    first, second = pairs[:2]
    binding = {}
    for sensor, command in zip(spec.feature_names, controls):
        x0, x1 = first[0].features[command], second[0].features[command]
        y0, y1 = first[1].features[sensor], second[1].features[sensor]
        slope = (y1 - y0) / (x1 - x0)
        binding[sensor] = binary("add", constant(y0 - slope * x0),
                                  binary("mul", constant(slope), feature(command)))
    assert all(abs(model.predict(command) - measured.features[name]) < 1e-12
        for name, model in binding.items() for command, measured in pairs)
    return binding


def fit_thresholds(expression, pairs):
    lower = max(expression.predict(measured) for _, measured in pairs if not measured.target)
    upper = min(expression.predict(measured) for _, measured in pairs if measured.target)
    values = [lower + (upper - lower) * fraction for fraction in (.25, .75)]
    programs = [binary("ge", expression, constant(value)) for value in values]
    assert all(program.predict(measured) == measured.target for program in programs for _, measured in pairs)
    return programs, [lower, upper]


def frozen_planner():
    path = ROOT / "artifacts/joint_experiment_methods_20260913/v069_runtime/src/hegel_machine/cognition/program_disagreement_probes.py"
    spec = importlib.util.spec_from_file_location("hegel_machine.cognition._joint_trial_v069", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.propose_program_disagreement_probes


def make_plan(proposal, programs, binding):
    probe = proposal["probe"]
    return {"probe": probe,
        "proposal_source": "learned_method" if "method_id" in proposal else "program_disagreement",
        "proposal": {key: value for key, value in proposal.items() if key not in {"probe", "predictions"}},
        "hypotheses": [{"hypothesis_id": f"threshold_{index}", "candidate_id": f"threshold_{index}",
            "program": program.to_dict(), "predictions": prediction}
            for index, (program, prediction) in enumerate(zip(programs, proposal["predictions"]))],
        "predicted_fields": {name: [program.predict(Observation(point, 0.)) for point in probe.sequence]
                             for name, program in binding.items()}}


def run_trial(operator):
    methods = ProgramExperimentMethods()
    tasks = []
    baseline = frozen_planner()
    for index in range(2):
        controls, sensors = (("c_A", "c_B"), ("m_A", "m_B")) if index == 0 else (
            ("c_F", "c_E"), ("m_Z", "m_X"))
        scales = (1., 1.) if index == 0 else (.7, 1.4)
        maximum = sum(scales) if operator == "add" else scales[0] * scales[1]
        lab = ThresholdApparatus(controls, sensors, scales, maximum - .000999985401, operator)
        expression = binary(operator, *(feature(name) for name in sensors))
        old_rows = paid_initial_rows(lab, expression)
        binding = fit_instruments(lab.spec, old_rows)
        programs, old_interval = fit_thresholds(expression, old_rows)
        old_budget = SearchBudget(6000)
        old_proposals = baseline(lab.spec, programs, old_budget, binding=binding, serial=31)
        budget = SearchBudget(6000)
        if index == 0:
            proposals = propose_program_disagreement_probes(lab.spec, programs, budget, binding=binding, serial=31)
        else:
            # Restore JSON before transfer; the new task has different names,
            # calibration coefficients and a newly measured threshold bracket.
            methods = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
            proposals = methods.propose(lab.spec, programs, budget, binding=binding, serial=32)
        assert not old_proposals, "the fixed-context baseline unexpectedly reached the narrow joint region"
        assert proposals, "joint experiment could not be generated"
        proposal = proposals[0]
        assert proposal["boundary"]["solver_strategy"] == "joint_segment"
        assert len(proposal["boundary"]["controls"]) == 2
        plan = make_plan(proposal, programs, binding)
        actual = lab.acquire(plan["probe"])
        assessment = assess_competing_object_probe(plan, actual, budget)
        action = methods.observe(plan, assessment, task_id=f"{operator}:task_{index}",
                                 command_spec=lab.spec, binding=binding)
        revised_programs, interval = fit_thresholds(expression, [*old_rows, *actual])
        assert assessment["status"] == "distinguished"
        assert assessment["calibration_status"] == "supported_on_probe"
        assert interval[1] - interval[0] < old_interval[1] - old_interval[0]
        assert action["action"] == ("learn_program_experiment_method" if index == 0 else "use_program_experiment_method")
        if index:
            assert action["cross_task"] and action["source_task_id"] == f"{operator}:task_0"
        tasks.append({"task_id": f"{operator}:task_{index}", "controls": controls,
            "old_observation_rows": len(old_rows), "old_compatible_hypotheses": 2,
            "initial_threshold_interval": old_interval, "revised_threshold_interval": interval,
            "baseline_proposals": len(old_proposals), "baseline_compute_cost": old_budget.spent,
            "proposal": {**proposal, "probe": asdict(proposal["probe"])},
            "actual_observations": [{"commands": command.features, "features": measured.features,
                                     "target": measured.target} for command, measured in actual],
            "assessment": assessment, "method_action": action, "compute_cost": budget.spent,
            "costs": lab.costs, "revised_programs": [program.to_dict() for program in revised_programs]})
    return {"operator": operator, "tasks": tasks, "methods": methods.to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"hypothesis_prior": "threshold of two measured inputs with independently linear instruments",
              "trials": [run_trial(operator) for operator in ("add", "mul")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{ "operator": trial["operator"], "tasks": [{
        "task_id": task["task_id"], "status": task["assessment"]["status"],
        "action": task["method_action"]["action"], "costs": task["costs"],
        "compute_cost": task["compute_cost"]} for task in trial["tasks"]]} for trial in result["trials"]]))


if __name__ == "__main__":
    main()
