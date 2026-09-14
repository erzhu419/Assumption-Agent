#!/usr/bin/env python3
"""Actual finite-gate experiments and condition-scoped method transfer."""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from random import Random
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/compound_experiment_methods_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
ARMS = ("v070", "compound_methods")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_runtime(path):
    sys.path.insert(0, str(path.resolve()))
    imports = {
        "hegel_machine.cognition.models": ["Observation", "Probe", "SearchBudget", "TaskSpec", "Candidate"],
        "hegel_machine.cognition.language": ["Program"],
        "hegel_machine.cognition.program_experiment_methods": ["ProgramExperimentMethods"],
        "hegel_machine.cognition.competing_object_experiments": ["plan_competing_object_probe", "assess_competing_object_probe"],
    }
    for module_name, names in imports.items():
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(path.resolve()):
            raise RuntimeError(f"loaded {module.__file__}, expected {path}")
        globals().update({name: getattr(module, name) for name in names})


def feature(name):
    return Program("feature", name=name)


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def task_settings(protocol, seed, case, stage):
    count = case["gates"]
    rng = Random(seed + count * 173)
    prior = protocol["public_prior"]
    centers = [center + rng.uniform(-prior["center_seed_jitter"], prior["center_seed_jitter"])
               + stage * prior["stage_center_shift"] for center in prior["centers"][:count]]
    offset = 37 if stage == 2 else 0
    controls = [f"c{index * 101 + 7001 + offset}" for index in range(count)]
    sensors = [f"m{index * 307 + 9013 + offset}" for index in range(count)]
    scales = ([.92, 1.05, .94] if stage < 2 else [1.08, .93, 1.12])[:count]
    offsets = ([0., 0., 0.] if stage == 0 else [.035, -.025, .018]
               if stage == 1 else [-.015, .02, -.03])[:count]
    return {"task_id": f"{seed}:{case['name']}:{stage}", "controls": controls, "sensors": sensors,
        "centers": centers, "scales": scales, "offsets": offsets,
        "half_width": prior["window_half_width"], "expression": case["expression"],
        "lower_candidates": [centers[0] - prior["window_half_width"] + delta
                             for delta in prior["first_lower_edge_candidate_offsets"]],
        "chosen_index": Random(seed + count * 7001 + stage * 1009).choice((1, 2))}


class GateApparatus:
    """Acquire independent readings from an opaque, finite-gate apparatus."""

    def __init__(self, settings):
        self._settings = settings
        self.spec = TaskSpec(settings["task_id"], "Predict a controlled process using the supplied gate prior.",
            tuple(settings["sensors"]), (), (), feature_bounds={name: (0., 1.) for name in settings["controls"]},
            max_probe_length=64)
        self.costs = {"observation_cost": 0, "measurement_cost": 0}

    def acquire(self, probe):
        settings, pairs = self._settings, []
        for point in probe.sequence:
            if set(point) != set(self.spec.feature_bounds) or any(not 0 <= value <= 1 for value in point.values()):
                raise ValueError("probe outside public control bounds")
            values = [point[name] * scale + offset for name, scale, offset in
                      zip(settings["controls"], settings["scales"], settings["offsets"])]
            target = 1.
            for index, (value, center) in enumerate(zip(values, settings["centers"])):
                lower = (settings["lower_candidates"][settings["chosen_index"]] if index == 0
                         else center - settings["half_width"])
                target *= float(lower <= value < center + settings["half_width"])
            pairs.append((Observation(dict(point), target),
                          Observation(dict(zip(settings["sensors"], values)), target)))
        self.costs["observation_cost"] += len(pairs)
        self.costs["measurement_cost"] += len(pairs) * len(settings["sensors"])
        return pairs


def candidate_programs(settings):
    programs = []
    for lower in settings["lower_candidates"]:
        gates = [binary("and", binary("ge", feature(name), constant(lower if index == 0
                         else center - settings["half_width"])),
                         binary("lt", feature(name), constant(center + settings["half_width"])))
                 for index, (name, center) in enumerate(zip(settings["sensors"], settings["centers"]))]
        program = constant(1.)
        for gate in reversed(gates):
            program = (binary("mul", gate, program) if settings["expression"] == "product" else
                       Program("if", children=(gate, program, constant(0.))))
        programs.append(program)
    return programs


def calibration(spec, pairs):
    first, second = pairs[:2]
    binding, inverse = {}, {}
    # Independent affine instrument roles are part of the finite task prior.
    # Coefficients come only from the purchased endpoint readings.
    for sensor, control in zip(spec.feature_names, spec.feature_bounds):
        x0, x1 = first[0].features[control], second[0].features[control]
        y0, y1 = first[1].features[sensor], second[1].features[sensor]
        slope = (y1 - y0) / (x1 - x0)
        offset = y0 - slope * x0
        binding[sensor] = binary("add", constant(offset), binary("mul", constant(slope), feature(control)))
        inverse[sensor] = {"control": control, "slope": slope, "offset": offset}
    return binding, inverse


def commands(measured, inverse):
    return {item["control"]: (measured[name] - item["offset"]) / item["slope"]
            for name, item in inverse.items()}


def initial_evidence(lab, settings):
    controls = list(lab.spec.feature_bounds)
    pairs = lab.acquire(Probe("paid_instrument_endpoints", tuple(dict.fromkeys(controls, value) for value in (0., 1.))))
    binding, inverse = calibration(lab.spec, pairs)
    center = dict(zip(settings["sensors"], settings["centers"]))
    lower = settings["lower_candidates"]
    stimuli = [center, {**center, settings["sensors"][0]: lower[0]},
               {**center, settings["sensors"][0]: (lower[2] + lower[3]) / 2}]
    pairs.extend(lab.acquire(Probe("paid_prior_discrimination", tuple(commands(point, inverse) for point in stimuli))))
    return pairs, binding, inverse


def fit_candidates(programs, pairs, budget):
    fitted = []
    for index, program in enumerate(programs):
        if not budget.spend(len(pairs)):
            break
        loss = sum((program.predict(measured) - measured.target) ** 2 for _, measured in pairs) / len(pairs)
        if loss <= 1e-7:
            fitted.append({"candidate": Candidate(f"gate_candidate_{index}", program, "supplied_finite_gate_prior"),
                           "hypothesis_id": f"gate_hypothesis_{index}", "training_loss": loss})
    return fitted


def serializable_plan(plan):
    return {**plan, "probe": asdict(plan["probe"]) if plan.get("probe") else None}


def run_task(settings, methods, previous_binding, protocol, stage):
    lab = GateApparatus(settings)
    budget = SearchBudget(protocol["acquisition"]["compute_budget"])
    prior = {key: value for key, value in settings.items() if key not in {"chosen_index", "scales", "offsets"}}
    pairs, binding, inverse = initial_evidence(lab, prior)
    programs = candidate_programs(prior)
    candidates = fit_candidates(programs, pairs, budget)
    if len(candidates) != 2:
        raise RuntimeError("paid prior observations did not leave the two predefined middle candidates")
    actions, experiments = [], []
    initial_ids = [item["hypothesis_id"] for item in candidates]

    def purchase(plan, chosen_binding, phase):
        observed = lab.acquire(plan["probe"])
        assessment = assess_competing_object_probe(plan, observed, budget)
        feedback = methods.observe(plan, assessment, task_id=settings["task_id"],
            command_spec=lab.spec, binding=chosen_binding)
        if feedback:
            actions.append(feedback)
        experiments.append({"phase": phase, "plan": serializable_plan(plan), "assessment": assessment,
            "actual_observations": [{"commands": a.features, "features": b.features, "target": b.target}
                                    for a, b in observed], "method_action": feedback})
        pairs.extend(observed)

    # This controlled challenge tests the saved method under its frozen old
    # instrument hypothesis before the newly measured calibration is adopted.
    if stage == 1 and previous_binding:
        plan = plan_competing_object_probe(lab.spec, candidates, budget, binding=previous_binding,
            methods=methods, serial=100, length=protocol["acquisition"]["probe_length"])
        if plan.get("probe"):
            purchase(plan, previous_binding, "old_calibration_challenge")
        if methods.to_dict():
            blocked = methods.propose(lab.spec, [item["candidate"].program for item in candidates],
                budget, binding=previous_binding, serial=101, length=4)
            actions.append({"action": "retry_frozen_instrument_hypothesis", "proposals": len(blocked),
                            "method_feedback": getattr(methods, "last_proposal", {})})
        candidates = fit_candidates(programs, pairs, budget)

    for index in range(protocol["acquisition"]["max_competition_experiments"]):
        if len(candidates) < 2:
            break
        plan = plan_competing_object_probe(lab.spec, candidates, budget, binding=binding,
            methods=methods, serial=200 + index, length=protocol["acquisition"]["probe_length"])
        actions.append({"action": "plan", "phase": "current_calibration", "plan": serializable_plan(plan)})
        if not plan.get("probe"):
            break
        purchase(plan, binding, "current_calibration")
        candidates = fit_candidates(programs, pairs, budget)
    if lab.costs["observation_cost"] > protocol["acquisition"]["observation_budget"] or lab.costs["measurement_cost"] > protocol["acquisition"]["measurement_budget"]:
        raise RuntimeError("study exceeded its original acquisition budget")
    selected = candidates[0]["candidate"].program if candidates else constant(0.)
    return {"task_id": settings["task_id"], "stage": stage, "initial_hypotheses": initial_ids,
        "remaining_hypotheses": [item["hypothesis_id"] for item in candidates],
        "selected_program": selected.to_dict(), "instrument_binding": {name: p.to_dict() for name, p in binding.items()},
        "inverse_calibration": inverse, "actions": actions, "experiments": experiments,
        "costs": {**lab.costs, "compute_cost": budget.spent}, "methods": methods.to_dict()}, binding


def evaluate(settings, result, protocol, seed):
    lab, rng = GateApparatus(settings), Random(seed + protocol["terminal"]["seed_offset"])
    program = Program.from_dict(result["selected_program"])
    inverse, rows, categories = result["inverse_calibration"], [], []
    for trajectory in range(protocol["terminal"]["trajectories"]):
        sequence = []
        for index in range(protocol["terminal"]["length"]):
            category = index % 4
            if category == 0:
                point = {name: rng.random() for name in lab.spec.feature_bounds}
            else:
                measured = {name: center + rng.uniform(-.4, .4) * settings["half_width"]
                            for name, center in zip(settings["sensors"], settings["centers"])}
                if category == 2:
                    item = rng.randrange(len(settings["sensors"]))
                    measured[settings["sensors"][item]] = settings["centers"][item] + settings["half_width"] * rng.uniform(1.2, 1.6)
                elif category == 3:
                    measured[settings["sensors"][0]] = rng.uniform(*settings["lower_candidates"][1:3])
                point = commands(measured, inverse)
            sequence.append(point)
            categories.append(category)
        rows.extend(lab.acquire(Probe(f"independent_terminal:{trajectory}", tuple(sequence))))
    errors = [(program.predict(measured) - measured.target) ** 2 for _, measured in rows]
    return {"loss": sum(errors) / len(errors), "rows": len(rows), "costs": lab.costs,
        "categories": [{"category": category, "rows": sum(c == category for c in categories),
            "positive_targets": sum(measured.target > 0 for c, (_, measured) in zip(categories, rows) if c == category),
            "loss": sum(error for c, error in zip(categories, errors) if c == category) / sum(c == category for c in categories)}
                       for category in range(4)]}


def run(output, arm, seed, protocol):
    results = []
    for case in protocol["cases"]:
        methods, previous_binding, case_results = ProgramExperimentMethods(), None, []
        for stage in range(len(protocol["stages"])):
            settings = task_settings(protocol, seed, case, stage)
            result, previous_binding = run_task(settings, methods, previous_binding, protocol, stage)
            methods = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
            case_results.append(result)
            print(json.dumps({"arm": arm, "seed": seed, "case": case["name"], "stage": stage,
                "remaining": len(result["remaining_hypotheses"]), "costs": result["costs"],
                "method_actions": [item["action"] for item in result["actions"] if "method" in item["action"]]}), flush=True)
        # No terminal outcome is available until this complete case sequence ends.
        for stage, result in enumerate(case_results):
            result["evaluation"] = evaluate(task_settings(protocol, seed, case, stage), result, protocol,
                                             seed + case["gates"] * 10007 + stage * 3011)
        results.append({"case": case["name"], "tasks": case_results})
    report = {"arm": arm, "seed": seed, "cases": results}
    write(output / "runs" / f"{seed}_{arm}.json", report)
    return report


def summarize(output):
    reports = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*.json"))]
    summaries = {}
    for arm in ARMS:
        tasks = [task for report in reports if report["arm"] == arm for case in report["cases"] for task in case["tasks"]]
        experiments = [experiment for task in tasks for experiment in task["experiments"]]
        summaries[arm] = {"tasks": len(tasks), "solved": sum(task["evaluation"]["loss"] <= 1e-6 for task in tasks),
            "identified_unique_program": sum(len(task["remaining_hypotheses"]) == 1 for task in tasks),
            "max_loss": max((task["evaluation"]["loss"] for task in tasks), default=None),
            "learning_costs": {key: sum(task["costs"][key] for task in tasks)
                               for key in ("observation_cost", "measurement_cost", "compute_cost")},
            "actual_compound_experiments": sum(experiment["plan"].get("proposal", {}).get("boundary", {}).get("solver_strategy") == "compound_conditions" for experiment in experiments),
            "distinguished_experiments": sum(experiment["assessment"]["status"] == "distinguished" for experiment in experiments),
            "scope_revisions": sum(action["action"] == "revise_program_experiment_method" for task in tasks for action in task["actions"]),
            "cross_task_method_uses": sum(action["action"] == "use_program_experiment_method" and action.get("cross_task", False) for task in tasks for action in task["actions"])}
    write(output / "summary.json", {"arms": summaries, "runs": reports})
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="compound_methods")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    source = args.output / (args.arm + "_source") / "src"
    if args.prepare_only:
        if source.exists():
            raise ValueError("runtime already frozen")
        original = ARTIFACTS / "v070_runtime/src" if args.arm == "v070" else ROOT / "src"
        shutil.copytree(original, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(Path(__file__), source.parent / Path(__file__).name)
        write(args.output / "protocol.json", protocol)
        return
    if args.summary_only:
        summarize(args.output)
        return
    if not source.exists():
        parser.error("prepare the arm source before running")
    load_runtime(source)
    run(args.output, args.arm, args.seed, protocol)


if __name__ == "__main__":
    main()
