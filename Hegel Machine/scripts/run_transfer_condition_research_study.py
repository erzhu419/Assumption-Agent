#!/usr/bin/env python3
"""Complete-library Unified research across activation, gain and name changes."""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from random import Random
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/transfer_condition_research_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
ARMS = ("v071", "condition_research")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_runtime(source):
    sys.path.insert(0, str(source.resolve()))
    imports = {
        "hegel_machine.cognition.models": ["Observation", "Probe", "TaskSpec", "Candidate", "ResearchConfig"],
        "hegel_machine.cognition.language": ["Program"],
        "hegel_machine.cognition.measurement_lab": ["MeasurementBatch"],
        "hegel_machine.cognition.memory": ["KnowledgeLibrary"],
        "hegel_machine.cognition.evaluation": ["loss"],
        "hegel_machine.cognition.unified_research": ["UnifiedResearchAgent"],
    }
    for module_name, names in imports.items():
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f"loaded {module.__file__}, expected {source}")
        globals().update({name: getattr(module, name) for name in names})

    class FinitePriorResearcher(UnifiedResearchAgent):
        """Supply fixed candidate laws; leave every transfer and acquisition path intact."""

        def _research_view(self, spec, training, validation, budget, *args, **kwargs):
            claims = []
            for index, program in enumerate(prior_programs(spec)):
                if not budget.spend(len(training) + len(validation)):
                    break
                claims.append({"candidate": Candidate(f"finite_prior_{index}", program, "finite_prior"),
                    "kind": "finite_prior", "family_id": "T02",
                    "training_loss": loss(program, training, 0., .5),
                    "validation_loss": loss(program, validation, 0., .5)})
            return claims

    globals()["FinitePriorResearcher"] = FinitePriorResearcher


def feature(name):
    return Program("feature", name=name)


def constant(value):
    return Program("const", value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


def prior_programs(spec):
    prior = json.loads(spec.context["finite_prior"])
    if prior["case"] == "ordinary_affine":
        return [binary("add", binary("mul", constant(gain), feature(spec.feature_names[0])), constant(offset))
                for gain in prior["gains"] for offset in prior["affine_offsets"]]
    programs = []
    for centers in prior["window_locations"]:
        windows = [binary("and", binary("ge", feature(name), constant(center - prior["window_half_width"])),
                          binary("lt", feature(name), constant(center + prior["window_half_width"])))
                   for name, center in zip(spec.feature_names, centers)]
        gate = binary("and", *windows)
        programs.extend(Program("if", children=(gate, constant(gain), constant(0.)))
                        for gain in prior["gains"])
    return programs


def task_settings(protocol, seed, case, stage):
    parameters = protocol["public_prior"]
    rng = Random(seed + 7919)
    source = [center + rng.uniform(-parameters["center_seed_jitter"], parameters["center_seed_jitter"])
              for center in parameters["window_centers"]]
    alternative = [center + shift for center, shift in zip(source, parameters["alternative_window_shift"])]
    generation = protocol["task_generation"]
    return {"case": case, "names": generation["names"][stage],
        "public_prior": {**parameters, "case": case, "window_locations": [source, alternative]},
        "true_gain": generation["true_gains"][stage],
        "true_location": generation["true_window_location"][stage],
        "true_offset": generation["true_affine_offset"][stage]}


class TransferLaboratory:
    channels = ()

    def __init__(self, settings):
        self._settings = settings
        self.acquisitions = []
        self.spec = TaskSpec("controlled_transfer", "Predict responses under public controls using the given finite law prior.",
            tuple(settings["names"]), (), (), feature_bounds={name: (0., 1.) for name in settings["names"]},
            context={"finite_prior": json.dumps(settings["public_prior"])}, max_probe_length=64)

    def spec_for(self, channels=()):
        return self.spec

    def acquire(self, probe, channels=()):
        if channels:
            raise ValueError("this apparatus has public control readings only")
        settings, rows, history = self._settings, [], []
        for point in probe.sequence:
            if set(point) != set(self.spec.feature_bounds) or any(not 0 <= value <= 1 for value in point.values()):
                raise ValueError("probe outside public control bounds")
            if settings["case"] == "ordinary_affine":
                target = settings["true_gain"] * point[settings["names"][0]] + settings["true_offset"]
            else:
                centers = settings["public_prior"]["window_locations"][settings["true_location"]]
                half_width = settings["public_prior"]["window_half_width"]
                target = settings["true_gain"] if all(center - half_width <= point[name] < center + half_width
                    for name, center in zip(settings["names"], centers)) else 0.
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.acquisitions.append({"probe": asdict(probe), "rows": len(rows)})
        return MeasurementBatch(rows, len(rows), 0, ())


def evaluate(settings, report, protocol, seed):
    lab, rng = TransferLaboratory(settings), Random(seed + protocol["terminal"]["seed_offset"])
    program = Program.from_dict(report["program"])
    observations, categories = [], []
    prior = settings["public_prior"]
    for trajectory in range(protocol["terminal"]["trajectories"]):
        sequence = []
        for index in range(protocol["terminal"]["length"]):
            category = index % 4 if settings["case"] == "narrow_windows" else 0
            if category == 0:
                point = {name: rng.random() for name in settings["names"]}
            elif category in (1, 2):
                point = {name: center + rng.uniform(-.4, .4) * prior["window_half_width"]
                         for name, center in zip(settings["names"], prior["window_locations"][category - 1])}
            else:
                point = {name: center + (3 + rng.random()) * prior["window_half_width"]
                         for name, center in zip(settings["names"], prior["window_locations"][0])}
            sequence.append(point)
            categories.append(category)
        observations.extend(lab.acquire(Probe(f"independent_terminal:{trajectory}", tuple(sequence))).observations)
    errors = [(program.predict(row) - row.target) ** 2 for row in observations]
    return {"loss": sum(errors) / len(errors), "rows": len(observations),
        "costs": {"observation_cost": len(observations), "measurement_cost": 0},
        "categories": [{"category": category, "rows": sum(item == category for item in categories),
            "positive_targets": sum(row.target > 0 for item, row in zip(categories, observations) if item == category),
            "loss": sum(error for item, error in zip(categories, errors) if item == category)
                    / sum(item == category for item in categories)} for category in sorted(set(categories))]}


def knowledge_counts(knowledge):
    return {key: len(knowledge.get(key, [])) for key in
            ("concepts", "abstracts", "meta_priors", "definitions", "program_experiment_methods")}


def run(output, protocol, arm, seed):
    cases = []
    for case in protocol["cases"]:
        library, results = KnowledgeLibrary(), []
        case_seed = seed + (protocol["task_generation"]["ordinary_seed_offset"] if case == "ordinary_affine" else 0)
        for stage in range(len(protocol["stages"])):
            settings = task_settings(protocol, seed, case, stage)
            lab = TransferLaboratory(settings)
            before = library.to_dict()
            config = ResearchConfig(**protocol["research_config"], seed=case_seed + stage * protocol["task_generation"]["config_seed_offset_per_stage"])
            report = FinitePriorResearcher(config, library=library,
                        measurement_budget=protocol["measurement_budget"]).run(lab)
            after = library.to_dict()
            actual_rows = sum(item["rows"] for item in lab.acquisitions)
            if actual_rows != report["costs"]["observation_cost"]:
                raise RuntimeError("reported observation costs differ from actual acquisitions")
            entry = {"stage": stage, "case": case, "research": report,
                "actual_queries": len(lab.acquisitions), "actual_rows": actual_rows,
                "knowledge_before": knowledge_counts(before), "knowledge_after": knowledge_counts(after)}
            results.append(entry)
            # Restore the entire library, including concepts and callable bodies.
            library = KnowledgeLibrary.from_dict(json.loads(json.dumps(after)))
            print(json.dumps({"arm": arm, "seed": seed, "case": case, "stage": stage,
                "status": report["status"], "kind": report["selected_kind"], "costs": report["costs"],
                "knowledge_before": entry["knowledge_before"], "knowledge_after": entry["knowledge_after"]}), flush=True)
        for stage, entry in enumerate(results):
            entry["evaluation"] = evaluate(task_settings(protocol, seed, case, stage), entry["research"],
                protocol, case_seed + stage * protocol["task_generation"]["config_seed_offset_per_stage"])
        cases.append({"case": case, "tasks": results, "knowledge": library.to_dict()})
    report = {"arm": arm, "seed": seed, "cases": cases}
    write(output / "runs" / f"{seed}_{arm}.json", report)
    return report


def summarize(output):
    reports = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*.json"))]
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {}
        for case in ("all", "narrow_windows", "ordinary_affine"):
            tasks = [task for report in reports if report["arm"] == arm for stream in report["cases"]
                     if case == "all" or stream["case"] == case for task in stream["tasks"]]
            summaries[arm][case] = {"tasks": len(tasks),
                "solved": sum(task["evaluation"]["loss"] <= 1e-6 for task in tasks),
                "max_loss": max((task["evaluation"]["loss"] for task in tasks), default=None),
                "learning_costs": {key: sum(task["research"]["costs"][key] for task in tasks)
                    for key in ("observation_cost", "measurement_cost", "compute_cost")},
                "actual_queries": sum(task["actual_queries"] for task in tasks),
                "tasks_calling_prior_concepts": sum(bool(task["research"].get("used_sources")) for task in tasks),
                "short_transfer_acceptances": sum(action["action"] == "validate_frozen_program"
                    and bool(action.get("short_trial")) and bool(action.get("accepted")) for task in tasks for action in task["research"]["actions"]),
                "false_short_transfer_tasks": sum(task["evaluation"]["loss"] > 1e-6 and any(
                    action["action"] == "validate_frozen_program" and action.get("short_trial")
                    and action.get("accepted") for action in task["research"]["actions"]) for task in tasks),
                "actual_transfer_condition_experiments": sum(action["action"] == "test_transfer_condition_predictions"
                    for task in tasks for action in task["research"]["actions"]),
                "reopened_condition_research": sum(action["action"] == "reopen_transfer_condition_research"
                    for task in tasks for action in task["research"]["actions"]),
                "actual_competing_experiments": sum(action["action"] == "test_competing_program_predictions"
                    for task in tasks for action in task["research"]["actions"])}
    result = {"arms": summaries, "runs": reports}
    write(output / "summary.json", result)
    print(json.dumps(summaries, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="condition_research")
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
        original = ARTIFACTS / "v071_runtime/src" if args.arm == "v071" else ROOT / "src"
        shutil.copytree(original, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(Path(__file__), source.parent / Path(__file__).name)
        write(args.output / "protocol.json", protocol)
        return
    if args.summary_only:
        summarize(args.output)
        return
    if not source.exists():
        parser.error("prepare the full arm source before running")
    load_runtime(source)
    run(args.output, protocol, args.arm, args.seed)


if __name__ == "__main__":
    main()
