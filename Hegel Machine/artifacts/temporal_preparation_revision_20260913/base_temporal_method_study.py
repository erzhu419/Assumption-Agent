#!/usr/bin/env python3
"""Paid temporal method transfer through a fixed, controlled hypothesis family."""

import argparse
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/temporal_method_reuse_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
ARMS = ("v075", "temporal_methods")
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_runtime(source):
    sys.path.insert(0, str(source.resolve()))
    exports = {"models": ("Observation", "Probe", "SearchBudget", "Candidate"),
        "language": ("Program",), "memory": ("KnowledgeLibrary",),
        "instrument_models": ("fit_instrument_model", "rebind_instrument_model"),
        "competing_object_experiments": ("plan_competing_object_probe", "assess_competing_object_probe")}
    for name, items in exports.items():
        module = importlib.import_module(f"hegel_machine.cognition.{name}")
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f"loaded {module.__file__}, expected {source}")
        globals().update({item: getattr(module, item) for item in items})
    name = "hegel_machine.cognition.temporal_method_lab"
    spec = importlib.util.spec_from_file_location(name, ARTIFACTS / "temporal_method_lab.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    globals()["TemporalMethodLab"] = module.TemporalMethodLab


def candidate_programs(laboratory, center, protocol):
    value = Program("feature", name=laboratory.sensor_name)
    gate = Program("and", children=(Program("ge", children=(value,
        Program("const", value=center - protocol["half_width"]))), Program("lt", children=(value,
        Program("const", value=center + protocol["half_width"])))))
    terms = [gate]
    if laboratory.case == "event":
        terms.append(Program("event_delta", lag=1, children=(gate,)))
    return [Program("mul", children=(Program("const", value=gain), term))
            for term in terms for gain in protocol["candidate_gains"]] + [Program("const", value=0.)]


def acquire_pairs(laboratory, probe, protocol):
    if (laboratory.observation_cost + probe.cost > protocol["observation_budget"] or
            laboratory.measurement_cost + 3 * probe.cost > protocol["measurement_budget"]):
        return None
    measured = laboratory.acquire(probe, (laboratory.sensor_name,)).observations
    control = laboratory.control_name
    commands = [Observation({control: row.features[control]}, row.target,
        tuple({control: past[control], "__target__": past["__target__"]} for past in row.history)) for row in measured]
    return list(zip(commands, measured))


def calibrate(laboratory, protocol, budget, seed, serial, actions, library):
    batches = []
    for index in range(protocol["calibration_batches"]):
        random = Random(seed + 1009 * serial + 7919 * index)
        probe = Probe(f"paid_calibration:{serial}:{index}", tuple(
            {laboratory.control_name: random.uniform(-1., 1.)} for _ in range(protocol["calibration_length"])))
        pairs = acquire_pairs(laboratory, probe, protocol)
        if pairs is None:
            return {}, []
        batches.append(pairs)
    before = budget.spent
    measured_spec = laboratory.spec_for((laboratory.sensor_name,))
    model = fit_instrument_model(laboratory.spec, measured_spec, batches[0], budget)
    binding = rebind_instrument_model(model, measured_spec)
    errors = {}
    for name, program in binding.items():
        if budget.spend(len(batches[1])):
            errors[name] = sum((program.predict(command) - measured.features[name]) ** 2
                for command, measured in batches[1]) / len(batches[1])
    binding = {name: program for name, program in binding.items()
               if name in errors and errors[name] <= protocol["tolerance"]}
    model.update(model_id=f"instrument_{len(library.instrument_models) + 1}",
        status="supported_on_measured_trajectory" if laboratory.sensor_name in binding else "unresolved",
        validation_losses=errors)
    library.instrument_models.append(model)
    actions.append({"action": "paid_instrument_calibration", "serial": serial,
        "observation_rows": sum(map(len, batches)), "compute_cost": budget.spent - before,
        "model_id": model["model_id"], "status": model["status"], "validation_losses": errors,
        "sensor_history_depth": model.get("fields", {}).get(laboratory.sensor_name, {}).get("history_depth")})
    return binding, [pair for batch in batches for pair in batch]


def plan_payload(plan):
    return {key: asdict(value) if key == "probe" and value is not None else value
            for key, value in plan.items()}


def run(output, protocol, arm, seed, source):
    streams, terminal_pending = [], []
    for case_index, case in enumerate(protocol["cases"]):
        library, tasks = KnowledgeLibrary(), []
        case_seed = seed + 100003 * case_index
        jitter = Random(case_seed).uniform(-protocol["center_jitter"], protocol["center_jitter"])
        for stage, center_base in enumerate(protocol["centers"]):
            center = center_base + jitter
            laboratory = TemporalMethodLab(case, stage, case_seed, center=center,
                half_width=protocol["half_width"], response_gain=protocol["true_gains"][stage],
                calibration_rows=protocol["calibration_length"] * protocol["calibration_batches"])
            budget, actions = SearchBudget(protocol["compute_budget"]), []
            before_methods = len(library.program_experiment_methods.records)
            binding, paid = calibrate(laboratory, protocol, budget, case_seed + stage * 10007, 0, actions, library)
            programs = candidate_programs(laboratory, center, protocol)
            entries = [{"candidate": Candidate(f"hypothesis_{index}", program, "supplied_temporal_prior"),
                        "hypothesis_id": f"hypothesis_{index}"} for index, program in enumerate(programs)]
            survivors = list(entries)
            for attempt in range(protocol["planning_attempts"]):
                plan = plan_competing_object_probe(laboratory.spec, entries, budget, binding=binding,
                    serial=stage * 10 + attempt, length=protocol["probe_length"], tolerance=protocol["tolerance"],
                    methods=library.program_experiment_methods, task_id=f"{case_seed}:{case}:{stage}")
                if plan.get("probe") is None:
                    actions.append({"action": "unresolved_temporal_plan", "plan": plan_payload(plan)})
                    break
                pairs = acquire_pairs(laboratory, plan["probe"], protocol)
                if pairs is None:
                    break
                evidence = assess_competing_object_probe(plan, pairs, budget, tolerance=protocol["tolerance"])
                feedback = library.program_experiment_methods.observe(plan, evidence,
                    task_id=f"{case_seed}:{case}:{stage}", command_spec=laboratory.spec, binding=binding,
                    tolerance=protocol["tolerance"])
                actions.append({"action": "paid_temporal_experiment", "plan": plan_payload(plan),
                    "assessment": evidence, "method_feedback": feedback})
                paid.extend(pairs)
                refuted = {item["hypothesis_id"] for item in evidence["hypotheses"] if item["status"] == "counterexample"}
                survivors = [entry for entry in survivors if entry["hypothesis_id"] not in refuted]
                if evidence["calibration_status"] == "counterexample" or plan["forecast_status"] == "requires_paired_measurements":
                    binding, more = calibrate(laboratory, protocol, budget, case_seed + stage * 10007,
                                               attempt + 1, actions, library)
                    paid.extend(more)
                    continue
                if len(survivors) == 1:
                    break
            selected = min(survivors or entries, key=lambda item: item["candidate"].program.size)
            if len(survivors) == 1 and budget.spend(len(paid)):
                actions.append({"action": "learn_supported_hypothesis", "compute_cost": len(paid)})
                # The chosen known law is learned only from actual observations.
                library.learn(selected["candidate"], laboratory.spec_for((laboratory.sensor_name,)),
                              [row for _, row in paid])
            after = library.to_dict()
            result = {"stage": stage, "selected_program": selected["candidate"].program.to_dict(),
                "identified": len(survivors) == 1, "survivors": [item["hypothesis_id"] for item in survivors],
                "costs": {**laboratory.costs, "compute_cost": budget.spent}, "actions": actions,
                "methods_before": before_methods, "methods_after": len(after["program_experiment_methods"]),
                "concepts_after": len(after["concepts"]), "instrument_models_after": len(after["instrument_models"])}
            tasks.append(result)
            terminal_pending.append((laboratory, result, case_seed + stage * 10007))
            library = KnowledgeLibrary.from_dict(json.loads(json.dumps(after)))
            print(json.dumps({"arm": arm, "seed": seed, "case": case, "stage": stage,
                "identified": result["identified"], "costs": result["costs"],
                "methods": result["methods_after"]}), flush=True)
        streams.append({"case": case, "tasks": tasks, "knowledge": library.to_dict()})
    for laboratory, result, task_seed in terminal_pending:
        rows = laboratory.evaluation_rows(seed=task_seed + protocol["terminal"]["seed_offset"],
            trajectories=protocol["terminal"]["trajectories"], length=protocol["terminal"]["length"])
        program = Program.from_dict(result["selected_program"])
        result["terminal"] = {"rows": len(rows), "loss": sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows),
            "nonzero_targets": sum(row.target != 0. for row in rows),
            "negative_targets": sum(row.target < 0. for row in rows), "measurement_cost": 3 * len(rows)}
    result = {"arm": arm, "seed": seed, "runtime_source": str(source.resolve().relative_to(ROOT)), "streams": streams}
    write(output / "runs" / f"{seed}_{arm}.json", result)
    return result


def summarize(output, protocol):
    result = {"cohorts": {}, "missing_runs": []}
    for cohort, seeds in (("development", protocol["development_seeds"]), ("formal", protocol["formal_seeds"])):
        result["cohorts"][cohort] = {}
        for arm in ARMS:
            tasks = []
            for seed in seeds:
                path = output / "runs" / f"{seed}_{arm}.json"
                if cohort == "development" and not path.exists():
                    path = ARTIFACTS / "development_v1/runs" / f"{seed}_{arm}.json"
                if not path.exists():
                    result["missing_runs"].append({"cohort": cohort, "arm": arm, "seed": seed})
                    continue
                run_result = json.loads(path.read_text())
                for stream in run_result["streams"]:
                    for task in stream["tasks"]:
                        experiments = [a for a in task["actions"] if a["action"] == "paid_temporal_experiment"]
                        calibrations = [a for a in task["actions"] if a["action"] == "paid_instrument_calibration"]
                        tasks.append({"seed": seed, "case": stream["case"], "stage": task["stage"],
                            "source": str(path.resolve().relative_to(ROOT)), "costs": task["costs"],
                            "identified": task["identified"], "terminal_loss": task["terminal"]["loss"],
                            "solved": task["terminal"]["loss"] <= protocol["terminal"]["solved_threshold"],
                            "planning_compute": sum(a["plan"]["compute_cost"] for a in task["actions"]
                                if a["action"] in {"paid_temporal_experiment", "unresolved_temporal_plan"}),
                            "knowledge_compute": sum(a["compute_cost"] for a in task["actions"] if a["action"] == "learn_supported_hypothesis"),
                            "calibration_compute": sum(a["compute_cost"] for a in calibrations),
                            "assessment_compute": sum(a["assessment"]["compute_cost"] for a in experiments),
                            "method_uses": sum(a.get("method_feedback", {}).get("action") == "use_program_experiment_method"
                                for a in experiments if a.get("method_feedback")),
                            "cross_task_method_uses": sum(a["method_feedback"].get("action") == "use_program_experiment_method"
                                and bool(a["method_feedback"].get("cross_task")) for a in experiments if a.get("method_feedback")),
                            "methods_learned": sum(a["method_feedback"].get("action") == "learn_program_experiment_method"
                                for a in experiments if a.get("method_feedback")),
                            "method_revisions": sum(a.get("method_feedback", {}).get("action") == "revise_program_experiment_method"
                                for a in experiments if a.get("method_feedback")),
                            "calibration_failures": sum(a["assessment"]["calibration_status"] == "counterexample" for a in experiments),
                            "recalibrations": max(0, len(calibrations) - 1), "actual_experiments": len(experiments),
                            "fresh_synthesis_skipped": sum(a["plan"].get("fresh_synthesis_skipped", False) for a in experiments),
                            "methods_after": task["methods_after"], "concepts_after": task["concepts_after"]})
            def aggregate(rows):
                return {"tasks": len(rows), "solved": sum(t["solved"] for t in rows),
                    "max_loss": max((t["terminal_loss"] for t in rows), default=None),
                    "costs": {key: sum(t["costs"][key] for t in rows) for key in (*COST_KEYS, "queries")},
                    **{key: sum(t[key] for t in rows) for key in ("identified", "planning_compute", "calibration_compute", "knowledge_compute", "cross_task_method_uses", "methods_learned",
                        "assessment_compute", "method_uses", "method_revisions", "calibration_failures", "recalibrations",
                        "actual_experiments", "fresh_synthesis_skipped")}}
            result["cohorts"][cohort][arm] = {"all": aggregate(tasks),
                "cases": {case: aggregate([t for t in tasks if t["case"] == case]) for case in protocol["cases"]},
                "stages": {case: {name: aggregate([t for t in tasks if t["case"] == case and t["stage"] == stage])
                            for stage, name in enumerate(protocol["stages"])} for case in protocol["cases"]}, "tasks": tasks}
    write(output / "summary.json", result)
    print(json.dumps({cohort: {arm: value["all"] for arm, value in arms.items()} for cohort, arms in result["cohorts"].items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison_v1")
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--arm", choices=ARMS, default="temporal_methods")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.summary_only:
        summarize(args.output, protocol)
        return
    if args.seed not in protocol["development_seeds"] + protocol["formal_seeds"]:
        parser.error("seed is not declared in the frozen protocol")
    if (args.output / "runs" / f"{args.seed}_{args.arm}.json").exists():
        parser.error("completed result already exists")
    source = args.source or ARTIFACTS / ("v075_runtime" if args.arm == "v075" else "v076_runtime") / "src"
    if not source.is_dir():
        parser.error("freeze the runtime source before running")
    load_runtime(source)
    run(args.output, protocol, args.arm, args.seed, source)


if __name__ == "__main__":
    main()
