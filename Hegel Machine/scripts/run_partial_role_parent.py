#!/usr/bin/env python3
"""Continue the paid partial-role scene into the existing full response researcher.

The role-only runs contain no response predictor. Their first response pool is
therefore fitted by the ordinary Unified researcher on retained observations,
then continued through run(previous=...). No apparatus restore/reacquisition is
used. Final random and public-control-step evaluation follows both frozen runs.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import monotonic, process_time

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint, write_text_incrementally
from hegel_machine.cognition.evaluation import initial_prediction, loss, predict
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ARMS = ("source", "with_frozen_source")
DEFAULT_SOURCE = Path("/mnt/hegel-snapshot-cow/hegel-v8-partial-role-frontier-20261029")
LIMITS = {"compute_cost": 510000, "observation_cost": 608, "measurement_cost": 7296}
BOOTSTRAP_COMPUTE = 90000
ACTIVITY_SECONDS = 1800
SEED = 97034007
TERMINAL = {"length": 48, "trajectories": 4, "random_seed": 98034011,
            "random_backgrounds": 2, "background_seed": SEED + 2000003,
            "sweep_fractions": [0., .25, .5, .75, 1.], "mse_tolerance": 1e-6}


def load(path):
    return json.loads(Path(path).read_text())


def plain(value):
    # Existing native reports occasionally retain executable Program terms.
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def save(path, value):
    write_json_checkpoint(Path(path), plain(value))


def read_role_scene(root, arm):
    directory = Path(root) / arm
    summary, model = load(directory / "summary.json"), load(directory / "parent_model.json")
    if not model.get("complete") or summary["confirmation"]["status"] != "supported":
        raise ValueError("Complete measured roles and their existing confirmation are required")
    records = load(directory / "laboratory_observations.json")
    # A pure public-data projection, not laboratory.restore(), which would buy again.
    batches = [{"probe": {"probe_id": item["probe_id"], "sequence": item["action"]["sequence"]},
        "channels": item["action"]["channels"], "observations": item["rows"],
        "transitions": item["transitions"]} for item in records]
    model = deepcopy(model)
    model["model_id"] = "paid_roles:" + arm
    method = load(directory / "formed_method.json")
    model["dependency_methods"] = method["dependency_methods"]
    old_costs = {key: summary["costs_including_shared_start"][key] for key in LIMITS}
    old_costs["queries"] = len(batches)
    return {"model": model, "batches": batches, "costs": old_costs,
        "records_ref": {"file": str(directory / "laboratory_observations.json")},
        "role_summary_ref": {"file": str(directory / "summary.json")},
        "role_model_ref": {"file": str(directory / "parent_model.json")}}


def native_config():
    return ResearchConfig(observation_budget=LIMITS["observation_cost"],
        compute_budget=LIMITS["compute_cost"], minimum_improvement=1e-7,
        max_rounds=6, max_history=4, active_probes=True, seed=SEED)


def observed(batch):
    return [Observation(row["features"], row["target"], tuple(row.get("history", [])))
            for row in batch["observations"]]


def bootstrap(agent, laboratory, scene, output):
    """Use the complete ordinary response-search route; no selected family supplied."""
    model, batches = scene["model"], scene["batches"]
    agent._object_models[("fine",)] = deepcopy(model)
    agent._object_schemas[("fine",)] = deepcopy(model["schema"])
    agent.library.object_models.append(deepcopy(model))
    agent._learn_instruments = agent._instrument_knowledge(laboratory)
    agent._revise_inputs = agent._revisable_inputs(laboratory)
    spec = agent._spec(laboratory, ("fine",))
    training = [row for batch in batches[:-1] for row in observed(batch)]
    validation = observed(batches[-1])
    transitions = [row for batch in batches[:-1] for row in batch["transitions"]]
    actions, transfer, graph = [], [], ResearchGraph()
    allowance = min(BOOTSTRAP_COMPUTE, (LIMITS["compute_cost"] - scene["costs"]["compute_cost"]) // 2)
    budget = SearchBudget(allowance)
    started = monotonic()
    try:
        candidates = agent._research_view(spec, training, validation, budget, actions, transfer, graph,
            transitions=transitions,
            observation_remaining=LIMITS["observation_cost"] - scene["costs"]["observation_cost"],
            measurement_remaining=LIMITS["measurement_cost"] - scene["costs"]["measurement_cost"])
    except Exception as error:
        save(output / "bootstrap_failure.json", {"error": repr(error), "compute_cost": budget.spent,
            "elapsed_seconds": monotonic() - started, "actions": actions,
            "paid_records_ref": scene["records_ref"], "role_costs": scene["costs"]})
        raise
    selected = agent._best(candidates)
    save(output / "bootstrap_pool.json", {"candidates": [encode_entry(item) for item in candidates],
        "compute_cost": budget.spent, "elapsed_seconds": monotonic() - started,
        "training_rows": len(training), "working_validation_rows": len(validation),
        "validation_scope": "previously_paid_role_confirmation; not fresh response confirmation",
        "paid_records_ref": scene["records_ref"], "actions": actions})
    if selected is None:
        return None, budget.spent
    selected = {**selected, "fallback": initial_prediction(training, spec.risk_quantile)}
    previous = {"task_id": laboratory.spec.task_id, "status": "unresolved",
        "program": selected["candidate"].program.to_dict(), "fallback": selected["fallback"],
        "selected_kind": selected["kind"], "selected_channels": ["fine"],
        "selected_structure": selected.get("structure"), "research_program": selected.get("research_program"),
        "selection_loss": selected["validation_loss"], "confirmation_loss": None,
        "selection_basis": selected.get("selection_basis", "paid_working_validation"),
        "actions": actions, "transfer": transfer, "research_graph": graph.to_dict(),
        "object_research": deepcopy(model), "learned_object_schema": deepcopy(model["schema"]),
        "batches": batches, "costs": {**scene["costs"],
            "compute_cost": scene["costs"]["compute_cost"] + budget.spent},
        "object_continuation_state": {"selected": encode_entry(selected),
            "finalists": [encode_entry({**item, "continuation_channels": ["fine"]}) for item in candidates],
            "schemas": [{"channels": ["fine"], "schema": deepcopy(model["schema"])}],
            "models": [{"channels": ["fine"], "model": deepcopy(model)}],
            "competitions": [{"features": list(names), "competition": deepcopy(value)}
                for names, value in agent._object_competitions.items()],
            "candidate_frontiers": [{"features": list(names), "entries": deepcopy(value)}
                for names, value in agent._object_candidate_frontiers.items()]}}
    saved = {key: value for key, value in previous.items() if key != "batches"}
    saved["batches_source_ref"] = scene["records_ref"]
    save(output / "bootstrap_previous.json", saved)
    return previous, budget.spent


def train(root, arm):
    timing = ExecutionTiming(("paid_restore", "bootstrap", "native_response", "save"))
    registration = load(root / "registration.json")
    with timing.phase("paid_restore"):
        scene = read_role_scene(registration["role_source_root"], arm)
    output = root / arm
    output.mkdir(exist_ok=False)
    started, cpu = monotonic(), process_time()
    with AnonymousObjectLab(**registration["settings"]) as laboratory:
        agent = UnifiedResearchAgent(native_config(), measurement_budget=LIMITS["measurement_cost"])
        with timing.phase("bootstrap"), python_execution_deadline(ACTIVITY_SECONDS, activity=True):
            previous, bootstrap_cost = bootstrap(agent, laboratory, scene, output)
        if previous is None:
            save(output / "summary.json", {"status": "no_bootstrap_response_candidate",
                "bootstrap_compute_cost": bootstrap_cost, "old_costs": scene["costs"],
                "new_costs": {"compute_cost": bootstrap_cost, "observation_cost": 0, "measurement_cost": 0}})
            return
        try:
            with timing.phase("native_response"), python_execution_deadline(
                    ACTIVITY_SECONDS - (monotonic() - started), activity=True):
                report = agent.run(laboratory, previous=previous, continuation_candidate=None)
        except Exception as error:
            save(output / "failure.json", {"error": repr(error), "bootstrap_compute_cost": bootstrap_cost,
                "actual_new_acquisition_costs": laboratory.costs,
                "native_compute_cost": "unknown; run failed before returning its account",
                "native_compute_upper_bound": LIMITS["compute_cost"] - previous["costs"]["compute_cost"],
                "paid_role_costs_ref": scene["role_summary_ref"],
                "new_acquisitions": laboratory.acquisitions,
                "elapsed_seconds": monotonic() - started, "cpu_seconds": process_time() - cpu,
                "execution_timing": timing.result()})
            raise
        old_count = len(scene["batches"])
        new_batches = report["batches"][old_count:]
        old_sequences = [batch["probe"]["sequence"] for batch in scene["batches"]]
        repetitions = [batch["probe"]["probe_id"] for batch in new_batches
            if batch["probe"]["sequence"] in old_sequences]
        # Store the new research once; retained source observations stay by reference.
        compact = {key: value for key, value in report.items() if key != "batches"}
        compact.update(paid_batches_ref=scene["records_ref"], new_batches=new_batches,
                       paid_role_summary_ref=scene["role_summary_ref"])
        with timing.phase("save"):
            save(output / "response_report.json", compact)
            save(output / "knowledge.json", agent.library.to_dict())
            frozen = {key: report[key] for key in ("program", "fallback", "selected_channels", "selected_kind", "status")}
            save(output / "frozen_predictor.json", frozen)
        summary = {"status": report["status"], "confirmation_loss": report["confirmation_loss"],
            "world_confirmation_threshold": native_config().minimum_improvement,
            "selected_channels": report["selected_channels"], "selected_kind": report["selected_kind"],
            "costs_including_paid_roles": report["costs"], "old_role_costs": scene["costs"],
            "new_costs": {key: report["costs"][key] - scene["costs"][key] for key in LIMITS},
            "bootstrap_compute_cost": bootstrap_cost, "bootstrap_pool_ref": {"file": "bootstrap_pool.json"},
            "retained_paid_rows": sum(len(batch["observations"]) for batch in scene["batches"]),
            "new_batches": len(new_batches), "repeated_old_sequences": repetitions,
            "role_research_actions": [item["action"] for item in report["actions"] if item["action"] in
                {"resume_object_induction", "test_object_dependency_question", "propose_discovered_objects"}],
            "elapsed_seconds": monotonic() - started, "cpu_seconds": process_time() - cpu,
            "execution_timing": timing.result(),
            "model_calls": 0, "independent_task_transfer_tested": False,
            "frozen_predictor_ref": {"file": "frozen_predictor.json"}}
        save(output / "summary.json", summary)


def evaluate_reports(settings, reports, terminal_config, output):
    """One physical collection per required bank, shared by every frozen report."""
    frozen, terminal = reports, terminal_config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    timing = ExecutionTiming(("collect_and_score", "save"))
    views = {tuple(item["selected_channels"]) for item in frozen.values()} - {()}
    views = sorted(views) or [()]
    programs = {arm: Program.from_dict(item["program"]) for arm, item in frozen.items()}
    results, paid_batches, new_compute = [], [], 0
    with (AnonymousObjectLab(**settings) as laboratory,
            timing.phase("collect_and_score"), python_execution_deadline(ACTIVITY_SECONDS, activity=True)):
        public = laboratory.spec.feature_bounds
        probes = [("random", laboratory.random_probe(terminal["random_seed"] + index * 10007,
            length=terminal["length"], name=f"terminal_random_{index}"))
            for index in range(terminal["trajectories"])]
        backgrounds = [{name: (lo + hi) / 2 for name, (lo, hi) in public.items()}]
        for index in range(terminal["random_backgrounds"]):
            rng = Random(terminal["background_seed"] + index * 10007)
            backgrounds.append({name: rng.uniform(lo, hi) for name, (lo, hi) in public.items()})
        for index, point in enumerate(backgrounds):
            for control, (low, high) in sorted(public.items()):
                sequence = [dict(point) for _ in range(3)] + [{**point, control: low + (high - low) * part}
                    for part in terminal["sweep_fractions"]] + [dict(point), dict(point)]
                probes.append(("public_control_step", Probe(f"terminal_step:{index}:{control}", tuple(sequence))))
        for kind, probe in probes:
            measured, item = {}, {"kind": kind, "probe": asdict(probe), "arms": {},
                                   "paid_observation_refs": []}
            try:
                for view in views:
                    measured[view] = laboratory.acquire(probe, view)
                    item["paid_observation_refs"].append({"file": "paid_observations.json",
                        "json_path": [len(paid_batches)]})
                    paid_batches.append({"probe_id": probe.probe_id,
                        "batch": asdict(measured[view])})
                sample = next(iter(measured.values())).observations
                commands, history = [], []
                for point, row in zip(probe.sequence, sample):
                    commands.append(Observation(dict(point), row.target, tuple(history)))
                    history.append({**point, "__target__": row.target})
                item.update(rows=len(sample), targets=[row.target for row in sample])
                for arm, predictor in frozen.items():
                    channels = tuple(predictor["selected_channels"])
                    rows = measured[channels].observations if channels else commands
                    predictions = []
                    item["arms"][arm] = {"predictions": predictions}
                    for row in rows:
                        new_compute += 1
                        predictions.append(predict(programs[arm], row, predictor["fallback"]))
                    item["arms"][arm]["mse"] = sum(
                        (value - row.target) ** 2 for value, row in zip(predictions, rows)) / len(rows)
                results.append(item)
            except Exception as error:
                save(output / "paid_observations.json", paid_batches)
                save(output / "partial_failure.json", {"error": repr(error),
                    "completed_probes": results, "pending_probe": item,
                    "paid_observations_ref": {"file": "paid_observations.json"},
                    "shared_actual_costs": {**laboratory.costs, "compute_cost": new_compute},
                    "execution_timing": timing.result()})
                raise
        terminal_costs = {**laboratory.costs, "compute_cost": new_compute}
    per_arm = {}
    for arm in frozen:
        random_mse = mean(item["arms"][arm]["mse"] for item in results if item["kind"] == "random")
        step_mse = mean(item["arms"][arm]["mse"] for item in results if item["kind"] == "public_control_step")
        per_arm[arm] = {"random_mse": random_mse, "public_control_step_mse": step_mse,
            "prediction_pass": max(random_mse, step_mse) <= terminal["mse_tolerance"],
            "native_status": frozen[arm].get("status", "frozen"), "threshold": terminal["mse_tolerance"]}
    summary = {"arms": per_arm, "shared_actual_costs": terminal_costs,
        "actual_legal_views": views, "raw_ref": {"file": "raw_predictions.json"},
        "paid_observations_ref": {"file": "paid_observations.json"},
        "protocol": terminal, "learning_frozen_before_evaluation": True}
    with timing.phase("save"):
        save(output / "paid_observations.json", paid_batches)
        save(output / "raw_predictions.json", results)
    summary["execution_timing"] = timing.result()
    save(output / "summary.json", summary)
    return summary


def evaluate(root):
    registration = load(root / "registration.json")
    frozen = {arm: load(root / arm / "frozen_predictor.json") for arm in ARMS}
    return evaluate_reports(registration["settings"], frozen, registration["terminal"], root / "terminal")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("register", "inspect", "train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "register":
        original = load(args.role_source / "registration.json")
        args.output.mkdir(parents=True, exist_ok=False)
        repo = Path(__file__).resolve().parents[1]
        for relative in ["scripts/run_partial_role_parent.py", *["src/hegel_machine/cognition/" + name + ".py"
                for name in ("unified_research", "object_continuation", "object_induction_continuation",
                             "unified_structure", "anonymous_object_lab", "models")]]:
            target = args.output / "source_at_run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text_incrementally(target, (repo / relative).read_text())
        save(args.output / "registration.json", {"role_source_root": str(args.role_source.resolve()),
            "settings": original["settings"], "arms": ARMS, "total_limits_including_old_roles": LIMITS,
            "native_config": asdict(native_config()), "bootstrap_compute_max": BOOTSTRAP_COMPUTE,
            "activity_seconds_per_arm": ACTIVITY_SECONDS,
            "terminal": TERMINAL, "terminal_costs_separate_from_research": True,
            "terminal_seed_adjustment": "98034010 has 12 already-paid role-confirmation rows; use 98034011",
            "scope": "complete the paid same-physical development parent; no new apparatus or seed search",
            "budget_source": "run_anonymous_revisit_study.py:167-174 pooled legacy contract",
            "old_records_preserved_by_reference": True, "model_calls": 0})
    elif args.mode == "inspect":
        registration = load(args.output / "registration.json")
        for arm in ARMS:
            scene = read_role_scene(registration["role_source_root"], arm)
            print(json.dumps({"arm": arm, "rows": sum(len(b["observations"]) for b in scene["batches"]),
                "complete_roles": scene["model"]["complete"], "old_costs": scene["costs"],
                "remaining": {key: LIMITS[key] - scene["costs"][key] for key in LIMITS}}))
    elif args.mode == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        train(args.output, args.arm)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
