#!/usr/bin/env python3
"""Ordinary Python research from the same paid source and first counterexample."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import run_instrument_relation_diagnosis as diagnosis


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
DEFAULT_OUTPUT = ROOT / "artifacts/instrument_relation_code_comparison_20260914"
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def save(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=True, allow_nan=True)


def load_runtime(protocol):
    runtime = protocol["code_runtime"]
    module, source = diagnosis.load({**protocol, "new_development_runtime": runtime}, "revised")
    for name, attributes in {
        "code_agent": ("CodeAgentConfig", "evaluate_python_model"),
        "unified_code_control": ("UnifiedCodeSession",),
    }.items():
        imported = importlib.import_module("hegel_machine.cognition." + name)
        globals().update({attribute: getattr(imported, attribute) for attribute in attributes})
    # The interface adapter can explicitly select model effort while the
    # experiment and Python-agent runtime remain the frozen 0.84 implementation.
    provider_path = ROOT / protocol.get("code_provider_source", "src/hegel_machine/cognition/code_agent.py")
    provider_spec = importlib.util.spec_from_file_location(
        "hegel_machine.cognition.study_code_provider", provider_path)
    provider_module = importlib.util.module_from_spec(provider_spec)
    sys.modules[provider_spec.name] = provider_module
    provider_spec.loader.exec_module(provider_module)
    globals()["CodexCodeProvider"] = provider_module.CodexCodeProvider
    return module, source


def task_for(protocol, seed, case):
    return {"environment": "objects", "seed": seed, "apparatus_seed": seed,
            "objects": protocol["objects"], **protocol["apparatus_settings"],
            "relation_condition": protocol["cases"][case]["current_relation_condition"]}


def _paid_records(records):
    from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch
    result = []
    for record in records:
        probe = diagnosis.Probe(record["probe"]["probe_id"], tuple(record["probe"]["sequence"]))
        batch = ObjectMeasurementBatch.from_dict(record["batch"])
        result.append({"probe": asdict(probe), "channels": list(batch.channels),
            "command_observations": [asdict(command) for command, _ in diagnosis.paired(probe, batch)],
            "observations": deepcopy(record["batch"]["observations"]),
            "transitions": deepcopy(record["batch"]["transitions"]),
            "observation_cost": batch.observation_cost, "measurement_cost": batch.measurement_cost})
    return result


def prepare_case(module, protocol, seed, case, source, *, correct_input_records=()):
    """Supply prior evidence separately from current response-fitting rows."""
    from instrument_relation_code_source import export_source
    started = time.perf_counter()
    index = list(protocol["cases"]).index(case)
    library = diagnosis.KnowledgeLibrary.from_dict(deepcopy(source["knowledge"]))
    agent = diagnosis.agent_for(protocol, seed, library)
    setup_budget = diagnosis.SearchBudget(protocol["budgets_per_case"]["compute_budget"])
    batches = diagnosis.restore_pairs(source)
    extra_records = list(correct_input_records)
    with diagnosis.laboratory(module, protocol, seed, "source_less") as source_lab:
        if case == "relation_only":
            if not extra_records:
                raise ValueError("the correct-input source control requires its recorded paid batches")
            # Replay only the public source-side preparation computation; the
            # recorded measurements are imported, never remeasured under the new law.
            path = diagnosis.OUTPUT / "common_setup/instrument_condition_probes.py"
            spec = importlib.util.spec_from_file_location(
                "hegel_machine.cognition.code_source_input_conditions", path)
            setup = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(setup)
            binding = agent._instrument_binding(source_lab.spec_for(("fine",)), source_lab.spec)
            setup.instrument_condition_probe(binding, source_lab.spec, batches, setup_budget,
                reference=diagnosis.edge(protocol, "source_background", focus=1.).sequence[0], serial=0)
            batches.extend(diagnosis.restore_pairs({"batches": extra_records}))
            agent._fit_instruments(source_lab.spec, source_lab.spec_for(("fine",)),
                                   batches, setup_budget, [])
        measured_spec = source_lab.spec_for(("fine",))
        binding = agent._instrument_binding(measured_spec, source_lab.spec)
        schema = json.loads(measured_spec.context["object_schema"])
    source_prior = source["knowledge"]["meta_priors"][0]["structure"]
    python_source = export_source(source["measured_program"],
        {name: program.to_dict() for name, program in binding.items()}, schema)
    prior = {"radius": source_prior["fits"][0]["radius"],
             "radius_interval": source_prior["radius_interval"], "object_schema": schema,
             "source_fit_mse": source["source_loss"]}
    files = {"shared/source_model.py": python_source,
             "shared/source_measured_model.py": "from shared.source_model import predict_measured as predict\n",
             "shared/source_prior.json": json.dumps(prior, indent=2),
             "shared/source_notes.txt": (
                 "The source mechanism and calibration are supplied equally to both researchers. "
                 "shared/source_model.py is editable ordinary Python; predict takes command rows and "
                 "predict_measured takes paid sensor rows. shared/source_measured_model.py exposes the same "
                 "measured function as predict, directly usable with evaluate/finish and channels=['fine']. "
                 "Choose the model and view from your evidence. run with channels refreshes "
                 "model_observations.json in that view for fitting and loss calculations. "
                 "source_evidence.json contains all paid source "
                 "trajectories, their command/sensor pairs and before/after transitions. These are old "
                 "source responses, not current fitting targets. observations.json contains the current "
                 "task's paid evidence. source_counterexample.json records the prescribed first query "
                 "and the two prior forecasts. Additional experiments and changes are your decision.")}
    source_records = _paid_records([*source["batches"], *extra_records])
    probe = diagnosis.edge(protocol, "first_parameter_counterexample")
    namespace = {}
    exec(compile(python_source, "shared/source_model.py", "exec"), namespace)
    history, forecast_rows = [], []
    for point in probe.sequence:
        forecast_rows.append({"features": dict(point), "history": tuple(history)})
        history.append(dict(point))
    forecasts = [{"radius": radius,
                  "predictions": [namespace["predict"](row, radius=radius) for row in forecast_rows]}
                 for radius in (prior["radius"], .7)]
    setup_budget.spend(len(forecast_rows) * len(forecasts))
    with module.laboratory_for(task_for(protocol, seed, case)) as lab:
        batch = lab.acquire(probe, ())
    setup_costs = {"observation_cost": batch.observation_cost + sum(r["observation_cost"] for r in source_records[len(source["batches"]):]),
        "measurement_cost": batch.measurement_cost + sum(r["measurement_cost"] for r in source_records[len(source["batches"]):]),
        "compute_cost": setup_budget.spent, "queries": 1 + len(extra_records)}
    initial_costs = {**{name: setup_costs[name] for name in COSTS},
        "python_execution_seconds": 0., "model_calls": 0, "reported_tokens": 0}
    transitions = ([{"channels": [], "observation_start": 0, "transitions": deepcopy(batch.transitions)}]
                   if batch.transitions else [])
    agent_state = {"observations": [asdict(row) for row in batch.observations],
        "transition_batches": transitions, "events": [], "files": deepcopy(files), "costs": initial_costs,
        "tokens_available": True, "model": None, "selected_channels": [], "lesson": "",
        "reused_files": [], "evaluated_model": None}
    return {"seed": seed, "case": case, "case_index": index, "task": task_for(protocol, seed, case),
        "files": files, "data_files": {"source_evidence.json": json.dumps({"batches": source_records}),
            "source_counterexample.json": json.dumps({"probe": asdict(probe),
                "observations": agent_state["observations"], "forecasts": forecasts})},
        "first_probe": asdict(probe), "agent_state": agent_state,
        "setup_costs": setup_costs, "source_costs": deepcopy(source["costs"]),
        "preparation_wall_seconds": time.perf_counter() - started}


def new_session(module, protocol, directory, prepared, *, state=None, provider_factory=None):
    """Each neutral workspace starts from its own copy of the common source."""
    directory = Path(directory)
    workspace = directory / "workspace"
    task_workspace = workspace / "task_000"
    task_workspace.mkdir(parents=True, exist_ok=True)
    # Data stay as ordinary local files, without repeating long source
    # trajectories inside every provider request's code-file dictionary.
    for name, contents in prepared["data_files"].items():
        (task_workspace / name).write_text(contents)

    def laboratory(item):
        lab = module.laboratory_for(item)
        probe = prepared["first_probe"]
        lab.acquire(diagnosis.Probe(probe["probe_id"], tuple(probe["sequence"])), ())
        return lab

    provider_factory = provider_factory or (lambda path: CodexCodeProvider(path, **protocol["provider"]))
    session = UnifiedCodeSession(workspace, tasks=[prepared["task"]],
        config=CodeAgentConfig(**protocol["code_config"]), state=state,
        provider_factory=provider_factory, lab_factory=laboratory)
    if state is None:
        session.state["knowledge"] = {"files": deepcopy(prepared["files"]), "lessons": [],
                                      "shared_artifacts": list(prepared["data_files"])}
        session.state["active"] = {"task_position": 0, "agent_state": deepcopy(prepared["agent_state"])}
        session._costs()
    return session


def learn_case(module, protocol, directory, prepared, *, provider_factory=None):
    directory = Path(directory)
    path = directory / "code_checkpoint.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    state = previous.get("session")
    if state and state["results"]:
        return previous
    started = time.perf_counter()
    prior_seconds = previous.get("learning_wall_seconds", 0.)

    def checkpoint(current):
        result = {"seed": prepared["seed"], "case": prepared["case"], "case_index": prepared["case_index"],
            "session": current, "setup_costs": prepared["setup_costs"], "source_costs": prepared["source_costs"],
            "preparation_wall_seconds": prepared["preparation_wall_seconds"],
            "learning_wall_seconds": prior_seconds + time.perf_counter() - started}
        save(path, result)
        active = current.get("active")
        print(json.dumps({"seed": prepared["seed"], "case_index": prepared["case_index"],
            "status": current["status"], "costs": current["costs"],
            "events": len(active["agent_state"]["events"]) if active else None}), flush=True)

    session = new_session(module, protocol, directory, prepared, state=state, provider_factory=provider_factory)
    session.run(on_checkpoint=checkpoint)
    checkpoint(session.to_dict())
    return json.loads(path.read_text())


def evaluate_case(module, protocol, checkpoint):
    """Score the retained Python model on the common random and boundary rows."""
    state = checkpoint["session"]
    if len(state["results"]) != len(state["tasks"]):
        raise ValueError("complete learning before terminal evaluation")
    result, task = state["results"][0], state["tasks"][0]
    setting = protocol["terminal"]
    expected = setting["random_length"] * setting["random_trajectories"] + (
        len(setting["boundary_focus"]) * len(setting["boundary_offsets"]) * setting["boundary_probe_length"])
    if not result["model"]:
        return {"status": "missing_model", "solved": False, "random_mse": None, "boundary_mse": None,
                "expected_rows": expected, "rows": 0, "queries": 0, **{name: 0 for name in COSTS}}
    channels = tuple(result["selected_channels"])
    with module.laboratory_for(task) as lab:
        random_rows = lab.evaluation_rows(channels, seed=task["seed"] + setting["seed_offset"],
            length=setting["random_length"], trajectories=setting["random_trajectories"])
        rows, boundaries = list(random_rows), []
        for focus in setting["boundary_focus"]:
            for offset in setting["boundary_offsets"]:
                probe = diagnosis.edge(protocol, "terminal_boundary", focus + offset, heldout=True)
                actual = lab.acquire(probe, channels).observations
                boundaries.append((focus + offset, len(rows), len(actual)))
                rows.extend(actual)
        rate = sum(channel.cost_per_row for channel in lab.channels if channel.name in channels)
        risk_quantile = lab.spec.risk_quantile
    scored = evaluate_python_model(result["workspace"], result["model"], rows,
        timeout_seconds=state["config"]["execution_seconds"], risk_quantile=risk_quantile)
    report = {"status": "completed" if scored["returncode"] == 0 else "model_failed",
        "returncode": scored["returncode"], "stderr": scored.get("stderr"),
        "selected_channels": list(channels), "model_origin": result["model_origin"],
        "rows": len(rows), "expected_rows": expected, "observation_cost": len(rows),
        "queries": setting["random_trajectories"] + len(boundaries),
        "measurement_cost": len(rows) * rate, "compute_cost": len(rows),
        "random_mse": None, "boundary_mse": None, "solved": False}
    if scored["returncode"] == 0:
        errors = [(prediction - row.target) ** 2 for prediction, row in zip(scored["predictions"], rows)]
        mean = lambda start, length: sum(errors[start:start + length]) / length
        report.update(random_mse=mean(0, len(random_rows)), boundary_mse=mean(len(random_rows), len(rows)-len(random_rows)),
            boundary_trajectories=[{"focus": focus, "rows": length, "mse": mean(start, length)}
                                   for focus, start, length in boundaries])
        report["solved"] = max(report["random_mse"], report["boundary_mse"]) <= setting["solved_threshold"]
    return report


def case_directory(output, protocol, seed, case):
    return Path(output) / "code" / str(seed) / f"case_{list(protocol['cases']).index(case)}"


def summarize(protocol, output):
    cases = []
    sources = {}
    for seed in protocol["development_apparatus"]:
        for case in protocol["cases"]:
            directory = case_directory(output, protocol, seed, case)
            path = directory / "code_checkpoint.json"
            if path.exists():
                saved = json.loads(path.read_text())
            elif (directory / "prepared.json").exists():
                prepared = json.loads((directory / "prepared.json").read_text())
                saved = {**prepared, "learning_wall_seconds": 0., "session": {
                    "status": "prepared", "results": [], "costs": prepared["agent_state"]["costs"],
                    "active": {"agent_state": prepared["agent_state"]}}}
            else:
                continue
            state = saved["session"]
            sources[seed] = saved["source_costs"]
            terminal = json.loads((directory / "terminal.json").read_text()) if (directory / "terminal.json").exists() else None
            research = state["results"][0] if state["results"] else None
            events = research["events"] if research else state["active"]["agent_state"]["events"]
            usage = [event["usage"] for event in events if "usage" in event]
            calls = state["costs"]["model_calls"]

            def reported(name):
                return (sum(item[name] for item in usage) if len(usage) == calls
                        and all(item.get(name) is not None for item in usage) else None)

            tokens = state["costs"]["reported_tokens"]
            cases.append({"seed": seed, "case": case, "status": state["status"], "terminal": terminal,
                "learning_stopped": bool(state["results"]),
                "research_status": research["status"] if research else None,
                "costs": state["costs"], "queries": saved["setup_costs"]["queries"] + sum(
                    event.get("action") == "query" and bool(event.get("feedback", {}).get("observation_cost")) for event in events),
                "learning_wall_seconds": saved["learning_wall_seconds"],
                "preparation_wall_seconds": saved["preparation_wall_seconds"],
                "provider_usage": {"model_calls": calls, "reported_tokens": tokens,
                    "input_tokens": reported("input_tokens"), "output_tokens": reported("output_tokens"),
                    "cached_input_tokens": reported("cached_input_tokens"),
                    "provider_wall_seconds": reported("wall_time_seconds"),
                    "python_execution_seconds": state["costs"]["python_execution_seconds"],
                    "token_budget_excess": max(0, tokens - protocol["code_config"]["token_budget"]) if tokens is not None else None,
                    "models_reported": sorted({item["model"] for item in usage if item.get("model")})},
                "model_origin": research.get("model_origin") if research else None})
    terminals = [row["terminal"] for row in cases if row["terminal"] is not None]

    def totals(rows, keys):
        return {key: sum(row[key] for row in rows) if all(row.get(key) is not None for row in rows)
                else None for key in keys}

    def outcomes(rows, expected):
        return {"expected_cases": expected, "learning_stopped": sum(row["learning_stopped"] for row in rows),
            "research_completed": sum(row["research_status"] == "completed" for row in rows),
            "evaluated": sum(row["terminal"] is not None for row in rows),
            "solved": sum(bool(row["terminal"] and row["terminal"]["solved"]) for row in rows),
            "fully_solved": len(rows) == expected and all(row["terminal"] and row["terminal"]["solved"] for row in rows)}

    by_apparatus = [{"seed": seed, **outcomes([row for row in cases if row["seed"] == seed], len(protocol["cases"]))}
                    for seed in protocol["development_apparatus"]]
    provider_totals = totals([row["provider_usage"] for row in cases], ("model_calls", "reported_tokens",
        "input_tokens", "output_tokens", "cached_input_tokens", "provider_wall_seconds", "python_execution_seconds"))
    provider_totals.update(models_reported=sorted({name for row in cases for name in row["provider_usage"]["models_reported"]}),
        max_token_budget_excess=max((row["provider_usage"]["token_budget_excess"] or 0 for row in cases), default=0),
        cases_with_unknown_usage=sum(row["provider_usage"]["reported_tokens"] is None for row in cases))
    source_costs = totals(list(sources.values()), (*COSTS, "queries"))
    branch_costs = totals([row["costs"] for row in cases], COSTS)
    branch_costs["queries"] = sum(row["queries"] for row in cases)
    result = {"statistical_unit": "apparatus; three dependent diagnosis cases", "arm": "ordinary_python_code",
        "expected_cases": len(protocol["development_apparatus"]) * len(protocol["cases"]),
        "prepared_cases": len(cases),
        "learning_stopped": sum(row["learning_stopped"] for row in cases),
        "research_completed": sum(row["research_status"] == "completed" for row in cases),
        "evaluated": sum(row["terminal"] is not None for row in cases),
        "solved": sum(bool(row["terminal"] and row["terminal"]["solved"]) for row in cases),
        "source_costs_counted_once": source_costs, "branch_costs": branch_costs,
        "learning_costs": {key: source_costs[key] + branch_costs[key] for key in (*COSTS, "queries")},
        "terminal_costs": totals(terminals, (*COSTS, "queries")), "provider_usage": provider_totals,
        "learning_wall_seconds": sum(row["learning_wall_seconds"] for row in cases),
        "preparation_wall_seconds": sum(row["preparation_wall_seconds"] for row in cases),
        "by_case": {case: outcomes([row for row in cases if row["case"] == case], len(protocol["development_apparatus"]))
                    for case in protocol["cases"]},
        "by_apparatus": by_apparatus, "fully_solved_apparatus": sum(row["fully_solved"] for row in by_apparatus),
        "hegel_reference": str(ROOT / protocol["hegel_results_directory"]),
        "cases": cases}
    save(Path(output) / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "learn", "evaluate", "summarize"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seeds, cases = args.seeds or protocol["development_apparatus"], args.cases or list(protocol["cases"])
    if not set(seeds) <= set(protocol["development_apparatus"]) or not set(cases) <= set(protocol["cases"]):
        parser.error("use the declared apparatus and cases")
    if args.mode == "summarize":
        print(json.dumps(summarize(protocol, args.output), indent=2))
        return
    module, _ = load_runtime(protocol)
    if args.mode == "evaluate":
        for seed in protocol["development_apparatus"]:
            for case in protocol["cases"]:
                path = case_directory(args.output, protocol, seed, case) / "code_checkpoint.json"
                if not path.exists() or not json.loads(path.read_text())["session"]["results"]:
                    raise ValueError("all declared learning cases must stop before terminal evaluation")
    for seed in seeds:
        for case in cases:
            directory = case_directory(args.output, protocol, seed, case)
            preparation = directory / "prepared.json"
            if args.mode == "prepare":
                if preparation.exists():
                    continue
                source = json.loads((ROOT / protocol["source_results_directory"] / str(seed) / "source.json").read_text())
                records = []
                if case == "relation_only":
                    control = json.loads((ROOT / protocol["hegel_results_directory"] / str(seed) / "relation_only.json").read_text())
                    records = [row for row in control["acquired"] if row["purpose"] == "correct_input_control"]
                save(preparation, prepare_case(module, protocol, seed, case, source, correct_input_records=records))
            elif args.mode == "learn":
                learn_case(module, protocol, directory, json.loads(preparation.read_text()))
            elif not (directory / "terminal.json").exists():
                checkpoint = json.loads((directory / "code_checkpoint.json").read_text())
                save(directory / "terminal.json", evaluate_case(module, protocol, checkpoint))


if __name__ == "__main__":
    main()
