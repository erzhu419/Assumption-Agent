#!/usr/bin/env python3
"""Controlled real-laboratory diagnosis after a prescribed parameter probe."""

import argparse
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import run_directed_relation_composition as common


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/instrument_relation_diagnosis_20260914"


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load(protocol, arm):
    source = protocol["baseline_runtime"] if arm == "baseline" else protocol["new_development_runtime"]
    module, path = common.load_runtime({"arms": {arm: {"runtime_source": source}},
        "shared_laboratory_source": protocol["shared_laboratory_source"]}, arm)
    names = {"models": ("Observation", "Probe", "SearchBudget", "ResearchConfig", "Candidate"),
        "language": ("Program",), "memory": ("KnowledgeLibrary",),
        "unified_research": ("UnifiedResearchAgent",), "research_graph": ("ResearchGraph",),
        "competing_object_experiments": ("_forecast", "assess_competing_object_probe"),
        "scale_bindings": ("bind_measurement_arguments",),
        "instrument_models": ("rebind_instrument_model",)}
    for name, attributes in names.items():
        imported = importlib.import_module("hegel_machine.cognition." + name)
        globals().update({attribute: getattr(imported, attribute) for attribute in attributes})
    return module, path


def edge(protocol, name, focus=None, *, heldout=False):
    setting = protocol["first_counterexample_probe"]
    quantity = [.27, .81, .49, 0.] if heldout else setting["q"]
    points = []
    for phase in setting["phases"]:
        point = {f"{role}_o{i}": value for role, values in
            (("q", quantity), ("u", setting["u"]), ("x", setting[phase + "_x"]))
            for i, value in enumerate(values)}
        point["focus"] = setting["focus"] if focus is None else focus
        points.append(point)
    return Probe(name, tuple(points))


def paired(probe, batch):
    history, result = [], []
    for point, measured in zip(probe.sequence, batch.observations):
        command = Observation(dict(point), measured.target, tuple(history))
        result.append((command, measured))
        history.append({**point, "__target__": measured.target})
    return result


def laboratory(module, protocol, seed, condition):
    return module.laboratory_for({"environment": "objects", "seed": seed,
        "apparatus_seed": seed, "objects": protocol["objects"],
        **protocol["apparatus_settings"], "relation_condition": condition})


def source_law(spec, coupling):
    fields = json.loads(spec.context["object_schema"])["fields"].values()
    fields = list(fields)
    f = lambda name: Program("feature", name=name)
    c = lambda value: Program("const", value=value)
    predicate = Program("lt", children=(Program("arg", name="edge_source_quantity"),
                                         Program("arg", name="edge_target_quantity")))
    relation = Program("relation_sum", name="right", children=(c(.8), predicate,
        *(f(item[role]) for item in fields for role in ("f", "x", "q"))))
    flow = Program("mul", children=(c(coupling), relation))
    total = Program("object_sum", children=tuple(f(item[role]) for item in fields for role in ("q", "u")))
    response = Program("add", children=(total, Program("mul", children=(c(.7),
        Program("event_delta", lag=1, children=(flow,))))))
    return Program("window_sum", lag=3, children=(response,))


def radius_variant(program, radius):
    children = tuple(radius_variant(child, radius) for child in program.children)
    if program.op == "relation_sum":
        children = (Program("const", value=radius), *children[1:])
    definition = program.definition
    if definition:
        definition = replace(definition, body=radius_variant(definition.body, radius))
    return replace(program, children=children, definition=definition)


def agent_for(protocol, seed, library=None):
    config = {k: v for k, v in protocol["budgets_per_case"].items() if k != "measurement_budget"}
    agent = UnifiedResearchAgent(ResearchConfig(**config, seed=seed, active_probes=True),
        measurement_budget=protocol["budgets_per_case"]["measurement_budget"], library=library)
    agent._learn_instruments = agent._revise_inputs = True
    return agent


def make_source(module, protocol, seed):
    with laboratory(module, protocol, seed, "source_less") as lab:
        agent, budget, actions = agent_for(protocol, seed), SearchBudget(480000), []
        batches, records = [], []
        probes = (lab.random_probe(183, length=48), edge(protocol, "source_active", focus=1.),
                  lab.random_probe(885, length=32))
        for probe in probes:
            batch = lab.acquire(probe, ("fine",))
            batches.append(paired(probe, batch))
            records.append({"probe": asdict(probe), "batch": batch.to_dict()})
        # The declared source geometry has two active outgoing edges from o0.
        transition = records[1]["batch"]["transitions"][1]
        q = transition["before"]["measured_q_o0"]
        coupling = (q - transition["after"]["measured_q_o0"]) / (2 * q)
        budget.spend(3)
        spec = lab.spec_for(("fine",))
        agent._fit_instruments(lab.spec, spec, batches, budget, actions)
        program = source_law(spec, coupling)
        rows = [row for batch in batches for _, row in batch]
        budget.spend(len(rows))
        source_loss = sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)
        concept, _ = agent.library.learn(Candidate("source", program, "supplied_source_family_fit"), spec, rows)
        agent.library.meta_priors.append({"source_concept_id": concept, "family_id": "T08",
            "kind": "supplied_source_family_fit", "object_schema": json.loads(spec.context["object_schema"]),
            "source_features": list(spec.feature_names), "control_features": list(lab.spec.feature_names),
            "status": "supported", "counterexamples": [], "uses": 0, "research_program": None,
            "structure": {"fits": [{"radius": .8}], "radius_interval": [.35, 1.2]}})
        return {"knowledge": agent.library.to_dict(), "source_loss": source_loss,
            "measured_program": program.to_dict(), "batches": records,
            "costs": {**lab.costs, "compute_cost": budget.spent}, "actions": actions}


def restore_pairs(source):
    from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch
    return [paired(Probe(record["probe"]["probe_id"], tuple(record["probe"]["sequence"])),
                   ObjectMeasurementBatch.from_dict(record["batch"]))
            for record in source["batches"]]


def refine_intervals(agent, lab, selected, paid, validation, budget, actions, acquire, pairs, arm):
    """Use the same existing interval researcher and paid continuation in both arms."""
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from hegel_machine.cognition.competing_object_experiments import plan_competing_object_probe
    origin = deepcopy(selected)
    research = RelationIntervalResearch(selected["candidate"].program, agent.library.meta_priors, budget,
        structure=selected.get("structure"), research_program=selected.get("research_program"))
    experiments = []
    # This driver also executes the preserved older runtimes, whose interval
    # researcher predates independently retained experimental trajectories.
    arguments = {"experiments": experiments} if "experiments" in inspect.signature(research.alternatives).parameters else {}
    while research.parameters and budget.remaining > (len(paid) + 9) * 10:
        candidates = research.alternatives(paid, budget, tolerance=agent.config.minimum_improvement, **arguments)
        actions.append({"action": "condition_inherited_relation_intervals", "status": research.status,
            "intervals": research.intervals, "candidate_count": len(candidates), "evidence_rows": len(paid)})
        if not candidates:
            revised = agent._research_view(lab.spec, paid, validation, budget, actions, [],
                                          ResearchGraph(), parent=selected)
            if revised:
                selected = agent._best(revised)
            break
        selected = agent._best(candidates)
        research.adopt(selected)
        questions = research.questions(candidates)
        if not questions or all(x["radius_interval"][1] - x["radius_interval"][0] <= 1e-6
                                for x in research.intervals):
            break
        plan = plan_competing_object_probe(lab.spec, candidates, budget, binding={},
            serial=len(actions), length=5, tolerance=agent.config.minimum_improvement,
            methods=agent.library.program_experiment_methods, task_id=f"controlled:{agent.config.seed}",
            questions=questions)
        if plan["probe"] is None:
            break
        actual = acquire((), "distinguish_inherited_relation", probe=plan["probe"])
        if actual is None:
            break
        evidence = assess_competing_object_probe(plan, pairs[()][-1], budget,
                                                 tolerance=agent.config.minimum_improvement)
        actions.append({"action": "test_inherited_relation_predictions", "probe_id": plan["probe"].probe_id,
                        "evidence": evidence})
        feedback = agent.library.program_experiment_methods.observe(plan, evidence,
            task_id=f"controlled:{agent.config.seed}", command_spec=lab.spec, binding={},
            tolerance=agent.config.minimum_improvement)
        if feedback:
            actions.append(feedback)
        paid.extend(actual)
        if evidence["status"] == "distinguished":
            experiments.append(tuple(actual))
        if arm == "revised":
            from hegel_machine.cognition.relation_input_diagnosis import diagnose_relation_inputs
            diagnosis = diagnose_relation_inputs(agent, lab, selected, origin, plan, evidence,
                acquire, lambda channels: pairs.get(tuple(channels), []), budget, actions)
            if diagnosis:
                paid.extend(diagnosis["command_rows"])
                if diagnosis["entry"] is not None:
                    selected, origin = diagnosis["entry"], deepcopy(diagnosis["entry"])
                    research = RelationIntervalResearch(selected["candidate"].program,
                        agent.library.meta_priors, budget, structure=selected.get("structure"),
                        research_program=selected.get("research_program"))
    return selected


def run_case(module, protocol, seed, case, arm, source):
    started = time.perf_counter()
    condition = protocol["cases"][case]["current_relation_condition"]
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(source["knowledge"])))
    agent, budget, actions = agent_for(protocol, seed, library), SearchBudget(480000), []
    pairs = {("fine",): restore_pairs(source), (): []}
    acquired = []
    with laboratory(module, protocol, seed, condition) as lab:
        def acquire(channels, purpose, *, probe=None):
            probe = probe or lab.random_probe(seed + 2309 + len(acquired), length=24)
            rate = sum(c.cost_per_row for c in lab.channels if c.name in channels)
            if (lab.costs["observation_cost"] + probe.cost > protocol["budgets_per_case"]["observation_budget"] or
                    lab.costs["measurement_cost"] + probe.cost * rate > agent.measurement_budget):
                return None
            batch = lab.acquire(probe, channels)
            pairs.setdefault(tuple(channels), []).append(paired(probe, batch))
            acquired.append({"purpose": purpose, "probe": asdict(probe), "batch": batch.to_dict()})
            actions.append({"action": "acquire", "purpose": purpose, "probe_id": probe.probe_id,
                "channels": list(channels), "observation_cost": batch.observation_cost,
                "measurement_cost": batch.measurement_cost})
            return batch.observations

        if case == "relation_only":
            # Correct-input control is established by paid source measurements,
            # before changing the relation and before the forecast under test.
            with laboratory(module, protocol, seed, "source_less") as source_lab:
                setup_path = OUTPUT / "common_setup/instrument_condition_probes.py"
                module_spec = importlib.util.spec_from_file_location(
                    "hegel_machine.cognition.study_input_conditions", setup_path)
                setup = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(setup)
                binding = agent._instrument_binding(source_lab.spec_for(("fine",)), source_lab.spec)
                boundary_probe, questions = setup.instrument_condition_probe(binding, source_lab.spec,
                    pairs[("fine",)], budget, reference=edge(protocol, "background", focus=1.).sequence[0], serial=0)
                actions.append({"action": "prepare_correct_input_control", "questions": questions})
                for probe in (boundary_probe, source_lab.random_probe(1885, length=32)):
                    if probe is None:
                        continue
                    batch = source_lab.acquire(probe, ("fine",))
                    pairs[("fine",)].append(paired(probe, batch))
                    acquired.append({"purpose": "correct_input_control", "probe": asdict(probe),
                                     "batch": batch.to_dict()})
                agent._fit_instruments(lab.spec, lab.spec_for(("fine",)), pairs[("fine",)], budget, actions)
        binding = agent._instrument_binding(lab.spec_for(("fine",)), lab.spec)
        call = agent.library.callable_atoms(lab.spec_for(("fine",)), limit=1)[0]
        program = bind_measurement_arguments(call, binding)
        selected = {"candidate": Candidate("retained_source", program, "reuse"), "kind": "retained_source",
            "family_id": "T08", "training_loss": 0., "validation_loss": 0., "research_program": None,
            "structure": {"fits": [{"radius": .8}], "radius_interval": [.35, 1.2]}}
        origin = deepcopy(selected)
        probe = edge(protocol, "first_parameter_counterexample")
        budget.spend(probe.cost * 2)
        hypotheses = [{"hypothesis_id": name, "candidate_id": name,
            "program": candidate.to_dict(), "predictions": _forecast(candidate, probe.sequence)}
            for name, candidate in (("retained_source", program),
                                    ("radius_alternative", radius_variant(program, .7)))]
        plan = {"probe": probe, "predicted_fields": {}, "hypotheses": hypotheses}
        observed = acquire((), "prescribed_parameter_experiment", probe=probe)
        assessment = assess_competing_object_probe(plan, pairs[()][-1], budget,
            tolerance=agent.config.minimum_improvement)
        actions.append({"action": "assess_prescribed_parameter_experiment", "evidence": assessment})
        diagnosed = None
        if arm == "revised":
            from hegel_machine.cognition.relation_input_diagnosis import diagnose_relation_inputs
            diagnosed = diagnose_relation_inputs(agent, lab, selected, origin, plan, assessment,
                acquire, lambda channels: pairs.get(tuple(channels), []), budget, actions)
            if diagnosed and diagnosed["entry"] is not None:
                selected = diagnosed["entry"]
        # Both versions continue through the existing response research method.
        validation = acquire((), "confirm_response_after_parameter_experiment")
        training = list(observed)
        if diagnosed:
            training.extend(diagnosed["command_rows"])
        program = selected["candidate"].program
        budget.spend(len(training) + len(validation))
        losses = [sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)
                  for rows in (training, validation)]
        selected.update(training_loss=losses[0], validation_loss=losses[1])
        if max(losses) > agent.config.minimum_improvement:
            revised = agent._research_view(lab.spec, training, validation, budget, actions, [],
                                          ResearchGraph(), parent=selected)
            if revised:
                selected = agent._best(revised)
        selected = refine_intervals(agent, lab, selected, [*training, *validation], validation,
                                    budget, actions, acquire, pairs, arm)
        branch_observations = sum(x["batch"]["observation_cost"] for x in acquired)
        branch_measurements = sum(x["batch"]["measurement_cost"] for x in acquired)
        return {"seed": seed, "case": case, "arm": arm,
            "diagnosis": diagnosed["status"] if diagnosed else None,
            "program": selected["candidate"].program.to_dict(), "selected_kind": selected["kind"],
            "research_program": selected.get("research_program"), "structure": selected.get("structure"),
            "training_loss": selected["training_loss"], "validation_loss": selected["validation_loss"],
            "initial_assessment": assessment, "actions": actions, "acquired": acquired,
            "knowledge": agent.library.to_dict(), "costs": {"observation_cost": branch_observations,
                "measurement_cost": branch_measurements, "queries": len(acquired), "compute_cost": budget.spent},
            "learning_wall_seconds": time.perf_counter() - started}


def evaluate(module, protocol, result):
    program = Program.from_dict(result["program"])
    with laboratory(module, protocol, result["seed"],
                    protocol["cases"][result["case"]]["current_relation_condition"]) as lab:
        rows = lab.evaluation_rows((), seed=result["seed"] + 1000003, length=64, trajectories=4)
        boundary, boundary_losses = [], []
        for focus in protocol["terminal"]["boundary_focus"]:
            for offset in protocol["terminal"]["boundary_offsets"]:
                acquired = lab.acquire(edge(protocol, "terminal_boundary", focus + offset,
                                            heldout=True), ()).observations
                boundary.extend(acquired)
                boundary_losses.append({"focus": focus + offset, "rows": len(acquired),
                    "mse": sum((program.predict(row) - row.target) ** 2 for row in acquired) / len(acquired)})
        mse = lambda data: sum((program.predict(row) - row.target) ** 2 for row in data) / len(data)
        random_mse, boundary_mse = mse(rows), mse(boundary)
        return {"random_mse": random_mse, "boundary_mse": boundary_mse,
            "boundary_trajectories": boundary_losses,
            "solved": max(random_mse, boundary_mse) <= protocol["terminal"]["solved_threshold"],
            "observation_cost": len(rows) + len(boundary), "measurement_cost": 0,
            "compute_cost": len(rows) + len(boundary)}


def summarize(protocol, directory, baseline_directory=None):
    result = {"scope": protocol["scope"], "statistical_unit": "apparatus; three dependent cases per apparatus",
        "independent_apparatus": len(protocol["development_apparatus"]),
        "formal_evidence": protocol.get("formal_evidence", False), "arms": {},
        "reused_baseline_directory": str(baseline_directory) if baseline_directory else None}
    keys = ("observation_cost", "measurement_cost", "compute_cost", "queries")
    for arm in ("baseline", "revised"):
        records, sources = [], []
        for seed in protocol["development_apparatus"]:
            path = directory / arm / str(seed)
            source_file = path / "source.json"
            if arm == "baseline" and baseline_directory and not source_file.exists():
                source_file = baseline_directory / arm / str(seed) / "source.json"
            if source_file.exists():
                sources.append(json.loads(source_file.read_text()))
            for case in protocol["cases"]:
                file = path / (case + ".json")
                if arm == "baseline" and baseline_directory and not file.exists():
                    file = baseline_directory / arm / str(seed) / (case + ".json")
                if file.exists():
                    records.append(json.loads(file.read_text()))
        costs = lambda rows: {key: sum(row["costs"][key] for row in rows) for key in keys}
        source_costs, branch_costs = costs(sources), costs(records)
        actions = [action for record in records for action in record["actions"]]
        diagnoses = [action for action in actions if action["action"] == "diagnose_relation_experiment_inputs"]
        result["arms"][arm] = {"expected_cases": len(protocol["cases"]) * len(protocol["development_apparatus"]),
            "learned_cases": len(records), "evaluated_cases": sum("terminal" in row for row in records),
            "solved": sum(row.get("terminal", {}).get("solved", False) for row in records),
            "random_solved": sum(row.get("terminal", {}).get("random_mse", float("inf")) <= 1e-6 for row in records),
            "fully_solved_apparatus": sum(all(any(row["seed"] == seed and row["case"] == case
                and row.get("terminal", {}).get("solved", False) for row in records)
                for case in protocol["cases"]) for seed in protocol["development_apparatus"]),
            "source_costs_counted_once": source_costs, "branch_costs": branch_costs,
            "learning_costs": {key: source_costs[key] + branch_costs[key] for key in keys},
            "learning_wall_seconds": sum(row["learning_wall_seconds"] for row in [*sources, *records]),
            "terminal_costs": {key: sum(row.get("terminal", {}).get(key, 0) for row in records)
                               for key in keys if key != "queries"},
            "diagnosis_statuses": dict(Counter(row["status"] for row in diagnoses)),
            "acquisition_purposes": dict(Counter(row["purpose"] for row in actions if row["action"] == "acquire")),
            "cases": [{"seed": row["seed"], "case": row["case"], "diagnosis": row["diagnosis"],
                "selected_kind": row["selected_kind"], "costs": row["costs"],
                "terminal": row.get("terminal"), "program_nodes": common.serialized_program_nodes(row["program"])}
                for row in records]}
    save(directory / "summary.json", result)
    print(json.dumps({arm: {k: row[k] for k in ("learned_cases", "solved", "random_solved", "learning_costs")}
                      for arm, row in result["arms"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learn", "evaluate", "summarize"))
    parser.add_argument("--arm", choices=("baseline", "revised"))
    parser.add_argument("--results-directory", type=Path, default=OUTPUT / "controlled_development")
    parser.add_argument("--cases", nargs="+", choices=("input_only", "relation_only", "joint"))
    parser.add_argument("--source-results-directory", type=Path)
    parser.add_argument("--baseline-results-directory", type=Path)
    parser.add_argument("--protocol", type=Path, default=OUTPUT / "development_protocol.json")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.mode == "summarize":
        summarize(protocol, args.results_directory, args.baseline_results_directory)
        return
    if args.arm is None:
        parser.error("learning and evaluation require an arm")
    seeds = args.seeds or protocol["development_apparatus"]
    if any(seed not in protocol["development_apparatus"] for seed in seeds):
        parser.error("seed must be declared before execution")
    module, source_path = load(protocol, args.arm)
    import hegel_machine
    directory = args.results_directory / args.arm
    save(directory / "runtime_info.json", {"source": str(source_path), "version": hegel_machine.__version__,
        "recorded_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode})
    for seed in seeds:
        path = directory / str(seed)
        source_file = path / "source.json"
        if args.source_results_directory:
            source_file = args.source_results_directory / args.arm / str(seed) / "source.json"
        if args.mode == "learn" and not source_file.exists():
            started = time.perf_counter()
            source = make_source(module, protocol, seed)
            source["learning_wall_seconds"] = time.perf_counter() - started
            save(source_file, source)
        source = json.loads(source_file.read_text())
        for case in args.cases or protocol["cases"]:
            result_file = path / (case + ".json")
            if args.mode == "learn":
                if result_file.exists():
                    continue
                result = run_case(module, protocol, seed, case, args.arm, source)
            else:
                result = json.loads(result_file.read_text())
                if "terminal" in result:
                    continue
                result["terminal"] = evaluate(module, protocol, result)
            save(result_file, result)
            print(json.dumps({"seed": seed, "case": case, "arm": args.arm, "mode": args.mode,
                "diagnosis": result["diagnosis"], "costs": result["costs"],
                "terminal": result.get("terminal")}), flush=True)


if __name__ == "__main__":
    main()
