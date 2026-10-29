#!/usr/bin/env python3
"""Fixed historical window development: measured methods versus current defaults.

No task/seed search and no response-prediction claim. The shared starting scene is
paid once; the source run is also the without-source control and is not repeated.
The source method is formed by Common from complete measured pairs, never filled
from the regression's expected answer. Use ``register`` before explicit ``run``.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import monotonic

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.common_object_lab import (
    CommonObjectLab, measured_selector_values, object_records_from_common,
)
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


SETTINGS = dict(objects=4, readout="window_total_event", profile="confirmation",
    apparatus_seed=97031007, name_seed=97032007, wiring_seed=97033007,
    actuator_curvature=.6, relation_radius=.9687842617019095)
COMPUTE, ROWS, CONFIRM_ROWS, MAX_STEPS = 120000, 192, 12, 80
INITIAL_SEED, CONFIRM_SEED = 97034007, 98034010
COSTS = ("compute_cost", "observation_cost", "measurement_cost")


def save(path, payload):
    write_json_checkpoint(Path(path), payload)


def make_agent(path, lab, scene=None):
    Path(path).mkdir(parents=True, exist_ok=False)
    agent = CommonArchiveInquiryAgent(None, path, arm="hegel", source_task=False,
        config=InquiryLLMConfig(observation_budget=ROWS, compute_budget=COMPUTE,
            confirmation_rows=CONFIRM_ROWS, confirmation_mse_threshold=1e-6,
            max_model_calls=0, token_budget=0))
    agent._method_public_spec = lab.public_spec()
    if scene:
        agent.observations = deepcopy(scene["observations"])
        agent.rows = [row for record in agent.observations for row in lab.observation_rows(record)]
        agent.costs.update(scene["costs"])
        lab.rows, lab.queries = len(agent.rows), len(agent.observations)
    return agent


def infer(agent, lab, model=None):
    budget = SearchBudget(max(0, COMPUTE - agent.costs["compute_cost"]))
    records = object_records_from_common(agent.observations)
    result = infer_object_model(lab.laboratory.spec_for(("fine",)), records, budget, prior=model)
    agent.costs["compute_cost"] += budget.spent
    return result


def buy(agent, lab, sequences, *, confirmation=False):
    requested = sum(len(sequence) for sequence in sequences)
    limit = ROWS if confirmation else ROWS - CONFIRM_ROWS
    if agent.costs["observation_cost"] + requested > limit:
        return {"error": "research row allowance exhausted; confirmation reserve retained",
                "requested": requested}
    return agent._execute("query", {"actions": [
        {"kind": "trajectory", "channels": ["fine"], "sequence": list(sequence)}
        for sequence in sequences]}, lab)


def prepare_scene(path):
    """Reuse the original public inducer until its first membership question."""
    with AnonymousObjectLab(**SETTINGS) as raw:
        lab = CommonObjectLab(raw)
        agent = make_agent(path, lab)
        started = monotonic()
        try:
            initial = raw.random_probe(INITIAL_SEED, length=12, name="historical_initial")
            bought = buy(agent, lab, [initial.sequence])
            if bought.get("error"):
                raise RuntimeError(bought["error"])
            model = infer(agent, lab)
            status = "prelude_step_limit"
            for serial in range(MAX_STEPS):
                records = object_records_from_common(agent.observations)
                probe, question = propose_object_probe(raw.spec_for(("fine",)), model, records, serial,
                    remaining_rows=ROWS - CONFIRM_ROWS - agent.costs["observation_cost"])
                if question.get("purpose") == "distinguish_membership_dependency":
                    status = "membership_start_ready"
                    break
                if probe is None or agent.costs["compute_cost"] >= COMPUTE:
                    status = question["purpose"] if probe is None else "compute_exhausted"
                    break
                bought = buy(agent, lab, [probe.sequence])
                if bought.get("error"):
                    status = bought["error"]
                    break
                model = infer(agent, lab, model)
        except Exception as error:
            save(Path(path) / "failure.json", {"error": repr(error), "costs": dict(agent.costs),
                "elapsed_seconds": monotonic() - started,
                "observation_ref": {"file": "laboratory_observations.json"}})
            raise
        scene = {"status": status, "settings": SETTINGS, "observations": agent.observations,
            "model": model, "costs": dict(agent.costs), "initial_seed": INITIAL_SEED,
            "scope": "fixed_exposed_historical_development_start"}
        save(Path(path) / "scene.json", scene)
        return scene


def proposal(agent, lab, model, sources, previous):
    records = object_records_from_common(agent.observations)
    values = {"spec": lab.public_spec(), "model": model, "sources": sources,
              "selected_values": measured_selector_values(model, records)}
    arguments = {"operator": "partial_role_experiment", "records": "common",
                 "allocation": {"compute": COMPUTE - agent.costs["compute_cost"]}}
    for name, value in values.items():
        agent._save(name + ".json", json.dumps(value, separators=(",", ":")))
        arguments[name + "_ref"] = {"file": name + ".json"}
    if previous is not None:
        agent._save("pending_pair.json", json.dumps(previous, separators=(",", ":")))
        arguments["previous_ref"] = {"file": "pending_pair.json"}
    return agent._execute("run", arguments, lab)


def continuation(path, scene, sources=()):
    start = monotonic()
    with AnonymousObjectLab(**SETTINGS) as raw:
        lab = CommonObjectLab(raw)
        agent = make_agent(path, lab, scene)
        model, previous, learned = deepcopy(scene["model"]), None, None
        steps, status = [], "step_limit"
        try:
            for serial in range(MAX_STEPS):
                if agent.costs["compute_cost"] >= COMPUTE:
                    status = "compute_exhausted"
                    break
                records = object_records_from_common(agent.observations)
                probe, question = propose_object_probe(raw.spec_for(("fine",)), model, records, serial,
                    remaining_rows=ROWS - CONFIRM_ROWS - agent.costs["observation_cost"])
                if previous is not None or question.get("purpose") == "distinguish_membership_dependency":
                    result = proposal(agent, lab, model, sources, previous)
                    receipt = f"pair_{serial:03d}.json"
                    save(Path(path) / receipt, result)
                    steps.append({"result_ref": {"file": receipt}, "status": result["status"],
                        "relation": result.get("relation"), "completed_relation": result.get("completed_relation"),
                        "method_actions": result.get("method_actions", []),
                        "parent_complete": bool(model.get("complete"))})
                    if result.get("method_object"):
                        learned = deepcopy(result["method_object"])
                    if model.get("complete"):
                        status = "roles_identified"
                        break
                    if result["status"] != "proposed_complete_partial_role_experiment":
                        status = result.get("reason", result["status"])
                        break
                    previous = result
                    bought = buy(agent, lab, [action["sequence"] for action in result["actions"]])
                else:
                    if probe is None:
                        status = question["purpose"]
                        break
                    bought = buy(agent, lab, [probe.sequence])
                    steps.append({"ordinary_question": question})
                if bought.get("error"):
                    status = bought["error"]
                    break
                model = infer(agent, lab, model)
            confirmation = {"status": "not_run", "response_prediction_confirmation": False}
            if model.get("complete"):
                frozen = deepcopy(model["schema"])
                fresh = raw.random_probe(CONFIRM_SEED, length=CONFIRM_ROWS, name="fresh_role_confirmation")
                measured = buy(agent, lab, [fresh.sequence], confirmation=True)
                if not measured.get("error"):
                    model = infer(agent, lab, model)
                    confirmation.update(status="supported" if model.get("complete")
                        and model.get("schema") == frozen else "changed_or_unresolved",
                        observation_rows=CONFIRM_ROWS, independent_probe_seed=CONFIRM_SEED)
                else:
                    confirmation.update(status="unfunded", reason=measured["error"])
            if learned:
                learned.update(model_id="fixed_window_development_method", source_ref={
                    "file": str(Path(path) / "formed_method.json")})
                save(Path(path) / "formed_method.json", learned)
            save(Path(path) / "parent_model.json", model)
            save(Path(path) / "steps.json", steps)
            # Existing evaluator is only called after the retained parent is frozen.
            from run_anonymous_revisit_study import role_score
            terminal = role_score({"object_research": model}, raw._evaluation_schema())
            summary = {"status": status, "confirmation": confirmation, "terminal_roles": terminal,
                "steps_ref": {"file": "steps.json"}, "parent_ref": {"file": "parent_model.json"},
                "formed_method_ref": {"file": "formed_method.json"} if learned else None,
                "costs_including_shared_start": dict(agent.costs),
                "incremental_costs": {key: agent.costs[key] - scene["costs"][key] for key in COSTS},
                "elapsed_seconds": monotonic() - start, "response_prediction_tested": False,
                "independent_task_transfer_tested": False, "model_calls": agent.costs["model_calls"]}
        except Exception as error:
            save(Path(path) / "failure.json", {"error": repr(error), "costs": dict(agent.costs),
                "elapsed_seconds": monotonic() - start, "steps": steps})
            raise
        save(Path(path) / "summary.json", summary)
        return summary, learned


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("register", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paid-start", type=Path, help="Existing scene.json from this fixed public prelude")
    args = parser.parse_args()
    if args.mode == "register":
        args.output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        frozen = ["scripts/run_partial_role_frontier.py", "scripts/run_anonymous_revisit_study.py",
            *["src/hegel_machine/cognition/" + name + ".py" for name in (
                "object_dependency_methods", "object_induction", "object_interventions",
                "common_object_lab", "common_archive_inquiry", "online_inquiry_llm",
                "anonymous_object_lab", "unified_object_lab", "structure_lab", "models")]]
        for relative in frozen:
            target = args.output / "source_at_run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text_incrementally(target, (root / relative).read_text())
        save(args.output / "registration.json", {"settings": SETTINGS,
            "compute_limit_including_shared_start_per_segment": COMPUTE,
            "row_limit_including_shared_start_per_segment": ROWS, "confirmation_rows_reserved": CONFIRM_ROWS,
            "initial_seed": INITIAL_SEED, "confirmation_seed": CONFIRM_SEED, "max_steps": MAX_STEPS,
            "segments": ["source_also_without_source", "with_frozen_source"],
            "frozen_source_files": frozen,
            "without_source_is_same_actual_source_run": True,
            "scope": "same_exposed_physical_development_scene; no seed search or independent transfer claim",
            "response_prediction_confirmation": "not included",
            "paid_start_ref": str(args.paid_start) if args.paid_start else None})
        return
    registration = json.loads((args.output / "registration.json").read_text())
    paid = registration["paid_start_ref"]
    scene = json.loads(Path(paid).read_text()) if paid else prepare_scene(args.output / "shared_start")
    if scene["settings"] != SETTINGS or scene["status"] != "membership_start_ready":
        raise ValueError("The fixed public starting scene is not ready")
    source, method = continuation(args.output / "source", scene)
    if (not method or not any(row.get("status") == "discriminating" for row in method["dependency_methods"])
            or source["confirmation"]["status"] != "supported"
            or not source["terminal_roles"]["declared_complete"]):
        save(args.output / "summary.json", {"status": "source_complete_research_not_supported",
            "source": source, "without_source_ref": {"file": "source/summary.json"},
            "with_frozen_source": "not_run", "shared_start_costs_once": scene["costs"],
            "new_costs": {key: (0 if paid else scene["costs"][key])
                + source["incremental_costs"][key] for key in COSTS}})
        return
    warm, _ = continuation(args.output / "with_frozen_source", scene, [method])
    models = [json.loads((args.output / name / "parent_model.json").read_text())
              for name in ("source", "with_frozen_source")]
    steps = [json.loads((args.output / name / "steps.json").read_text())
             for name in ("source", "with_frozen_source")]
    first_relations = [next((item["relation"] for item in rows if item.get("relation")), None)
                       for rows in steps]
    actions = [[record["action"] for record in json.loads((args.output / name /
        "laboratory_observations.json").read_text())][len(scene["observations"]):]
        for name in ("source", "with_frozen_source")]
    save(args.output / "summary.json", {"status": "completed_fixed_development",
        "source": source, "without_source_ref": {"file": "source/summary.json"},
        "with_frozen_source": warm, "same_actual_actions": actions[0] == actions[1],
        "first_method_relations": first_relations,
        "first_method_relation_changed": first_relations[0] != first_relations[1],
        "same_final_parent_model": models[0] == models[1],
        "same_final_schema": models[0].get("schema") == models[1].get("schema"),
        "shared_start_costs_once": scene["costs"],
        "new_costs": {key: (0 if paid else scene["costs"][key]) + sum(
            item["incremental_costs"][key] for item in (source, warm)) for key in COSTS},
        "independent_transfer_tested": False, "response_prediction_tested": False})


if __name__ == "__main__":
    main()
