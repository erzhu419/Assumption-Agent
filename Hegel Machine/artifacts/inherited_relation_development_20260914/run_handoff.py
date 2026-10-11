"""Replay a clean old source prefix, then hand its unresolved relation to consumers."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
STUDY = ROOT / "artifacts/relation_interval_confirmation_20260913"
SEED = 74713144600


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "consume"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    source = args.source or (STUDY / "v0802_runtime/src" if args.mode == "prepare" else ROOT / "src")
    sys.path.insert(0, str(source.resolve()))
    from hegel_machine.cognition.unified_session import UnifiedResearchSession, ResearchConfig, laboratory_for
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss

    started = monotonic()
    if args.mode == "prepare":
        if (OUTPUT / "source_prefix.json").exists():
            raise SystemExit("source prefix already exists; no repeat learning")
        protocol = json.loads((STUDY / "protocol.json").read_text())
        session = UnifiedResearchSession(ResearchConfig(**protocol["hegel_config"], seed=SEED),
            tasks=protocol["tasks_by_seed"][str(SEED)][:4],
            measurement_budget=protocol["measurement_budget"])
        state = session.run(max_tasks=2, on_checkpoint=lambda state: save("source_progress.json", state))
        save("source_prefix.json", state)
        save("source_provenance.json", {"runtime": str(source.resolve()), "seed": SEED,
            "tasks_learned": len(state["results"]), "learning_wall_seconds": monotonic() - started,
            "scope": "Fresh old-version total/event prefix on a previously observed development apparatus"})
        print(json.dumps({"status": "source_prepared", "tasks": len(state["results"])}), flush=True)
        return

    if (OUTPUT / "consumer_checkpoint.json").exists():
        raise SystemExit("consumer result already exists; choose a separate named development branch for another run")
    prefix = json.loads((OUTPUT / "source_prefix.json").read_text())
    session = UnifiedResearchSession(state=prefix)
    for _ in range(2):
        state = session.run(max_tasks=1, on_checkpoint=lambda state: save("consumer_progress.json", state))
        session = UnifiedResearchSession(state=json.loads(json.dumps(state)))
    learning_seconds = monotonic() - started
    save("consumer_checkpoint.json", state)
    reference = json.loads((STUDY / "runs/hegel_v0802" / str(SEED) / "evaluation_report.json").read_text())["session"]
    rows = []
    for position in (2, 3):
        task, result = state["tasks"][position], state["results"][position]["research"]
        program = Program.from_dict(result["program"])
        with laboratory_for(task) as lab:
            observations = lab.evaluation_rows(result["selected_channels"], seed=task["seed"] + 1000003,
                length=64, trajectories=4)
            measured = loss(program, observations, result["fallback"], lab.spec.risk_quantile)
        actions = result["actions"]
        inherited = [action for action in actions if action["action"] == "inherit_relation_intervals"]
        conditioned = [action for action in actions if action["action"] == "condition_inherited_relation_intervals"]
        inherited_probe_ids = {action["probe_id"] for action in actions if action["action"] == "acquire"
                               and action["purpose"] == "distinguish_inherited_relation"}
        selected_parameters = [parameter for step in (result.get("research_program") or {}).get("selection_steps", [])
                               if step["operation"] == "resolve_relation_intervals" for parameter in step["parameters"]]
        committed_intervals = [{**parameter, "radius_interval": interval} for parameter in selected_parameters
                               for interval in parameter["radius_intervals"]]
        saved_intervals = []
        for prior in state["knowledge"]["meta_priors"]:
            if prior["source_concept_id"] not in result["concepts_added"]:
                continue
            for payload in ("research_program", "input_research_program"):
                for step in (prior.get(payload) or {}).get("selection_steps", []):
                    if step["operation"] == "resolve_relation_intervals":
                        saved_intervals.append({"source_concept_id": prior["source_concept_id"],
                            "payload": payload, "parameters": step["parameters"]})
        rows.append({"task_position": position, "readout": task["readout"], "loss": measured,
            "previous_loss": reference["evaluation"][position]["loss"],
            "previous_costs": reference["results"][position]["research"]["costs"],
            "program_changed": result["program"] != reference["results"][position]["research"]["program"],
            "costs": result["costs"], "selected_channels": result["selected_channels"],
            "status": result["status"], "confirmation_loss": result["confirmation_loss"],
            "concepts_added": result["concepts_added"], "used_sources": result["used_sources"],
            "initial_intervals": inherited[0]["intervals"] if inherited else [],
            "final_intervals": committed_intervals or (conditioned[-1]["intervals"] if conditioned else []),
            "saved_intervals": saved_intervals,
            "directed_experiments": sum(action["action"] in (
                "test_competing_program_predictions", "test_inherited_relation_predictions") for action in actions),
            "inherited_relation_probes": [batch["probe"] for batch in result["batches"]
                                          if batch["probe"]["probe_id"] in inherited_probe_ids],
            "relation_actions": [{key: value for key, value in action.items()
                                  if key not in ("program", "plan", "evidence")}
                                 for action in actions if "relation" in action["action"]]})
    save("handoff_summary.json", {"runtime": str(source.resolve()), "seed": SEED,
        "consumer_learning_wall_seconds": learning_seconds,
        "original_definitions_preserved": all(item in state["knowledge"]["definitions"]
            for item in prefix["knowledge"]["definitions"]),
        "source_concepts": [item["concept_id"] for item in prefix["knowledge"]["concepts"]],
        "results": rows})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
