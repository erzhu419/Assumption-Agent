"""Delete a naturally learned source while preserving the adaptive research path."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_current_offline_ablation import protocol, _compact


def compact_visit(entry):
    report = entry["research"]
    return {"task_position": entry["task_position"], "status": report["status"],
        "available_concepts": entry["available_concepts"],
        "used_sources": report.get("used_sources", []),
        "concepts_added": report.get("concepts_added", []),
        "costs": report["costs"], "selected_kind": report["selected_kind"],
        "selected_channels": report["selected_channels"],
        "confirmation_loss": report.get("confirmation_loss"),
        "program": report["program"],
        "transfer": [{key: row[key] for key in
            ("source_ids", "accepted", "status", "validation_loss", "validation_rows")
            if key in row} for row in report.get("transfer", [])]}


def main():
    output = Path(__file__).with_name("results.json")
    if output.exists():
        raise FileExistsError(output)
    started = datetime.now(timezone.utc).isoformat()
    config = ResearchConfig(seed=12901, observation_budget=96, compute_budget=24000,
        max_rounds=1, minimum_improvement=1e-7)
    plan = protocol(12901, config=config, order=(1, 1, 2, 1), measurement_budget=384)
    plan["evaluation"].update(length=32, trajectories=2)
    plan["experiment"] = "learned_source_deletion_same_adaptive_path"
    plan["arms"] = {"full": {}, "delete_source": {}}
    plan["arm_contracts"] = {
        "full": "Resume the naturally learned shared prefix with its knowledge.",
        "delete_source": "Resume the identical prefix after KnowledgeLibrary.without removes "
            "the first task's concepts and their dependent knowledge; use_memory stays True.",
    }
    prefix_session = UnifiedResearchSession(config, tasks=plan["tasks"], measurement_budget=384)
    prefix = prefix_session.run(max_tasks=1)
    source_ids = set(prefix["results"][0]["research"]["concepts_added"])
    if not source_ids:
        raise RuntimeError("The registered shared prefix did not learn a source to delete")
    if prefix["evaluation"]:
        raise RuntimeError("The training prefix unexpectedly contains evaluation")
    sessions, first_visits, initial_knowledge = {}, {}, {}
    for name in plan["arms"]:
        session = UnifiedResearchSession(state=deepcopy(prefix))
        if name == "delete_source":
            session.library = session.library.without(source_ids)
        initial_knowledge[name] = {
            "concept_ids": [item.concept_id for item in session.library.concepts],
            "deleted_ids": list(session.library.deleted_ids),
            "use_memory": session.config.use_memory,
        }
        session.run(max_tasks=1)
        first_visits[name] = deepcopy(session.state["results"][1])
        session.run()
        sessions[name] = session
        print(json.dumps({"arm": name, "training_status": session.state["status"],
            "costs": session.to_dict()["costs"]}), flush=True)

    # Both arms finish all learning before any independent evaluation.
    results, first_successor = {}, {}
    for name, session in sessions.items():
        results[name] = _compact(session.evaluate(**plan["evaluation"]), plan)
        initial = first_visits[name]
        report = initial["research"]
        item = plan["tasks"][1]
        with session.lab_factory(item) as laboratory:
            rows = laboratory.evaluation_rows(report["selected_channels"],
                seed=item["seed"] + plan["evaluation"]["seed_offset"],
                length=32, trajectories=2)
            mse = loss(Program.from_dict(report["program"]), rows,
                       report["fallback"], laboratory.spec.risk_quantile)
        first_successor[name] = {**compact_visit(initial), "mse": mse,
            "evaluation_rows": len(rows), "scored_after_all_training": True}
        for row, entry in zip(results[name]["tasks"], session.state["results"]):
            row["transfer"] = compact_visit(entry)["transfer"]
    full, deleted = results["full"], results["delete_source"]
    differences = [a["mse"] - b["mse"] for a, b in zip(full["tasks"], deleted["tasks"])]
    payload = {"development_only": True, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "protocol": plan,
        "shared_prefix": compact_visit(prefix["results"][0]),
        "deleted_source_ids": sorted(source_ids), "arm_initial_knowledge": initial_knowledge,
        "first_successor_initial_visit": first_successor, "results": results,
        "full_minus_delete_source": {"task_mse": differences,
            "mean_mse": mean(differences),
            "costs": {key: full["costs"][key] - deleted["costs"][key]
                      for key in full["costs"]}},
        "external_model_calls": 0,
        "scope": "One development sequence after a common naturally learned prefix. "
            "Deletion occurs once; the delete-source arm may learn later sources normally. "
            "Both arms retain adaptive research and use_memory=True. This is source ablation, "
            "not an ordinary Code/LLM comparison or evidence of stable long-term gain."}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": {
        name: {key: result[key] for key in ("solved", "mean_mse", "costs")}
        for name, result in results.items()},
        "first_successor": {name: {key: value[key] for key in ("status", "mse", "costs", "used_sources")}
                            for name, value in first_successor.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
