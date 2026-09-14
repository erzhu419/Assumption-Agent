"""Run three real CLI processes with new questions and delayed observations."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from .agenda import RESOURCES
from .models import ResearchConfig


def run_agenda_demo(checkpoint, *, seed=71):
    """Exercise persisted mixed research through the public agenda command.

    The first invocation starts an external experiment and uses the waiting
    time for instrument research. A second invocation introduces a cheaper
    ordinary task. The third consumes the actual apparatus result and finishes
    the available work with the same library and four resource accounts.
    """
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.exists():
        raise ValueError("the agenda demonstration requires a new checkpoint")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source = {"kind": "suite", "seed": seed, "partition": "development", "task_index": 0}
    task_config = ResearchConfig(seed=seed, observation_budget=32, compute_budget=6000, max_rounds=2)
    requests = [{
        "limits": {"observation_budget": 400, "compute_budget": 350000,
                   "measurement_budget": 4096, "experiment_budget": 2},
        "questions": [
            {"question_id": "objects", "kind": "structure",
             "source": {"kind": "structure", "objects": 2, "readout": "total",
                        "variant": "exchange", "seed": seed},
             "config": asdict(ResearchConfig(seed=seed, observation_budget=80,
                                            compute_budget=100000, max_rounds=2)),
             "measurement_budget": 2048},
            {"question_id": "instruments", "kind": "measurement",
             "source": {"kind": "measurement", "task_index": 0, "seed": 313, "backend": "local"},
             "config": asdict(ResearchConfig(seed=313, observation_budget=64,
                                            compute_budget=60000, max_rounds=2)),
             "measurement_budget": 512},
            {"question_id": "awaited", "kind": "task", "source": source,
             "config": asdict(task_config)}],
        "experiments": [{"question_id": "awaited", "delay_seconds": 2.,
                         "probe": {"probe_id": "delayed_new_history", "sequence": [
                             {"x": x, "z": -x} for x in (-.8, -.2, .4, .9)]}}]},
        {"questions": [{"question_id": "arrived_later", "kind": "task",
                        "source": {**source, "task_index": 1},
                        "config": asdict(ResearchConfig(seed=seed, observation_budget=16,
                                                       compute_budget=3000, max_rounds=2))}]}]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH", ""))))
    stages = []
    for index in range(3):
        command = [sys.executable, "-m", "hegel_machine", "cognition", "agenda",
                   "--checkpoint", str(checkpoint)]
        if index < 2:
            request_path = checkpoint.with_name(checkpoint.stem + f"_request_{index + 1}.json")
            request_path.write_text(json.dumps(requests[index], indent=2) + "\n", encoding="utf-8")
            command.extend(["--request", str(request_path), "--steps", "1"])
        else:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            result_paths = [Path(ticket["result_path"]) for ticket in saved["feedback"]["tickets"].values()]
            deadline = monotonic() + 15.
            while not all(path.exists() for path in result_paths):
                if monotonic() >= deadline:
                    raise RuntimeError("the apparatus has not delivered its result; the saved agenda can be resumed")
                sleep(.02)
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
        stage = json.loads(completed.stdout)
        stages.append({"process": index + 1, "status": stage["status"],
                       "selected": [visit["question_id"] for visit in stage["visits"]],
                       "concepts": stage["concepts"], "spent": stage["spent"],
                       "pending_experiments": len(stage["pending_experiments"])})
    final = json.loads(checkpoint.read_text(encoding="utf-8"))
    summary = {"kind": "dynamic_agenda_three_process_demo", "checkpoint": str(checkpoint),
        "processes": stages, "status": final["status"], "limits": final["limits"],
        "spent": final["spent"], "remaining": final["remaining"],
        "evaluator_compute_cost": final.get("evaluator_compute_cost", 0),
        "questions": [{"question_id": item["question_id"], "kind": item["kind"],
                       "status": item["status"], "visits": item["visits"]}
                      for item in final["questions"].values()],
        "selection_events": [event for event in final["events"] if event["event"] == "research_selected"],
        "deliveries": [event for event in final["events"] if event["event"] == "evidence_delivered"],
        "actual_visit_costs": {name: sum(visit["costs"][name] for visit in final["visits"])
                               for name in RESOURCES},
        "external_costs": final["feedback"]["costs"],
        "concepts": len(final["knowledge"]["concepts"]),
        "structural_priors": len(final["knowledge"]["structural_priors"])}
    checkpoint.with_name(checkpoint.stem + "_demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
