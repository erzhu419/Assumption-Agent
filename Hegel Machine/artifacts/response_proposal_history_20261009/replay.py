"""Replay paid-history comparison of all five recorded model proposals."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import consolidate_response_measurement as study
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from scripts.run_response_family_recovery import declarations
from scripts.run_online_inquiry_llm import write_json
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_reference_hypothesis import _paid_family_check, _retain_declaration


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    source = ROOT / "artifacts/response_stagnation_live_20261009"
    protocol = json.loads((source / "protocol.json").read_text())
    package = json.loads((source / "initial_bundle.json").read_text())
    initial = Path(recipe.__file__).read_text()
    for hypothesis in declarations(package["files"][study.METHOD_PATH]):
        initial = _retain_declaration(initial, hypothesis)
    write_json(output / "protocol.json", {"source": str(source), "model_calls": 0,
        "scope": "All five previously authored proposals, checked against their own pre-submission paid records. "
                 "Replay the submission only when history excludes it; SAT/UNKNOWN comparison does not buy "
                 "new experiments in this diagnostic. This is not an online policy or task-success study."})
    rows = []
    for case, config in protocol["cases"].items():
        previous = source / case / "live"
        recorded = json.loads((previous / "result.json").read_text())
        events = [event for event in recorded["events"]
                  if event.get("arguments", {}).get("operator") == "reference_hypothesis"]
        if not events:
            continue
        failure = next((previous / "agent/shared").glob("reference_extension_*.json"))
        table = json.loads(failure.read_text())["research_table"]
        retained_source = initial
        for event in events:
            hypothesis = event["arguments"]["hypothesis"]
            prefix = min(event["feedback"]["new_observation_indices"])
            records = recorded["observations"][:prefix]
            directory = output / case / hypothesis["id"]
            agent = DiscoveryWorldAgent(None, directory / "agent", arm="hegel",
                limits={**protocol["limits"], "max_model_calls": 0})
            study.CASES = {case: config["config"]}
            lab = study.TransferSignalLab(case)
            agent.lab = lab
            replay = replay_prefix(lab, {"records": records, "environment_steps": prefix})
            agent.observations = deepcopy(records)
            agent.costs["environment_steps"] = prefix
            agent._execute("write", {"path": study.METHOD_PATH, "code": retained_source}, lab)
            state = {"research_table": deepcopy(table), "completed": [], "rejected": {},
                "target_numbers": [row["crystal"] for row in table["targets"]]}
            check = _paid_family_check(agent, lab, state, hypothesis)
            result = None
            if check["status"] == "UNSAT_VERIFIED":
                write_json(agent.workspace / "shared/reference_recalibration.json", state)
                result = agent._execute("run", {"operator": "reference_hypothesis",
                    "path": study.METHOD_PATH, "hypothesis": hypothesis}, lab)
            retained_source = _retain_declaration(retained_source, hypothesis)
            row = {"case": case, "hypothesis": hypothesis, "prefix_actions": prefix,
                "historical_trial_actions": event["feedback"]["new_observation_count"],
                "comparison": check, "submission": result, "replay": replay,
                "new_actions": lab.environment_steps - prefix,
                "research_compute": agent.costs["research_compute_units"], "model_calls": 0}
            write_json(directory / "result.json", row)
            rows.append(row)
            print(case, hypothesis["id"], check["status"], "new_actions", row["new_actions"], flush=True)
    write_json(output / "results.json", rows)
