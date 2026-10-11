"""Execute the model's recorded setpoint using the existing feedback controller."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from scripts import consolidate_response_measurement as study
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from scripts.run_online_inquiry_llm import write_json
from scripts.run_response_family_recovery import declarations


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    source = ROOT / "artifacts/response_stagnation_live_20261009"
    previous = json.loads((source / "task_02/live/result.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    recorded = previous["events"][29]
    prefix = min(recorded["feedback"]["observation_indices"])
    study.CASES = {"task_02": protocol["cases"]["task_02"]["config"]}
    lab = study.TransferSignalLab("task_02")
    agent = DiscoveryWorldAgent(None, output / "agent", arm="hegel",
        limits={**protocol["limits"], "max_model_calls": 0})
    agent.lab = lab
    records = previous["observations"][:prefix]
    replay = replay_prefix(lab, {"records": records, "environment_steps": prefix})
    agent.observations = deepcopy(records)
    agent.costs["environment_steps"] = prefix
    initial = json.loads((source / "initial_bundle.json").read_text())
    code = Path(recipe.__file__).read_text()
    for hypothesis in declarations(initial["files"][study.METHOD_PATH]):
        code = _retain_declaration(code, hypothesis)
    for event in previous["events"][:29]:
        if event.get("arguments", {}).get("operator") == "reference_hypothesis":
            code = _retain_declaration(code, event["arguments"]["hypothesis"])
    agent._execute("write", {"path": study.METHOD_PATH, "code": code}, lab)
    write_json(output / "protocol.json", {"source": str(source), "event_index": 29,
        "prefix_actions": prefix, "recorded_intent": recorded["arguments"]["notes"],
        "experiment": {"number": 4, "setting": 1012}, "limits": agent.limits,
        "scope": "Operator transcribes the already recorded model intent into the new tool. "
                 "No model adoption or new scientific selection is claimed; same actual paid prefix."})
    result = agent._execute("run", {"operator": "reference_response", "path": study.METHOD_PATH,
        "experiment": {"number": 4, "setting": 1012}}, lab)
    agent._save_state()
    write_json(output / "result.json", {"replay": replay, "feedback": result,
        "new_actions": lab.environment_steps - prefix,
        "historical_executed_actions": len(recorded["feedback"]["observation_indices"]),
        "historical_requested_actions": len(recorded["arguments"]["actions"]),
        "observations": agent.observations[prefix:], "costs": agent.result()["costs"]})
    print(json.dumps({"new_actions": lab.environment_steps - prefix, "feedback": result}), flush=True)
