#!/usr/bin/env python3
"""Resume ordinary research immediately after two retained authored refutations."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from scripts.run_discoveryworld_prerequisite_check import replay_prefix, run_check, write_json

SOURCE = ROOT / "artifacts/discoveryworld_analysis_repair_20261007"
SEED = ROOT / "artifacts/discoveryworld_counterexample_revision_20261007_seed"
OUTPUT = ROOT / "artifacts/discoveryworld_counterexample_revision_20261007"


def prepare_seed():
    prefix = json.loads((SOURCE / "prefix.json").read_text())
    plan = json.loads((SOURCE / "protocol.json").read_text())
    source_run = SOURCE / prefix["task"]["id"] / "hegel"
    active = json.loads((source_run / "active.json").read_text())
    offset = active["prefix_event_count"]
    repairs = active["events"][offset:offset + 2]
    if (len(prefix["records"]) != 421 or len(repairs) != 2
            or any(event["action"] != "evaluate" for event in repairs)):
        raise ValueError("Expected the retained 421-record prefix and its first two authored evaluations")
    SEED.mkdir(parents=True, exist_ok=False)
    workspace = SEED / prefix["task"]["id"] / "hegel"
    agent_workspace = workspace / "agent"
    shutil.copytree(SOURCE / "prefix_files", agent_workspace)
    prior_costs = deepcopy(prefix["prior_costs"])
    for event in repairs:
        usage = event["usage"]
        prior_costs["model_calls"] = prior_costs.get("model_calls", 0) + usage.get("calls", 1)
        for field in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens"):
            name = "reported_tokens" if field == "total_tokens" else field
            prior_costs[name] = prior_costs.get(name, 0) + usage[field]
        prior_costs["wall_time_seconds"] = (prior_costs.get("wall_time_seconds", 0)
                                               + usage.get("wall_time_seconds", 0))
    provenance = {
        "source_prefix": str(SOURCE / "prefix.json"),
        "source_evaluations": [{"source": str(source_run / "active.json"),
            "event_index": offset + index, "action": event["action"],
            "question_id": event["arguments"]["question_id"],
            "original_usage": deepcopy(event["usage"])} for index, event in enumerate(repairs)],
        "origin": "Replay of the first two existing Luna-authored evaluations from an operator-assigned review.",
        "continuation": "Both arms start after these paid analyses, before any parent decision; no new review assignment.",
        "accounting": "Original two model calls and token usage are added once to inherited prior costs. "
                      "Their original Python timing is unavailable separately. Re-execution is preparation, "
                      "with zero model calls and no new observations; its Python and wall costs are "
                      "recorded separately and inherited as preparation overhead, not continuation work.",
    }
    plan.pop("operator_assignment", None)
    plan.update(study="discoveryworld_retained_counterexample_seed", preparation=provenance,
                purpose="Ordinary parent continuation from two previously authored scoped refutations.")
    write_json(SEED / "protocol.json", plan)
    with DiscoveryWorldLab(prefix["task"], plan["runtime"], agent_workspace) as lab:
        replay = replay_prefix(lab, prefix)
        write_json(workspace / "replay.json", replay)
        agent = DiscoveryWorldAgent(None, agent_workspace, arm="hegel", limits=plan["limits"])
        agent.lab = lab
        agent.observations = deepcopy(prefix["records"])
        agent.questions = deepcopy(prefix["research"]["questions"])
        agent.methods = deepcopy(prefix["research"].get("methods", []))
        agent.relation_research = deepcopy(prefix["research"].get("relation_research", {}))
        agent.research_advice = deepcopy(prefix["research"].get("research_advice"))
        agent.research_focus = deepcopy(prefix["research"].get("research_focus"))
        agent._prerequisite_checked_through = prefix["research"].get("prerequisite_checked_through", -1)
        for name in ("program_executions", "child_answers_returned"):
            setattr(agent, name, prefix["research"].get(name, 0))
        agent.method_results = deepcopy(prefix["method_results"])
        agent.files.update({name: (agent_workspace / name).read_text(encoding="utf-8")
                            for name in prefix["registered_files"]})
        agent.events = deepcopy(prefix["events"])
        agent.costs["environment_steps"] = lab.environment_steps
        (agent_workspace / "research_notes.md").write_text(prefix["notes"], encoding="utf-8")
        for index, event in enumerate(repairs):
            # Run the retained source through the ordinary answer/dependency path.
            # Reproduce request preparation and action acknowledgement; no model
            # is called. The second returned answer remains new to the parent.
            agent._request()
            feedback = agent._execute(event["action"], deepcopy(event["arguments"]), lab)
            expected, actual = event["feedback"]["question"], feedback["question"]
            if any(actual.get(key) != expected.get(key) for key in ("status", "answer", "evidence")):
                raise ValueError("Retained authored analysis no longer reproduces its recorded answer")
            agent.events.append({"action": event["action"], "arguments": deepcopy(event["arguments"]),
                "feedback": feedback, "phase": "preparation_replay",
                "source_event": deepcopy(provenance["source_evaluations"][index])})
        if len(agent.observations) != 421 or lab.environment_steps != prefix["environment_steps"]:
            raise ValueError("Analysis replay unexpectedly acquired observations")
        agent._save_state()
        agent.status = "preparation_complete"
        result = agent.result()
        preparation_costs = deepcopy(result["costs"])
        preparation_costs["environment_steps"] = 0
        result.update(task_id=prefix["task"]["id"], arm="hegel", prior_costs=prior_costs,
                      replay=replay, preparation=provenance, preparation_costs=preparation_costs,
                      prefix_event_count=len(prefix["events"]), new_environment_steps=0)
        write_json(workspace / "active.json", result)
        write_json(workspace / "result.json", result)
    write_json(SEED / "preparation.json", {**provenance, "costs": preparation_costs,
        "program_executions": result["research"]["program_executions"]
                              - prefix["research"].get("program_executions", 0),
        "inherited_prior_costs": prior_costs})
    return workspace


def main():
    source = prepare_seed()
    print(json.dumps(run_check(source, OUTPUT, calls=8, workers=2), indent=2), flush=True)


if __name__ == "__main__":
    main()
