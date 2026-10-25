#!/usr/bin/env python3
"""Post-hoc fresh confirmation of the candidate actually selected in episode 02."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent
from scripts.run_online_inquiry_llm import write_json
from scripts.run_temporal_operator_llm import laboratory


def complete_selected_candidate(comparison, output):
    comparison, output = Path(comparison), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory; preserve the original cohort")
    plan = json.loads((comparison / "protocol.json").read_text(encoding="utf-8"))
    case = next(case for case in plan["cases"] if case["case_id"] == "episode_02")
    directory = comparison / "episode_02" / "method_retained"
    training = json.loads((directory / "training.json").read_text(encoding="utf-8"))["tasks"][0]
    selected = next(event for event in training["events"] if event["action"] == "finish")
    arguments = deepcopy(selected["arguments"])
    model = arguments["model"]
    candidate_bytes = (directory / model).read_bytes()
    memory = json.loads((comparison / "memories/second/method_retained.json").read_text(encoding="utf-8"))
    confirmation_seed = case["task"]["confirmation_seed"] + 400009
    agent = ProcedureConsumerAgent(None, output, arm="hegel", knowledge=memory, source_task=False,
        config=InquiryLLMConfig(**plan["transfer_limits"]), confirmation_seed=confirmation_seed)
    for name in ("scripts/complete_temporal_selected_candidate.py",
                 "src/hegel_machine/cognition/checkpoint_storage.py",
                 "src/hegel_machine/cognition/temporal_paired_method.py"):
        saved = output / "source_at_run" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, saved)
    for path, source in training["outgoing_knowledge"]["files"].items():
        if path.startswith("shared/"):
            agent._save(path, source)
    (output / model).write_bytes(candidate_bytes)
    # The failed finish preceded the later manifest-only editing attempts.
    manifest = arguments.get("manifest", "shared/knowledge.json")
    agent._save(manifest, memory["files"][manifest])
    lab = laboratory(case["task"])
    agent.observations = deepcopy(training["paid_observations"])
    agent.rows = [row for record in agent.observations for row in lab.observation_rows(record)]
    agent._write_evidence()
    historical = deepcopy(training["costs"])
    agent.costs = deepcopy(historical)
    # Resume paid observation indices without reacquiring any previous trajectory.
    lab._laboratory.observation_cost = historical["observation_cost"]
    lab._laboratory.measurement_cost = historical["measurement_cost"]
    feedback = agent._execute("finish", arguments, lab)
    agent.events.append({"step": 0, "action": "finish", "arguments": arguments,
                         "feedback": feedback, "usage": {"backend": "posthoc_completion", "total_tokens": 0}})
    if (output / model).read_bytes() != candidate_bytes:
        raise ValueError("the selected candidate source changed during completion")
    additional = {key: agent.costs[key] - value if value is not None else None
                  for key, value in historical.items()}
    result = agent.result("completed" if agent.model else "confirmation_failed")
    result.update(evidence_kind="posthoc_selected_candidate_completion",
        comparison=str(comparison), original_status=training["status"], selected_step=selected["step"],
        selected_model=model, original_selection_feedback=selected["feedback"],
        confirmation_seed=confirmation_seed, historical_costs=historical, additional_costs=additional,
        candidate_source_unchanged=True,
        interpretation="One researcher-initiated completion of the LLM's recorded candidate selection, "
            "using the repaired finish interface and new fresh trajectories. No LLM call, new candidate "
            "selection or refitting. This does not change the original cohort's 3/4 result and does not "
            "establish autonomous LLM completion.")
    write_json(output / "result.json", result)
    summary = {key: result[key] for key in ("status", "evidence_kind", "comparison", "original_status",
        "selected_step", "selected_model", "confirmation_seed", "historical_costs", "additional_costs",
        "candidate_source_unchanged", "interpretation")}
    summary["fresh_confirmation"] = {key: value for key, value in
        (result.get("fresh_confirmation") or {}).items() if key != "observations"}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(complete_selected_candidate(args.comparison, args.output), indent=2))


if __name__ == "__main__":
    main()
