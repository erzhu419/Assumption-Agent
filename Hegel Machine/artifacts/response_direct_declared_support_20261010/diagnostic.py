"""Replay one recorded direct cubic declaration in two isolated runtimes."""
import json
from copy import deepcopy
from pathlib import Path
import sys
from time import monotonic

arm, output = sys.argv[1], Path(sys.argv[2]).resolve()
runtime = Path(__file__).parent / arm
sys.path[:0] = [str(runtime / "src"), str(runtime / "tests"), str(runtime)]
from scripts import run_response_assignment_comparison as study
from hegel_machine.cognition import world_response_hypotheses as response

protocol = json.loads((output / "protocol.json").read_text())
source = Path(protocol["source"])
directory = output / arm
started = monotonic()
agent, lab, context = study.restore(source, directory, "Replay the recorded direct declaration.",
    config=protocol["case"], base_limits=protocol["base_limits"])
# The restored account is historical; no provider is installed or invoked.
agent.limits["max_model_calls"] = context["costs_before"]["model_calls"]
old_questions = {q["id"]: deepcopy(q) for q in agent.questions if q["status"] == "refuted"}
before = json.loads((agent.workspace / study.matched.recipe.STATE).read_text())

def checkpoint(data):
    row = {"arm": arm, "status": data["status"], "score": data["score"],
        "new_costs": {k: v - context["costs_before"][k] for k, v in data["costs"].items()},
        "observations": len(agent.observations), "runtime_module": response.__file__}
    (directory / "active.json").write_text(json.dumps(row, indent=2) + "\n")

agent._on_checkpoint = checkpoint
checkpoint(agent.result())
execution = agent._execute(protocol["action"], deepcopy(protocol["arguments"]), lab)
agent._save_state()
result = agent.result()
state = json.loads((agent.workspace / study.matched.recipe.STATE).read_text())
source_after = (agent.workspace / study.matched.study.METHOD_PATH).read_text()
(directory / "source_after.py").write_text(source_after)
new_costs = {k: v - context["costs_before"][k] for k, v in result["costs"].items()}
questions = {q["id"]: q for q in agent.questions}
unchanged_old_refutations = all(questions[qid] == q for qid, q in old_questions.items())
result.update(arm=arm, execution=execution, observations=deepcopy(agent.observations),
    prefix_observations=context["prefix_observations"], costs_before=context["costs_before"],
    new_costs=new_costs, reference_state={k: state[k] for k in ("tests", "test") if k in state},
    runtime_module=response.__file__, elapsed_seconds=monotonic()-started,
    retained_old_refuted_answers=len(old_questions), unchanged_old_refutations=unchanged_old_refutations)
(directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
points = study.matched.point_confirmations(result)
fresh_points = [p for p in points if any(e["index"] >= context["prefix_observations"]
    for e in p["actual_positive_evidence"])]
brief = {"arm": arm, "success": lab.success, "execution_status": execution.get("status"),
    "new_actions": len(agent.observations)-context["prefix_observations"],
    "prefix_observations": context["prefix_observations"], "new_costs": new_costs,
    "new_point_confirmations": fresh_points,
    "new_positive_responses": study.physical_responses(result, points),
    "response_research_status": execution.get("response_research", {}).get("status"),
    "reused_paid_family_exclusions": execution.get("reused_paid_family_exclusions"),
    "new_tests": state.get("tests", [])[len(before.get("tests", [])):],
    "pending_test": state.get("test"), "runtime_module": response.__file__,
    "retained_old_refuted_answers": len(old_questions),
    "unchanged_old_refutations": unchanged_old_refutations}
(directory / "brief.json").write_text(json.dumps(brief, indent=2) + "\n")
assert new_costs["model_calls"] == 0, "Diagnostic must not call a model"
assert unchanged_old_refutations, "Repair must preserve paid refuted answers"
assert Path(response.__file__).is_relative_to(runtime), "Wrong runtime imported"
print(json.dumps({k: brief[k] for k in ("arm", "success", "execution_status", "new_actions", "new_costs")}), flush=True)
