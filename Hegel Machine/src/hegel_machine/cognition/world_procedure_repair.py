"""Route actual world-program failures into the existing local research loop."""

from copy import deepcopy
import json

from .temporal_paired_method import execute_procedure
from .online_inquiry_llm import InquiryLLMAgent
from .world_llm_inquiry import execute_llm_inquiry


def repeated_control_cycle(records):
    """Recognize repeated public navigation/dialogue states, not repeated science."""
    controls = {"MOVE_DIRECTION", "ROTATE_DIRECTION", "TELEPORT_TO_OBJECT",
                "TELEPORT_TO_LOCATION", "TALK"}
    fields = ("agentLocation", "inventoryObjects", "accessibleEnvironmentObjects",
              "nearbyObjects", "dialog_box", "lastActionMessage", "taskProgress")
    for period in (1, 2):
        tail = records[-3 * period:]
        if len(tail) != 3 * period:
            continue
        if any(row["action"].get("action") not in controls
               and "chosen_dialog_option_int" not in row["action"] for row in tail):
            continue
        states = [{"action": row["action"], "feedback": row.get("feedback", {}),
                   "observation": {key: row.get("observation", {}).get(key) for key in fields}}
                  for row in tail]
        if states[:period] == states[period:2 * period] == states[2 * period:]:
            return {"period": period, "observation_indices": [row["index"] for row in tail]}
    return None


def run_world_procedure(agent, laboratory, path, *, max_steps=12, query_arguments=None):
    """Execute, research a concrete failure, then resume the parent in this world."""
    start = len(agent.observations)
    result = execute_procedure(agent, laboratory, path, max_steps=max_steps,
                               query_arguments=query_arguments)
    if (agent.agent_arm != "hegel" or agent.provider is None or agent._inquiry_stack
            or agent._procedure_terminal_status() or agent._procedure_budget_status()
            or agent.costs["model_calls"] >= agent.limits["max_model_calls"]):
        return result
    cycle = repeated_control_cycle(agent.observations[start:])
    if result["status"] != "method_error" and not (
            result["status"] == "method_step_limit" and cycle):
        return result

    arguments = query_arguments or {}
    parent_value = arguments.get("research_question") or arguments.get("question") or {
        "id": "task", "question": "\n".join(row.get("description", "")
            for row in laboratory.public_observation().get("taskProgress", []))}
    parent = agent._question(parent_value)
    parent["parent_program_path"] = path
    question_id = f"procedure_repair_{len(agent.method_results)}"
    original_source = (agent.workspace / path).read_text()
    original_path = f"shared/{question_id}_before.py"
    InquiryLLMAgent._execute(agent, "write", {"path": original_path, "code": original_source}, laboratory)
    trigger = {"kind": "execution_error" if result["status"] == "method_error" else "control_cycle",
               "path": path, "parent_id": parent["id"], "procedure_status": result["status"],
               "query_arguments": deepcopy(arguments), "cycle": cycle,
               "last_steps": deepcopy(result["steps"][-3:]),
               "original_source_path": original_path}
    trigger_file = f"shared/{question_id}_failure.json"
    InquiryLLMAgent._execute(agent, "write", {"path": trigger_file,
        "code": json.dumps(trigger, ensure_ascii=False)}, laboratory)
    question = {"id": question_id, "parent_id": parent["id"], "depends_on": [],
        "question": f"Which execution assumption or prerequisite prevents {path} from progressing, "
            f"and does a tested repair resolve it? Inspect {trigger_file} and the actual public records. "
            "Preserve working phases, diagnose the failure, revise the procedure as needed, and run it "
            "against current evidence. You choose the hypothesis and test; no scientific answer is supplied. "
            "Use an evidence-based answer to return a resolved prerequisite to the parent. "
            "For a code-only repair you may finish locally after editing; the parent will execute that "
            "revision, and the edit itself grants no scientific support. Return after the local repair "
            "so the parent can resume its remaining steps."}
    registered = agent._question(question)
    registered.update(origin="host_procedure_failure", failed_program_path=path,
                      failure_evidence_file=trigger_file, parent_program_path=path)
    agent._save_state()
    outcome = execute_llm_inquiry(agent, laboratory, {"question": question, "max_model_calls": 16})
    repair = {"trigger": trigger, "question_id": question_id,
              "original_source_path": original_path, "outcome": outcome}
    result = {**result, "repair": repair}
    terminal = agent._procedure_terminal_status()
    if terminal:
        result["status"] = terminal
        return result
    changed = (agent.workspace / path).read_text() != original_source
    returned = (outcome["status"] == "answered" and outcome["question"]["status"] == "supported") or (
        outcome["status"] == "inquiry_finished" and changed)
    if returned and not agent._procedure_budget_status():
        # Preserve the original invocation/phase start, parameters and current world.
        continuation = run_world_procedure(agent, laboratory, path, max_steps=max_steps,
                                           query_arguments=query_arguments)
        result.update(status=continuation["status"], continuation=continuation)
    agent._save_state()
    return result
