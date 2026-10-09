"""Operator-authored public control procedure; scientific interpretation is external.

The caller supplies memory.program_arguments = {
    "parameters": {"target": uuid, "perturb_label": text, "restore_label": text},
    "procedure_start": first_observation_index,
} and executes through the ordinary feedback-driven procedure runner. Menu labels
and the reference are chosen by the caller. This program supplies no physical law,
prediction, success classifier, or answer to the research question.
"""


def select_actions(spec, observations, memory):
    arguments = memory.get("program_arguments", {})
    parameters = arguments.get("parameters", {})
    target = parameters.get("target")
    labels = {phase: parameters.get(phase + "_label") for phase in ("perturb", "restore")}
    start = arguments.get("procedure_start")
    if (type(target) is not int or type(start) is not int
            or any(not isinstance(label, str) or not label for label in labels.values())):
        return {"stop": "Supply target, perturb_label, restore_label and procedure_start"}

    records = {row["index"]: row for row in observations if row["index"] >= start}
    completed = {}
    for result in memory.get("program_results", []):
        proposal = result.get("proposal") or {}
        phase = proposal.get("phase")
        if phase not in ("exit", "approach", "baseline", "perturb", "restore"):
            continue
        indices = result.get("feedback", {}).get("observation_indices", [])
        current = [records[index] for index in indices if index in records]
        if not current:
            continue
        if (len(current) != 1 or proposal.get("actions") != [current[0]["action"]]
                or current[0]["feedback"].get("success") is not True
                or current[0]["feedback"].get("errors")):
            return {"stop": "The previous reference phase did not execute successfully",
                    "phase_observations": completed}
        completed[phase] = current[0]["index"]

    if "restore" in completed:
        return {"stop": "Reference baseline, perturbation and restoration actions completed; interpret the actual outcomes",
                "phase_observations": completed}

    dialog = spec["current_observation"].get("dialog_box", {})

    def menu_action(phase, label):
        matches = [int(key) for key, value in dialog.get("dialogOptions", {}).items()
                   if value == label]
        if not dialog.get("is_in_dialog") or len(matches) != 1:
            return {"stop": "No unique live menu option for " + phase + ": " + label,
                    "phase_observations": completed}
        return {"phase": phase, "actions": [{"chosen_dialog_option_int": matches[0]}]}

    if "approach" not in completed:
        if dialog.get("is_in_dialog"):
            return menu_action("exit", "Exit")
        return {"phase": "approach", "actions": [{"action": "TELEPORT_TO_OBJECT", "arg1": target}]}
    if "baseline" not in completed:
        return {"phase": "baseline", "actions": [{"action": "TALK", "arg1": target}]}
    phase = "restore" if "perturb" in completed else "perturb"
    return menu_action(phase, labels[phase])
