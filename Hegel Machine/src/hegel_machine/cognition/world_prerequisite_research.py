"""Focus repeated public operation failures into a prerequisite question.

Historical numerical researchers assume their measurement/control interface is
already bound. This adapter identifies where that prerequisite is missing; it
does not supply device semantics, an experiment, or an answer.
"""

from copy import deepcopy
import re


def operation_failures(observations):
    """Group actual unsuccessful operations by the UUID they attempted to use."""
    grouped = {}
    for position, record in enumerate(observations):
        if record.get("feedback", {}).get("success") is not False:
            continue
        action = record.get("action", {})
        view = record.get("observation", record)
        reason = "\n".join([view.get("lastActionMessage", ""),
                            *record.get("feedback", {}).get("errors", [])]).strip()
        text = reason.lower()
        if any(message in text for message in (
                "is not readable", "is not activatable/deactivatable", "is not openable/closeable")):
            kind = "unsupported_action"
        elif any(message in text for message in (
                "couldn't find", "could not find", "cannot access", "can't access", "not accessible")):
            kind = "inaccessible"
        else:
            kind = "operation_failed"
        # The API can reject one USE argument even though it occurs in the
        # submitted JSON. Its error alone does not establish inaccessibility.
        target = action.get("arg2") if "missing argument 2" in text else action.get("arg1")
        if type(target) is not int:
            continue  # Direction/location names are not object UUIDs.
        grouped.setdefault(target, []).append({"index": record.get("index", position),
            "action": deepcopy(action), "reason": reason, "kind": kind})
    return grouped


def _public_object(current_observation, target):
    objects = [*current_observation.get("inventoryObjects", []),
               *current_observation.get("accessibleEnvironmentObjects", [])]
    nearby = current_observation.get("nearbyObjects", {}).get("objects", {})
    for values in nearby.values():
        objects.extend(values)
    return next((deepcopy(item) for item in objects if item.get("uuid") == target), {"uuid": target})


def public_object_context(observation, targets=None):
    """Project visible states and direct public containment relations, not labels.

    A missing object stays missing. The caller interprets names/descriptions;
    neither action success nor these strings are converted to scientific truth.
    """
    objects = {}
    for field in ("inventoryObjects", "accessibleEnvironmentObjects"):
        for item in observation.get(field, []):
            objects.setdefault(item["uuid"], item)
    if targets is None:
        return deepcopy(list(objects.values()))
    for values in observation.get("nearbyObjects", {}).get("objects", {}).values():
        for item in values:
            objects.setdefault(item["uuid"], item)
    targets = set(targets)
    selected = set(targets)
    for uuid, item in objects.items():
        references = {int(value) for value in re.findall(r"\[uuid: (\d+)\]", item.get("description", ""))}
        if uuid in targets:
            selected.update(references)
        if references & targets:
            selected.add(uuid)
    return deepcopy([item for uuid, item in objects.items() if uuid in selected])


def public_object_relations(objects):
    """Expose explicit in/on UUID clauses, retaining their public state text."""
    relations = []
    for item in objects:
        for match in re.finditer(r"\((in|on)\s+.*?\[uuid: (\d+)\]\)",
                                 item.get("description", "")):
            relations.append({"subject": item["uuid"], "relation": match[1],
                              "object": int(match[2]), "text": match[0]})
    return relations


def intervention_history(observations, target, after_index):
    """Keep applied controls and conditions alongside failures for one object."""
    history = []
    previous = None
    dialog_target = None
    for position, record in enumerate(observations):
        index = record.get("index", position)
        view = record.get("observation", record)
        action = record.get("action", {})
        in_dialog = view.get("dialog_box", {}).get("is_in_dialog")
        if in_dialog and record.get("feedback", {}).get("success") is True:
            if action.get("action") == "TALK":
                dialog_target = action.get("arg1")
            elif action.get("action") == "USE":
                dialog_target = action.get("arg2")
        dialog_step = "chosen_dialog_option_int" in action and dialog_target == target
        if not in_dialog:
            dialog_target = None
        if index <= after_index:
            continue
        objects = public_object_context(view, [target])
        if not any(item["uuid"] == target for item in objects):
            previous = None
            continue
        if (objects != previous or target in (action.get("arg1"), action.get("arg2"))
                or dialog_step):
            history.append({"index": index, "action": deepcopy(action),
                "feedback": deepcopy(record.get("feedback", {})),
                "lastActionMessage": view.get("lastActionMessage", ""),
                "public_objects": objects})
        previous = objects
    return history


def public_reference_history(observations):
    """Project paid object states for the author to choose and interpret controls.

    Keep the first and latest interaction for each distinct visible object and
    containment configuration. This is not a complete measurement history, and
    neither action success nor an object's name establishes reference validity.
    """
    configurations = {}
    dialog_target = None
    for position, record in enumerate(observations):
        action = record.get("action", {})
        view = record.get("observation", record)
        in_dialog = view.get("dialog_box", {}).get("is_in_dialog")
        if in_dialog and record.get("feedback", {}).get("success") is True:
            if action.get("action") == "TALK":
                dialog_target = action.get("arg1")
            elif action.get("action") == "USE":
                dialog_target = action.get("arg2")
        targets = {action.get(key) for key in ("arg1", "arg2")
                   if type(action.get(key)) is int}
        dialog_step = "chosen_dialog_option_int" in action and type(dialog_target) is int
        if dialog_step:
            targets.add(dialog_target)
        for target in sorted(targets):
            objects = sorted(public_object_context(view, [target]), key=lambda item: item["uuid"])
            row = {"target": target, "index": record.get("index", position),
                   "action": deepcopy(action), "feedback": deepcopy(record.get("feedback", {})),
                   "lastActionMessage": view.get("lastActionMessage", ""),
                   "public_objects": objects}
            if dialog_step or (in_dialog and target == dialog_target):
                row["dialog_box"] = deepcopy(view.get("dialog_box", {}))
            states = configurations.setdefault(target, [])
            state = next((state for state in states if state["objects"] == objects), None)
            if state is None:
                states.append({"objects": objects, "first": row, "latest": row})
            else:
                state["latest"] = row
        if not in_dialog:
            dialog_target = None
    projected = []
    for states in configurations.values():
        for state in states:
            projected.append(state["first"])
            if state["latest"] is not state["first"]:
                projected.append(state["latest"])
    return sorted(projected, key=lambda item: (item["index"], item["target"]))


def next_prerequisite(observations, questions, current_observation, *, after_index=-1):
    """Return the latest repeated unsupported-operation focus, or None.

    A supported interface answer is retained. Repeated failures after that
    answer instead reopen its parent's application conditions for investigation.
    The model supplies the hypothesis, experiment and interpretation.
    """
    choices = []
    for target, failures in operation_failures(observations).items():
        unsupported = [failure for failure in failures if failure["kind"] == "unsupported_action"]
        question_id = f"operation_{target}"
        answered = next((question for question in questions if question.get("status") == "supported" and (
            question.get("id") == question_id or str(question.get("target")) == str(target))
            and not question["id"].startswith("intervention_")), None)
        if answered:
            reviewed = answered.get("reviewed_through_observation")
            if reviewed is None:
                continue  # No dated answer establishes a subsequent failure.
            followup_id = f"intervention_{target}"
            followup = next((q for q in questions if q["id"] == followup_id), {})
            reviewed = max(reviewed, followup.get("reviewed_through_observation", -1))
            pending = [failure for failure in unsupported if failure["index"] > reviewed]
            if len(pending) >= 2:
                # The global cursor tracks scanning, not scientific analysis.
                choices.append((pending[-1]["index"], target, pending, answered))
            continue
        if len(unsupported) < 2 or unsupported[-1]["index"] <= after_index:
            continue
        choices.append((unsupported[-1]["index"], target, failures, None))
    if not choices:
        return None
    _, target, failures, answered = max(choices, key=lambda choice: choice[0])
    public_object = _public_object(current_observation, target)
    name = public_object.get("name", f"object {target}")
    if answered:
        evidence = [failure["index"] for failure in failures]
        return {"kind": "intervention", "origin": "host_parent_research_followup",
            "question": {"id": f"intervention_{target}",
                "question": f"With the supported interface for {name} (UUID {target}), "
                            "does the current intervention achieve its predicted outcome, and under what conditions?",
                "parent_id": answered.get("parent_id") or "task", "depends_on": [answered["id"]]},
            "target": target, "public_object": public_object,
            "known_operation": deepcopy(answered), "failures": failures, "evidence": evidence,
            "intervention_history": intervention_history(
                observations, target, answered["reviewed_through_observation"]),
            "public_states_at_failure": [{"index": record.get("index", position),
                "object": _public_object(record.get("observation", record), target)}
                for position, record in enumerate(observations) if record.get("index", position) in evidence]}
    question = {"id": f"operation_{target}",
        "question": f"What interaction does {name} (UUID {target}) actually support, and what conditions does it require?",
        "parent_id": "task", "depends_on": [], "origin": "execution_failure", "target": target}
    successful_interactions = []
    for position, record in enumerate(observations):
        action = record.get("action", {})
        view = record.get("observation", record)
        if (action.get("action") in {"TALK", "USE"} and target in (action.get("arg1"), action.get("arg2"))
                and record.get("feedback", {}).get("success") is True
                and view.get("dialog_box", {}).get("is_in_dialog") is True):
            successful_interactions.append({"index": record.get("index", position),
                "action": deepcopy(action), "message": view.get("lastActionMessage", ""),
                "dialog_box": deepcopy(view["dialog_box"])})
    return {"question": question, "target": target, "public_object": public_object,
            "failures": failures, "evidence": [failure["index"] for failure in failures],
            "successful_interactions": successful_interactions}
