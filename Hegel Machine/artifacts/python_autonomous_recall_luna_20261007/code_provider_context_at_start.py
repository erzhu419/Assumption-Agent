"""Shared request compaction for code decision providers."""

from copy import deepcopy
import json
from pathlib import Path


def _row_preview(row):
    preview = {key: value for key, value in row.items() if key != "history"}
    if "history" in row:
        preview["history_length"] = len(row["history"])
    return preview


def _preview_indices(count):
    return sorted({index for index in (0, 1, count - 2, count - 1) if 0 <= index < count})


def _file_contents_preview(files, field_path):
    # Generated numerical programs can contain hundreds of thousands of
    # characters. The full request is already saved before this compaction.
    return {name: source if not isinstance(source, str) or len(source) <= 12000 else {
        "characters": len(source), "preview": source[:800],
        "full_content_file": "research_context.json", "json_path": [*field_path, name]}
        for name, source in files.items()}


def _compact_trajectory_action(action):
    if not isinstance(action, dict) or action.get("kind") != "trajectory":
        return action
    sequence = action.get("sequence")
    # Rejected model actions must remain readable alongside their error feedback.
    if not isinstance(sequence, list) or any(not isinstance(point, dict) for point in sequence):
        return action
    return {**{key: value for key, value in action.items() if key != "sequence"},
            "sequence_rows": len(sequence),
            "controls": sorted({name for point in sequence for name in point})}


def _compact_trajectory_payload(value):
    """Summarize trajectory purchases wherever feedback repeats their records."""
    if isinstance(value, list):
        return [_compact_trajectory_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    if not isinstance(value.get("rows"), list):
        return {key: _compact_trajectory_payload(item) for key, item in value.items()}
    rows, measurements = value["rows"], value.get("measurements", [])
    indices = value.get("row_indices", [])
    preview = [{**_row_preview(rows[index]),
                "row_index": indices[index] if indices else index}
               for index in _preview_indices(len(rows))]
    return {**{key: _compact_trajectory_payload(item) for key, item in value.items()
               if key not in ("rows", "measurements", "action")},
            "action": _compact_trajectory_action(value.get("action", {})),
            "row_count": len(rows), "rows_preview": preview,
            "measurement_count": len(measurements),
            "measurements_preview": [_row_preview(measurements[index])
                                     for index in _preview_indices(len(measurements))]}


def _compact_practice_feedback(value):
    """Keep the complete practice chronology without repeated evidence snapshots."""
    if isinstance(value, list):
        return [_compact_practice_feedback(item) for item in value]
    if not isinstance(value, dict):
        return value
    compact = {}
    for key, item in value.items():
        if key in ("usage", "rows_preview", "measurements_preview", "row_indices", "observation_indices"):
            continue
        # These are the actual proposals, including any invalid arguments. Do
        # not summarize them as if they were duplicate observation records.
        if key in ("action", "arguments", "proposal", "program_action", "next_action"):
            compact[key] = item
        elif key == "research_advice" and isinstance(item, dict):
            advice = {name: detail for name, detail in item.items()
                      if name not in ("decision", "evidence_scope")
                      and not (name == "evaluated_model" and detail == value.get("evaluated_model"))}
            if isinstance(item.get("decision"), dict):
                advice["decision"] = {name: detail for name, detail in item["decision"].items()
                                      if name in ("selected_action", "reason")}
            compact[key] = _compact_practice_feedback(advice)
        else:
            compact[key] = _compact_practice_feedback(item)
    if "steps" in value and value.get("directory"):
        compact["full_trace_file"] = str(Path(value["directory"]) / "result.json")
    return compact


def prepare_code_context(workspace, request, *, schema, instructions):
    """Persist the full request and return the compact provider-facing copy."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "research_context.json").write_text(json.dumps(request), encoding="utf-8")
    (workspace / "code_action_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (workspace / "code_agent_instructions.txt").write_text(instructions, encoding="utf-8")
    context = deepcopy(request)
    observations = context.get("observations", [])
    trajectory_records = any(isinstance(record.get("rows"), list)
                             for record in context.get("laboratory_observations", []))
    offset = max(0, len(observations) - 12)
    context["observations"] = [{**(_row_preview(row) if trajectory_records else row),
                                "row_index": offset + index}
                                for index, row in enumerate(observations[offset:])]
    context["evidence"] = {"observations_file": "observations.json", "observation_count": len(observations),
                            "preview_start": offset, "full_context_file": "research_context.json"}
    if trajectory_records:
        context["laboratory_observations"] = _compact_trajectory_payload(context["laboratory_observations"])
        context["evidence"]["laboratory_observations_file"] = "laboratory_observations.json"
    events = context.get("events", [])
    for index, event in enumerate(events):
        arguments = event.get("arguments", {})
        latest = index == len(events) - 1
        compact_arguments = {key: value for key, value in arguments.items()
                             if key != "sequence" and (key != "code" or latest)}
        if "sequence" in arguments:
            compact_arguments["sequence_rows"] = len(arguments["sequence"])
        if "code" in arguments:
            compact_arguments["path"] = arguments.get("path", f"scratch_{index:03d}.py")
        if isinstance(arguments.get("actions"), list):
            compact_arguments["actions"] = [_compact_trajectory_action(action)
                                             for action in arguments["actions"]]
        feedback = _compact_trajectory_payload(event.get("feedback", {}))
        if arguments.get("operator") == "practice" and "steps" in feedback:
            feedback = _compact_practice_feedback(feedback)
        if not latest:
            failed = feedback.get("returncode") not in (None, 0) or feedback.get("error") or feedback.get("timeout")
            # Keep concise fit results available for later research decisions.
            feedback = {key: value for key, value in feedback.items()
                        if failed or key not in ("stdout", "stderr")
                        or (key == "stdout" and len(value) <= 2000)}
        events[index] = {key: value for key, value in event.items()
                         if key not in ("arguments", "feedback", "usage")}
        events[index].update(arguments=compact_arguments, feedback=feedback)
    knowledge = context.get("session_knowledge", {})
    if "files" in knowledge:
        knowledge["shared_files"] = list(knowledge["files"])
        knowledge["files"] = _file_contents_preview(
            {name: code for name, code in knowledge["files"].items()
             if request.get("files", {}).get(name) != code}, ["session_knowledge", "files"])
    if "files" in context:
        context["files"] = _file_contents_preview(context["files"], ["files"])
    if "method_decision.json" in request.get("files", {}):
        context = {"method_decision": json.loads(request["files"]["method_decision.json"]), **context}
        for name in ("method_decision.json", "method_candidates.json", "resume_context.json"):
            if name in context["files"]:
                context["files"][name] = {"full_content_file": name}
    return context
