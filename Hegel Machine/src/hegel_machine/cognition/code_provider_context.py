"""Shared request compaction for code decision providers."""

from copy import deepcopy
from collections import Counter
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


def _world_answer_observation_preview(row, field_path, full_content_file):
    """Keep the historical response inline and retrieve its full world snapshot."""
    snapshots = {'public_objects', 'public_relations', 'dialog_box'}
    if not snapshots.intersection(row):
        return row
    return {**{key: item for key, item in row.items() if key not in snapshots},
            'full_content_file': full_content_file, 'json_path': field_path}


def _compact_world_execution(value, field_path, full_content_file='research_context.json', *, previews=None):
    """Keep research outcomes inline; refer repeated world trajectories to disk."""
    if previews is None:
        previews = {}
    if isinstance(value, list):
        return [_compact_world_execution(item, [*field_path, index], full_content_file, previews=previews)
                for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == 'answer_observations' and (value.get('binding_path') or len(item) > 12
                                            or len(json.dumps(item)) > 12000):
            # Numerical rows already live in relation_research. Cumulative
            # physical answers also repeat the paid trajectory across questions.
            # A short world experiment can still carry large object snapshots;
            # retain citations and a physical preview rather than every copy.
            result[key] = {'observation_indices': [row['index'] for row in item],
                           'full_content_file': full_content_file, 'json_path': [*field_path, key]}
            if not value.get('binding_path'):
                preview = []
                for index in _preview_indices(len(item)):
                    row = item[index]
                    if previews.get(row['index']) != row:
                        preview.append(_world_answer_observation_preview(
                            row, [*field_path, key, index], full_content_file))
                        previews[row['index']] = row
                result[key]['observations_preview'] = preview
        elif key == 'answer_observations':
            result[key] = [_world_answer_observation_preview(
                row, [*field_path, key, index], full_content_file)
                for index, row in enumerate(item)]
        elif key == 'joint_parameter_search' and len(json.dumps(item)) > 12000:
            # These are solver diagnostics, not hypothesis programs or measured
            # predictions. Repeated refits retain their outcomes without inlining
            # every branch and iteration count in every past research answer.
            result[key] = {'count': len(item),
                           'status_counts': dict(Counter(row['status'] for row in item)),
                           'limited_by_branch_budget_count': sum(
                               bool(row.get('limited_by_branch_budget')) for row in item),
                           'models_retained': sum(row.get('models_retained', 0) for row in item),
                           'full_content_file': full_content_file, 'json_path': [*field_path, key]}
        elif key in ('action_results', 'steps') and isinstance(item, list):
            result[key] = {'count': len(item), 'full_content_file': full_content_file,
                           'json_path': [*field_path, key]}
        elif key == 'reviewed_child_answers' and isinstance(item, dict):
            result[key] = {'question_ids': list(item), 'full_content_file': full_content_file,
                           'json_path': [*field_path, key]}
        elif key == 'last_action' and isinstance(item, dict) and 'observation' in item:
            result[key] = _compact_world_execution(
                {name: detail for name, detail in item.items() if name != 'observation'},
                [*field_path, key], full_content_file, previews=previews)
            result[key]['observation'] = {
                'full_content_file': full_content_file,
                'json_path': [*field_path, key, 'observation']}
        elif key == 'proposal' and len(source := json.dumps(item)) > 12000:
            # Retained program proposals can embed the complete candidate library
            # each round. Use the same preview size as other generated sources.
            result[key] = {'characters': len(source), 'preview': source[:800],
                           'full_content_file': full_content_file, 'json_path': [*field_path, key]}
        else:
            result[key] = _compact_world_execution(item, [*field_path, key], full_content_file, previews=previews)
    return result


def prepare_code_context(workspace, request, *, schema, instructions):
    """Persist the full request and return the compact provider-facing copy."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "research_context.json").write_text(json.dumps(request), encoding="utf-8")
    (workspace / "code_action_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (workspace / "code_agent_instructions.txt").write_text(instructions, encoding="utf-8")
    context = deepcopy(request)
    trajectory_records = any(isinstance(record.get("rows"), list)
                             for record in context.get("laboratory_observations", []))
    # World laboratories supply their own evidence file/schema rather than
    # numerical rows. An absent row preview does not mean no observations exist.
    if "observations" in context or "evidence" not in context:
        observations = context.get("observations", [])
        offset = max(0, len(observations) - 12)
        context["observations"] = [{**(_row_preview(row) if trajectory_records else row),
                                    "row_index": offset + index}
                                   for index, row in enumerate(observations[offset:])]
        context["evidence"] = {"observations_file": "observations.json", "observation_count": len(observations),
                               "preview_start": offset, **context.get("evidence", {})}
    context["evidence"].setdefault("full_context_file", "research_context.json")
    if trajectory_records:
        context["laboratory_observations"] = _compact_trajectory_payload(context["laboratory_observations"])
        context["evidence"].setdefault("laboratory_observations_file", "laboratory_observations.json")
    events = context.get("events", [])
    for index, event in enumerate(events):
        arguments = event.get("arguments", {})
        latest = index == len(events) - 1
        compact_arguments = {key: value for key, value in arguments.items()
                             if key != "sequence" and (key != "code" or latest)}
        if "sequence" in arguments:
            compact_arguments["sequence_rows"] = len(arguments["sequence"])
        if "code" in arguments:
            # Recent-event windows no longer share the executor's global index.
            path = event.get("feedback", {}).get("executed_path") or arguments.get("path")
            if path:
                compact_arguments["path"] = path
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
    if context.get('evidence', {}).get('observations_file') == 'laboratory_observations.json' and not trajectory_records:
        # Cumulative questions cite the same physical records repeatedly. Keep
        # each identical preview once across the request; every citation still
        # points to its complete original array, including unpreviewed records.
        # Replace mirrored bodies before recording physical previews, so a
        # removed duplicate cannot consume the only inline copy of a record.
        if (isinstance(request.get('research', {}).get('research_focus'), dict)
                and request['research']['research_focus'] == request.get('research_focus')):
            context['research']['research_focus'] = {
                'full_content_file': 'research_context.json', 'json_path': ['research_focus']}
        if 'method_decision' in context:
            # The decision view already contains the complete current questions
            # plus their callable actions. Do not repeat the same question body
            # in the state view; ordinary Python still reads the full raw state.
            decision_questions = json.loads(request['files']['method_decision.json']).get('questions', [])
            by_id = {question['id']: (index, question) for index, question in enumerate(decision_questions)}
            # Parent returns repeat the same child bodies on every invocation.
            # Keep their order and identity, referring only exact mirrored data.
            for view, key in (('research', 'questions'), ('research_focus', 'returned_children')):
                for index, question in enumerate(request.get(view, {}).get(key, [])):
                    position, detailed = by_id.get(question['id'], (None, {}))
                    if position is not None and all(detailed.get(name) == value for name, value in question.items()):
                        context[view][key][index] = {
                            'id': question['id'], 'status': question['status'],
                            'full_content_file': 'method_decision.json', 'json_path': ['questions', position]}
            for question_id, answer in request.get('research', {}).get('answers', {}).items():
                position, detailed = by_id.get(question_id, (None, {}))
                if position is not None and all(detailed.get(key) == value for key, value in answer.items()):
                    context['research']['answers'][question_id] = {
                        'status': answer.get('status'), 'full_content_file': 'method_decision.json',
                        'json_path': ['questions', position]}
            trials = request.get('research', {}).get('experiment_trials')
            if trials:
                for view, key in [(context.get('research_focus', {}), 'prior_trials'),
                                  *((question, 'experiment_trials')
                                    for question in context['method_decision'].get('questions', []))]:
                    if isinstance(view, dict) and view.get(key) == trials:
                        view[key] = {'full_content_file': 'research_context.json',
                                     'json_path': ['research', 'experiment_trials']}
        previews = {}
        if 'research' in context:
            if 'method_results' in context['research']:
                outcomes = ('status', 'operator', 'diagnosis', 'reason', 'error',
                            'after_observation_error', 'observation_indices', 'evidence',
                            'question_id', 'selected_probe')
                results = context['research']['method_results']
                start = max(0, len(results) - 12)
                if start:
                    context['research']['method_results_history'] = {
                        'count': len(results), 'preview_start': start,
                        'full_content_file': 'research_context.json',
                        'json_path': ['research', 'method_results']}
                context['research']['method_results'] = [
                    {**{key: item[key] for key in ('step', 'path', 'proposal') if key in item},
                     'feedback': {key: value for key, value in item.get('feedback', {}).items()
                                  if key in outcomes},
                     'full_content_file': 'research_context.json',
                     'json_path': ['research', 'method_results', index]}
                    for index, item in enumerate(results[start:], start=start)]
            context['research'] = _compact_world_execution(context['research'], ['research'], previews=previews)
        context['events'] = _compact_world_execution(context.get('events', []), ['events'], previews=previews)
        for name in ('research_focus', 'llm_inquiry', 'procedure_repair'):
            if name in context:
                context[name] = _compact_world_execution(context[name], [name], previews=previews)
        # This file is a separate complete view, not a member of the raw request.
        if 'method_decision' in context:
            context['method_decision'] = _compact_world_execution(
                context['method_decision'], [], 'method_decision.json', previews=previews)
    return context
