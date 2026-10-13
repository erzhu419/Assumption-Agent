"""Shared request compaction for code decision providers."""

from copy import deepcopy
from collections import Counter
import json
from pathlib import Path


_PROVIDER_CONTEXT_CHARACTERS = 1_000_000


def _saved_context_reference(value, field_path, full_content_file="research_context.json"):
    reference = {"full_content_file": full_content_file, "json_path": field_path}
    if isinstance(value, dict):
        for key in ("step", "action", "operator", "status", "error", "returncode", "timeout",
                    "case", "directory", "written", "costs", "model", "fresh_confirmation",
                    "current_source_coverage", "experimental_design", "full_trace_file"):
            if key in value:
                item = value[key]
                reference[key] = (item if len(json.dumps(item)) <= 12000 else
                    _saved_context_reference(item, [*field_path, key], full_content_file))
        if isinstance(value.get("steps"), list):
            reference["steps"] = [
                {"step": step.get("step"), "path": step.get("path"),
                 "full_content_file": full_content_file, "json_path": [*field_path, "steps", index]}
                for index, step in enumerate(value["steps"])]
        if isinstance(value.get("edit_practice"), dict):
            reference["edit_practice"] = _saved_context_reference(
                value["edit_practice"], [*field_path, "edit_practice"], full_content_file)
    elif isinstance(value, str):
        reference.update(characters=len(value), preview=value[:800], tail=value[-800:])
    elif isinstance(value, list):
        reference["count"] = len(value)
    return reference


def _bound_provider_context(context):
    """Bound the whole serialized request, retaining the newest feedback first.

    Per-forecast previews alone left two 32-step practices at 1.19M characters.
    Older histories remain indexed on disk rather than accumulating inline.
    """
    if len(json.dumps(context)) <= _PROVIDER_CONTEXT_CHARACTERS:
        return context
    events = context.get("events", [])
    sections = [(event, "feedback", ["events", index, "feedback"], "research_context.json")
                for index, event in enumerate(events[:-1]) if "feedback" in event]
    sections.extend((context["files"], name, ["files", name], "research_context.json")
                    for name in context.get("files", {}))
    sections.extend((context["session_knowledge"], name, ["session_knowledge", name],
                     "research_context.json") for name in context.get("session_knowledge", {}))
    if events and "feedback" in events[-1]:
        sections.append((events[-1], "feedback", ["events", len(events) - 1, "feedback"],
                         "research_context.json"))
    # Final whole-request accounting also covers cumulative notes and other
    # supplied sections. Keep the task contract and current account inline.
    sections.extend((context, name, [] if name == "method_decision" else [name],
                     "method_decision.json" if name == "method_decision" else "research_context.json")
                    for name in context if name not in {
                        "task_interface", "capabilities", "limits", "costs", "evidence"})
    for parent, key, path, filename in sections:
        value = parent[key]
        if len(json.dumps(value)) <= 12000:
            continue
        replacement = _saved_context_reference(value, path, filename)
        if len(json.dumps(replacement)) >= len(json.dumps(value)):
            continue
        parent[key] = replacement
        if len(json.dumps(context)) <= _PROVIDER_CONTEXT_CHARACTERS:
            return context
    return context


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


def _compact_practice_feedback(value, field_path):
    """Keep the complete practice chronology without repeated evidence snapshots."""
    if isinstance(value, list):
        return [_compact_practice_feedback(item, [*field_path, index])
                for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    compact = {}
    forecast = "model" in value and ("prediction_inputs" in value or "predictions" in value)
    for key, item in value.items():
        path = [*field_path, key]
        if key in ("usage", "rows_preview", "measurements_preview", "row_indices", "observation_indices"):
            continue
        # Prospective comparisons repeat frozen sources and whole prediction
        # inputs on every invocation. They are already saved in the raw request;
        # retain candidate errors, paid fits and selection decisions inline.
        if forecast and key == "source" and isinstance(item, dict):
            compact[key] = {"model": item.get("model"), "files": list(item.get("files", {})),
                            "full_content_file": "research_context.json", "json_path": path}
        elif forecast and key in ("prediction_inputs", "predictions") and isinstance(item, list):
            compact[key] = {"count": len(item),
                            "full_content_file": "research_context.json", "json_path": path}
            if key == "predictions":
                compact[key]["preview"] = [item[index] for index in _preview_indices(len(item))]
        # These are the actual proposals, including any invalid arguments. Do
        # not summarize them as if they were duplicate observation records.
        elif key in ("action", "arguments", "proposal", "program_action", "next_action"):
            compact[key] = item
        elif key == "research_advice" and isinstance(item, dict):
            advice = {name: detail for name, detail in item.items()
                      if name not in ("decision", "evidence_scope")
                      and not (name == "evaluated_model" and detail == value.get("evaluated_model"))}
            if isinstance(item.get("decision"), dict):
                advice["decision"] = {name: detail for name, detail in item["decision"].items()
                                      if name in ("selected_action", "reason")}
            compact[key] = _compact_practice_feedback(advice, path)
        else:
            compact[key] = _compact_practice_feedback(item, path)
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


def _reference_world_response_answers(value, questions):
    """Keep one complete answer when response execution repeats that question."""
    if isinstance(value, list):
        for item in value:
            _reference_world_response_answers(item, questions)
    elif isinstance(value, dict):
        if value.get('operator') == 'reference_response' and isinstance(value.get('family_research'), dict):
            position, question = questions.get(value.get('question_id'), (None, {}))
            answer = question.get('answer', {})
            if position is not None and isinstance(answer, dict):
                for field in ('family_research', 'conditional_simplification'):
                    if isinstance(value.get(field), dict) and value[field] == answer.get(field):
                        value[field] = {'full_content_file': 'method_decision.json',
                                        'json_path': ['questions', position, 'answer', field]}
        for key, item in value.items():
            if key == 'question' and isinstance(item, dict):
                position, question = questions.get(item.get('id'), (None, {}))
                if position is not None:
                    # A program repair can update the question's wording while
                    # returning the exact same large scientific answer. Share
                    # that answer without hiding the changed question or result.
                    answer = item.get('answer')
                    if (isinstance(answer, dict) and answer == question.get('answer')
                            and len(json.dumps(answer)) > 12000):
                        item['answer'] = {'full_content_file': 'method_decision.json',
                                          'json_path': ['questions', position, 'answer']}
            rows = (item if key == 'response_research_history' and isinstance(item, list)
                    else [item] if key == 'response_research' and isinstance(item, dict) else [])
            for row in rows:
                if not isinstance(row, dict):
                    continue
                position, question = questions.get(row.get('question_id'), (None, {}))
                answer = question.get('answer')
                if position is None or not isinstance(answer, dict):
                    continue
                for field in ('family_research', 'conditional_simplification'):
                    if isinstance(row.get(field), dict) and row[field] == answer.get(field):
                        row[field] = {'full_content_file': 'method_decision.json',
                                      'json_path': ['questions', position, 'answer', field]}
            _reference_world_response_answers(item, questions)


def _preview_past_response_answers(questions):
    """Keep past response decisions inline and their complete snapshots readable."""
    responses = [(index, question['answer']) for index, question in enumerate(questions)
                 if isinstance(question.get('answer'), dict)
                 and isinstance(question['answer'].get('family_research'), dict)]
    prior_families = []
    for index, answer in responses[:-1]:
        path = ['questions', index, 'answer', 'family_research']
        research = answer['family_research']
        summary = {key: research[key] for key in (
            'scope', 'response_count', 'nonlocal_progress', 'no_progress_cycles',
            'selected_role', 'selected_test', 'research_obligation') if key in research}
        if 'families' in research:
            repeated = next((earlier for earlier, families in prior_families
                             if families == research['families']), None)
            if repeated is not None:
                # Later experiments often change the question's response count
                # and obligation while leaving every family verdict identical.
                # Keep those changes inline, sharing only the equal family body.
                summary['families'] = {
                    'count': len(research['families']), 'full_content_file': 'method_decision.json',
                    'json_path': ['questions', repeated, 'answer', 'family_research', 'families']}
            else:
                prior_families.append((index, research['families']))
                summary['families'] = [{**family, **({'basis': {
                    'full_content_file': 'method_decision.json',
                    'json_path': [*path, 'families', position, 'basis']}}
                    if 'basis' in family else {})}
                    for position, family in enumerate(research['families'])]
        summary.update(full_content_file='method_decision.json', json_path=path)
        answer['family_research'] = summary
        conditional = answer.get('conditional_simplification')
        if isinstance(conditional, dict):
            answer['conditional_simplification'] = {
                **{key: conditional[key] for key in ('adopted', 'status', 'assumption')
                   if key in conditional},
                'full_content_file': 'method_decision.json',
                'json_path': ['questions', index, 'answer', 'conditional_simplification']}


def _preview_past_response_comparisons(events):
    """Keep prior author verdicts and new evidence, with full search pools on disk."""
    comparisons = []
    for index, event in enumerate(events):
        feedback = event.get('feedback', {})
        path = ['events', index, 'feedback']
        if (feedback.get('operator') == 'reference_response'
                and isinstance(feedback.get('family_research'), dict)):
            comparisons.append((event, 'feedback', feedback, path))
        elif isinstance(feedback.get('response_comparison'), dict):
            comparisons.append((feedback, 'response_comparison', feedback['response_comparison'],
                                [*path, 'response_comparison']))
        elif isinstance(feedback.get('proposal', {}).get('comparison_result'), dict):
            # Ordinary inquiries receive an author's comparison inside its
            # proposal return. Include it in the same chronological history.
            proposal = feedback['proposal']
            comparisons.append((proposal, 'comparison_result', proposal['comparison_result'],
                                [*path, 'proposal', 'comparison_result']))
    for container, key, comparison, path in comparisons[:-1]:
        if len(json.dumps(comparison)) <= 12000:
            continue
        summary = {key: comparison[key] for key in (
            'operator', 'status', 'comparison_status', 'binding_path', 'question_id', 'response_count',
            'estimators', 'family_research', 'new_responses', 'evidence',
            'experiment_action', 'revision_action', 'premise_design_action') if key in comparison}
        summary.update(full_content_file='research_context.json', json_path=path)
        container[key] = summary


def _condition_research_preview(value, field_path, full_content_file, previews):
    """Keep current condition decisions readable without copying executable ASTs."""
    summary = {key: deepcopy(item) for key, item in value.items()
               if key not in ('inputs', 'candidates')}
    fields = ('id', 'claim', 'origin', 'authored_id', 'status', 'checked_response_count', 'conflict_count',
              'conflicting_responses')
    candidates = value.get('candidates', [])
    candidate_summaries = [{key: deepcopy(candidate[key]) for key in fields if key in candidate}
                          for candidate in candidates]
    summary['candidates'] = candidate_summaries
    if len(json.dumps(candidate_summaries)) > 12000:
        # Growing pools repeat in the author, its return, and the parent view.
        # Show actual early/latest examples of each verdict and proposed outcome,
        # retaining the complete pool through the same saved-request interface.
        pools = previews.setdefault('condition_candidate_pools', [])
        predictions = (value.get('selected_experiment') or {}).get('predictions', {})
        repeated = next((reference for pool, outcomes, reference in pools
                         if pool == candidates and outcomes == predictions), None)
        reference = {'full_content_file': full_content_file, 'json_path': [*field_path, 'candidates']}
        groups = {}
        for index, candidate in enumerate(candidates):
            group = (candidate.get('origin'), candidate.get('status'), predictions.get(candidate.get('id')))
            groups.setdefault(group, []).append(index)
        indices = sorted({index for group in groups.values() for index in (group[0], group[-1])})
        summary['candidates'] = {
            'count': len(candidates),
            'status_counts': dict(Counter(candidate.get('status') for candidate in candidates)),
            **(repeated or reference)}
        if repeated is None:
            pools.append((candidates, predictions, reference))
            summary['candidates'].update(
                preview_selection='first and last per origin, status and selected-experiment outcome',
                preview=[{**candidate_summaries[index], 'full_content_file': full_content_file,
                          'json_path': [*field_path, 'candidates', index]} for index in indices])
    summary.update(candidate_count=len(candidates), full_content_file=full_content_file,
                   json_path=field_path)
    # selected_experiment retains its whole actual forecast table. In particular
    # do not replace the latest decision's predictions with a past-vector link.
    return summary


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
        if key == 'condition_research' and isinstance(item, dict):
            result[key] = _condition_research_preview(item, [*field_path, key], full_content_file, previews)
        elif key == 'condition_extension_context' and isinstance(item, dict):
            # This is the previously attempted input snapshot for dispatch
            # deduplication, not the current author's evidence or assignment.
            result[key] = {'response_count': len(item.get('responses', [])),
                           'controller_path': item.get('controller_path'),
                           'targets': item.get('targets', []),
                           'full_content_file': full_content_file, 'json_path': [*field_path, key]}
        elif (key in ('response_comparison', 'comparison_result') and isinstance(item, dict)
              and len(json.dumps(item)) > 12000):
            comparisons = previews.setdefault('response_comparisons', [])
            repeated = next((reference for comparison, reference in comparisons if comparison == item), None)
            if repeated is None:
                comparisons.append((item, {'full_content_file': full_content_file,
                                           'json_path': [*field_path, key]}))
                result[key] = _compact_world_execution(
                    item, [*field_path, key], full_content_file, previews=previews)
            else:
                # An author return and the parent proposal mirror the same paid
                # comparison. Share that body while retaining its live verdict,
                # evidence update and callable next steps in each decision view.
                fields = ('operator', 'status', 'comparison_status', 'binding_path',
                          'question_id', 'response_count', 'target_number', 'selection',
                          'selected_setting', 'selected_number', 'experiment_action',
                          'revision_action', 'premise_design_action', 'new_responses',
                          'new_observation_indices', 'assumptions', 'family_research')
                result[key] = {**{name: item[name] for name in fields if name in item}, **repeated}
        elif (key == 'proposal' and isinstance(item, dict)
              and isinstance(item.get('condition_research'), dict)):
            # The generic source preview below would hide the actionable new
            # hypotheses and experiment after the first 800 characters.
            result[key] = _compact_world_execution(item, [*field_path, key],
                                                  full_content_file, previews=previews)
        elif key == 'answer_observations' and (value.get('binding_path') or len(item) > 12
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
        elif (key in ('candidates', 'response_candidates') and (
              (key == 'response_candidates' and field_path and field_path[-1] == 'proposal_inputs')
              or (field_path and field_path[-1] in ('response_comparison', 'comparison_result'))
              or (value.get('operator') == 'reference_response'
                  and isinstance(value.get('family_research'), dict)))
              and isinstance(item, list)
              and len(json.dumps(item)) > 12000):
            # Computed response proposals repeat large fitted AST pools in
            # author feedback and the parent's decision view. Keep every
            # candidate's prediction and counterexamples inline, with executable
            # details in the existing complete request rather than repeating it.
            fields = ('id', 'family', 'origin', 'authored_id', 'status', 'target_prediction', 'fit_mse',
                      'reason', 'conflicting_responses')
            pools = previews.setdefault('response_candidate_pools', [])
            repeated = next((reference for pool, reference in pools if pool == item), None)
            reference = {'full_content_file': full_content_file, 'json_path': [*field_path, key]}
            result[key] = {'count': len(item), **(repeated or reference)}
            if repeated is None:
                pools.append((item, reference))
                result[key]['preview'] = []
                conflicts_seen = previews.setdefault('response_candidate_conflicts', [])
                for index, row in enumerate(item):
                    summary = {name: row[name] for name in fields if name in row}
                    conflicts = row.get('conflicting_responses', [])
                    if len(conflicts) > 1:
                        shared = next((ref for prior, ref in conflicts_seen if prior == conflicts), None)
                        if shared is None:
                            conflicts_seen.append((conflicts, {
                                'full_content_file': full_content_file,
                                'json_path': [*field_path, key, index, 'conflicting_responses']}))
                        else:
                            # A new paid comparison can update the member's
                            # fit/count while retaining exactly the old failures.
                            summary['conflicting_responses'] = {'count': len(conflicts), **shared}
                    result[key]['preview'].append(summary)
            else:
                # The same pool can survive a new comparison verdict or be
                # mirrored under differently referenced family answers. Share
                # its exact body independently of those surrounding decisions.
                result[key]['status_counts'] = dict(Counter(row.get('status') for row in item))
        elif (key == 'hypotheses' and (
              (len(field_path) >= 2 and field_path[-2] == 'response_research_history')
              or (value.get('operator') == 'reference_response'
                  and isinstance(value.get('family_research'), dict)))
              and isinstance(item, list) and len(json.dumps(item)) > 12000):
            # Every past response search retains its full fitted member pool.
            # Keep the history's decisions, family constraints and evidence
            # inline; retrieve its large numerical pool from the saved request.
            result[key] = {'count': len(item), 'full_content_file': full_content_file,
                           'json_path': [*field_path, key]}
        elif (key == 'operational_fallback' and value.get('operator') == 'reference_response'
              and isinstance(value.get('family_research'), dict) and isinstance(item, dict)
              and isinstance(item.get('hypotheses'), list)
              and len(json.dumps(item['hypotheses'])) > 12000):
            # Ordinary response research exposes its search pool twice: once
            # beside the candidates and again inside the operational fallback.
            # Preserve the fallback's selected experiment and separation, using
            # the existing complete-pool reference for its executable members.
            result[key] = _compact_world_execution(
                {**item, 'hypotheses': {
                    'count': len(item['hypotheses']), 'full_content_file': full_content_file,
                    'json_path': [*field_path, key, 'hypotheses']}},
                [*field_path, key], full_content_file, previews=previews)
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
        elif (key == 'predictions' and isinstance(item, dict) and len(item) > 12
              and all(isinstance(prediction, bool) for prediction in item.values())):
            # Many proposed settings have exactly the same finite hypothesis
            # outcome vector. Show every unique vector once, keeping each
            # experiment and its complete prediction map separately retrievable.
            vectors = previews.setdefault('response_prediction_vectors', [])
            repeated = next((reference for predictions, reference in vectors if predictions == item), None)
            if repeated is None:
                result[key] = item
                vectors.append((item, {'full_content_file': full_content_file,
                                       'json_path': [*field_path, key]}))
            else:
                result[key] = dict(repeated)
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
        if ((arguments.get("operator") == "practice" and "steps" in feedback)
                or isinstance(feedback.get("edit_practice"), dict)):
            feedback = _compact_practice_feedback(feedback, ["events", index, "feedback"])
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
    archive_paths = set()
    file_names = set(context.get("files", {})) | set(knowledge.get("files", {}))
    for entry in knowledge.get("source_catalog", []):
        # The catalogue already locates each complete study. Repeating one
        # reference per nested snapshot can itself fill the provider context.
        source_paths = {name for name in file_names
                        if name.startswith(entry["directory"] + "/")
                        or name in (entry.get("model"), entry.get("paid_trace"))}
        if source_paths:
            entry["context_file_count"] = len(source_paths)
        archive_paths.update(source_paths)
    if "files" in knowledge:
        knowledge["shared_files"] = [name for name in knowledge["files"] if name not in archive_paths]
        knowledge["files"] = _file_contents_preview(
            {name: code for name, code in knowledge["files"].items()
             if name not in archive_paths and request.get("files", {}).get(name) != code},
            ["session_knowledge", "files"])
    if "files" in context:
        context["files"] = _file_contents_preview(
            {name: code for name, code in context["files"].items() if name not in archive_paths},
            ["files"])
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
            question = request.get('research_focus', {}).get('question')
            if isinstance(question, dict):
                position, detailed = by_id.get(question.get('id'), (None, {}))
                if position is not None and all(detailed.get(name) == value for name, value in question.items()):
                    context['research_focus']['question'] = {
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
            for name in ('events', 'research', 'research_focus', 'llm_inquiry', 'procedure_repair'):
                if name in context:
                    _reference_world_response_answers(context[name], by_id)
            # Returned source proposals mirror the same family answers as the
            # inquiry state. Normalize both before comparing whole bodies, or
            # one raw family versus one reference defeats exact deduplication.
            _reference_world_response_answers(
                context['method_decision'].get('explanation_proposals', []), by_id)
            _preview_past_response_answers(context['method_decision'].get('questions', []))
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
        _preview_past_response_comparisons(context.get('events', []))
        context['events'] = _compact_world_execution(context.get('events', []), ['events'], previews=previews)
        for name in ('research_focus', 'llm_inquiry', 'procedure_repair'):
            if name in context:
                context[name] = _compact_world_execution(context[name], [name], previews=previews)
        # This file is a separate complete view, not a member of the raw request.
        if 'method_decision' in context:
            context['method_decision'] = _compact_world_execution(
                context['method_decision'], [], 'method_decision.json', previews=previews)
    return _bound_provider_context(context)
