"""Shared request compaction for code decision providers."""

from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import re

from .executable_evidence import PAID_RELATION_FIELD_MEANINGS


_PROVIDER_CONTEXT_CHARACTERS = 1_000_000


def _saved_context_reference(value, field_path, full_content_file="research_context.json"):
    reference = {"full_content_file": full_content_file, "json_path": field_path}
    if isinstance(value, dict):
        for key in ("step", "action", "operator", "status", "error", "returncode", "timeout",
                    "case", "directory", "written", "costs", "model", "fresh_confirmation",
                    "current_source_coverage", "experimental_design", "full_trace_file",
                    "research_decision", "remaining_boundary", "boundary_update", "completion_reserve"):
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
    event_offset = context.get('event_history', {}).get('preview_start', 0)
    sections = [(event, "feedback", ["events", event_offset + index, "feedback"], "research_context.json")
                for index, event in enumerate(events[:-1]) if "feedback" in event]
    sections.extend((context["files"], name, ["files", name], "research_context.json")
                    for name in context.get("files", {}))
    sections.extend((context["session_knowledge"], name, ["session_knowledge", name],
                     "research_context.json") for name in context.get("session_knowledge", {}))
    if events and "feedback" in events[-1]:
        sections.append((events[-1], "feedback", ["events", event_offset + len(events) - 1, "feedback"],
                         "research_context.json"))
    # Final whole-request accounting also covers cumulative notes and other
    # supplied sections. Keep callable method choices with the task contract:
    # their cumulative executions have already been compacted separately.
    sections.extend((context, name, [] if name == "method_decision" else [name],
                     "method_decision.json" if name == "method_decision" else "research_context.json")
                    for name in sorted(context, key=lambda name: name in {
                        'online_research', 'parent_research_focus', 'research_dependency_focus',
                        'llm_inquiry', 'events'}) if name not in {
                        "task_interface", "capabilities", "limits", "costs", "evidence",
                        "evidence_notebook", "method_decision"})
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


def _reference_repeated_stdout(context, request):
    """Reference exact old output blocks without repeating them as new text."""
    previous = []
    repeated = {}
    offset = context.get('event_history', {}).get('preview_start', 0)
    shown = {offset + index: event for index, event in enumerate(context.get('events', []))}
    facts = context.get('evidence_notebook', {}).get('facts', {})
    for index, event in enumerate(request.get('events', [])):
        feedback = event.get('feedback', {})
        text = feedback.get('stdout')
        if not isinstance(text, str) or len(text) <= 2000:
            continue
        parts = [text]
        sources = []
        for original, reference in previous:
            replaced = []
            for part in parts:
                if not isinstance(part, str) or original not in part:
                    replaced.append(part)
                    continue
                pieces = part.split(original)
                for position, piece in enumerate(pieces):
                    if position:
                        replaced.append(deepcopy(reference))
                    if piece:
                        replaced.append(piece)
                if reference not in sources:
                    sources.append(reference)
            parts = replaced
        if sources:
            repeated[index] = {'parts': parts,
                'characters': feedback.get('stdout_characters', len(text)),
                'repeated_output_sources': deepcopy(sources)}
            if feedback.get('stdout_file'):
                repeated[index]['full_content_file'] = feedback['stdout_file']
            else:
                repeated[index].update(full_content_file='research_context.json',
                    json_path=['events', index, 'feedback', 'stdout'])
            if index in shown and 'stdout' in shown[index].get('feedback', {}):
                shown[index]['feedback']['stdout'] = repeated[index]
            fact = facts.get(f'analysis:{index}')
            if fact is not None:
                fact['stdout'] = deepcopy(repeated[index])
        if feedback.get('stdout_file'):
            reference = {'full_content_file': feedback['stdout_file'],
                         'source_event_index': index}
            count = feedback.get('stdout_characters', len(text))
            if count > len(text):
                reference['character_range'] = [count - len(text), count]
        else:
            reference = {'full_content_file': 'research_context.json',
                         'json_path': ['events', index, 'feedback', 'stdout'],
                         'source_event_index': index}
        previous.append((text, reference))


def _current_online_material(context, request):
    """Keep active science inline, retaining old scripts as ordinary readable files."""
    if 'online_research' not in context:
        return
    active = {'model_research_inputs.json', 'model_bindings.json'}
    selected = request.get('evaluated_model') or {}
    active.update(selected.get(key) for key in ('model', 'source_model'))
    for event in request.get('events', [])[-6:]:
        args, feedback = event.get('arguments', {}), event.get('feedback', {})
        active.update(args.get(key) for key in ('path', 'model'))
        active.update(feedback.get(key) for key in ('executed_path', 'written'))
    for repair in request.get('research_repairs', []):
        active.add(repair.get('model'))
    for name in context.get('files', {}):
        if name == 'research_notes.md':
            context['files'][name] = {'full_content_file': 'research_context.json',
                                      'json_path': ['files', name]}
        elif name not in active:
            context['files'][name] = {'full_content_file': name}
    old_files = context.get('session_knowledge', {}).get('files', {})
    for name in old_files:
        old_files[name] = {'full_content_file': 'research_context.json',
                           'json_path': ['session_knowledge', 'files', name]}
    events = context.get('events', [])
    # Each decision already carries one current notebook. Repeating the old
    # authored versions beside real tool results made an obsolete row range
    # look like several independent observations in the continuing run.
    def reference_notes(value, path):
        if isinstance(value, list):
            for index, item in enumerate(value):
                reference_notes(item, [*path, index])
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in {'notes', 'research_notes'}:
                    value[key] = {'full_content_file': 'research_context.json',
                                  'json_path': [*path, key]}
                else:
                    reference_notes(item, [*path, key])

    reference_notes(events, ['events'])
    if len(events) > 8:
        context['event_history'] = {'count': len(events), 'preview_start': len(events) - 8,
            'full_content_file': 'research_context.json', 'json_path': ['events']}
        context['events'] = events[-8:]
    online = context['online_research']
    questions = online.get('questions', [])
    by_id = {question['id']: question for question in questions}
    keep = {online.get('active_question'), *(q['id'] for q in questions[-4:])}
    keep.update(item.get('question_id') for item in online.get('pending_analyses', []))
    for returned in online.get('parent_returns', [])[-1:]:
        keep.add(returned['parent_id'])
        keep.update(answer.get('question_id') for answer in returned.get('answers', []))
    pending = list(keep)
    while pending:
        question = by_id.get(pending.pop(), {})
        for identifier in [question.get('parent_id'), *question.get('depends_on', [])]:
            if identifier in by_id and identifier not in keep:
                keep.add(identifier)
                pending.append(identifier)
    online['questions'] = [question if question['id'] in keep else {
        'id': question['id'], 'status': question.get('status'),
        'full_content_file': 'research_context.json',
        'json_path': ['online_research', 'questions', index]}
        for index, question in enumerate(questions)]
    for index, record in enumerate(online.get('recent_paid_records', [])):
        action = record.get('action')
        compact = _compact_trajectory_action(action)
        if compact != action:
            record['action'] = {**compact, 'full_content_file': 'research_context.json',
                'json_path': ['online_research', 'recent_paid_records', index, 'action']}
    if 'evidence_notebook' in context:
        # Runtime facts and the one current interpretation belong together,
        # ahead of archived programs, raw histories and older decisions.
        current = {name: context[name] for name in ('evidence_notebook', 'research_notes')
                   if name in context}
        current.update(context)
        context.clear()
        context.update(current)
    context['context_notice'] = (
        'Only active files and the latest eight actions are inline. All older code, '
        'complete trajectories and stdout remain at their file paths; the full '
        'decision history is in research_context.json. Previews are not full trajectories. '
        'Computed checks, regions and representation witnesses in online_research and '
        'research_repairs and evidence_notebook are execution results; research_notes is '
        'the current authored interpretation. Older notebooks remain in the full decision '
        'history and are not additional observations.')


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


def _compact_numeric_research(value, field_path):
    """Keep numerical research decisions, sharing their archived executable bodies."""
    if isinstance(value, list):
        return [_compact_numeric_research(item, [*field_path, index])
                for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    compact = {}
    for key, item in value.items():
        path = [*field_path, key]
        if key == 'reviewed_child_answers' and len(json.dumps(item)) > 12000:
            compact[key] = _saved_context_reference(item, path)
        elif key == 'child_answers' and isinstance(item, list):
            reviewed = value.get('reviewed_child_answers', [])
            latest = {answer.get('question_id'): index for index, answer in enumerate(item)}
            compact[key] = []
            for index, answer in enumerate(item):
                answer_path = [*path, index]
                if index == len(item) - 1 or (answer not in reviewed
                        and index == latest[answer.get('question_id')]):
                    compact[key].append(_compact_numeric_research(answer, answer_path))
                else:
                    summary = _compact_numeric_research({name: answer[name] for name in (
                        'question_id', 'status', 'authored_status', 'evidence_status',
                        'revision_index', 'scope', 'analysis_scope', 'check_results') if name in answer},
                        answer_path)
                    if isinstance(answer.get('answer'), str):
                        summary['answer'] = answer['answer']
                    compact[key].append({**summary, **_saved_context_reference(answer, answer_path)})
        elif key in ('check_results', 'checks') and isinstance(item, list) and len(json.dumps(item)) > 12000:
            groups = {}
            for index, check in enumerate(item):
                groups.setdefault(check.get('status'), []).append(index)
            indices = sorted({index for group in groups.values() for index in (group[0], group[-1])})
            compact[key] = {**_saved_context_reference(item, path),
                'status_counts': dict(Counter(check.get('status') for check in item)),
                'preview': [_compact_numeric_research(item[index], [*path, index]) for index in indices]}
        elif key in ('program', 'selected_program') and isinstance(item, dict):
            compact[key] = _saved_context_reference(item, path)
        elif key == 'history' and isinstance(item, list) and item:
            compact['history_length'] = len(item)
            compact[key] = _saved_context_reference(item, path)
        elif (isinstance(item, list) and len(json.dumps(item)) > 12000
              and (key.endswith('_evidence') or key in {
                  'evidence', 'acquired_evidence', 'available_evidence', 'unreviewed_evidence',
                  'row_indices', 'bound_rows', 'rows', 'answer_observations', 'witnesses'})):
            compact[key] = {**_saved_context_reference(item, path),
                'preview': [_compact_numeric_research(_row_preview(item[index])
                    if isinstance(item[index], dict) else item[index], [*path, index])
                    for index in _preview_indices(len(item))]}
        elif key in ('executions', 'executed_analyses') and isinstance(item, list) and len(item) > 8:
            compact[key] = {**_saved_context_reference(item, path),
                'preview_start': len(item) - 8,
                'latest': [_compact_numeric_research(entry, [*path, index])
                           for index, entry in enumerate(item[-8:], start=len(item) - 8)]}
        else:
            compact[key] = _compact_numeric_research(item, path)
    return compact


def _current_numeric_research(context, request):
    """Prevent accumulated paid facts from displacing the live parent handoff."""
    notebook = context.get('evidence_notebook', {})
    facts = notebook.get('facts', {})
    if not facts or len(json.dumps(facts)) <= 12000:
        return
    events = request.get('events', [])
    keep = set(range(max(0, len(events) - 8), len(events)))
    cited = [(int(start), int(end or start)) for start, end in re.findall(
        r'\banalysis:(\d+)(?:\s*[-–]\s*(\d+))?', request.get('research_notes', ''))]
    online = request.get('online_research', {})
    returned_questions = set()
    for returned in online.get('parent_returns', []):
        answers = returned.get('answers', [])
        if answers:
            returned_questions.add(answers[-1].get('question_id'))
    focus = request.get('parent_research_focus', {})
    returned_questions.add(focus.get('returned_research', {}).get('question_id'))
    returned_questions.discard(None)
    current_questions = {online.get('active_question'), focus.get('question_id'), focus.get('parent_id')}
    for question in online.get('questions', []):
        if question.get('id') not in current_questions:
            continue
        answers = question.get('child_answers', [])
        if answers:
            returned_questions.add(answers[-1].get('question_id'))
    for name, fact in facts.items():
        path = ['evidence_notebook', 'facts', name]
        if fact.get('kind') == 'executed_structural_research':
            facts[name] = _compact_numeric_research(fact, path)
        elif fact.get('kind') == 'executed_python_analysis':
            index = fact['event_index']
            current = (index in keep or any(start <= index <= end for start, end in cited)
                       or fact.get('research_question_id') in returned_questions)
            if not current:
                for key in ('stdout', 'stderr'):
                    if key in fact:
                        if fact.get(f'{key}_file'):
                            reference = {'full_content_file': fact[f'{key}_file'],
                                'characters': fact.get(f'{key}_characters', len(fact[key]))}
                        else:
                            reference = _saved_context_reference(fact[key], [*path, key])
                        reference.pop('preview', None)
                        reference.pop('tail', None)
                        fact[key] = reference
    for name in ('online_research', 'parent_research_focus', 'research_dependency_focus',
                 'llm_inquiry', 'relation_research', 'research_repairs'):
        if name in context:
            context[name] = _compact_numeric_research(context[name], [name])
    offset = context.get('event_history', {}).get('preview_start', 0)
    context['events'] = [_compact_numeric_research(event, ['events', offset + index])
                         for index, event in enumerate(context.get('events', []))]
    repairs = request.get('research_repairs')
    for index, event in enumerate(context['events'], start=offset):
        if repairs and request['events'][index].get('feedback', {}).get('research_repairs') == repairs:
            event['feedback']['research_repairs'] = _saved_context_reference(repairs, ['research_repairs'])
    context['context_notice'] += (
        ' Large numerical check sets retain verdict counts and actual examples; complete '
        'checks, control histories and executable programs remain at their cited JSON paths.')


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


def _compact_practice_feedback(value, field_path, full_content_file="research_context.json", *, previews=None):
    """Keep the complete practice chronology without repeated evidence snapshots."""
    if previews is None:
        previews = {}
    if isinstance(value, list):
        return [_compact_practice_feedback(item, [*field_path, index], full_content_file, previews=previews)
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
                            "full_content_file": full_content_file, "json_path": path}
        elif forecast and key in ("prediction_inputs", "predictions") and isinstance(item, list):
            compact[key] = {"count": len(item),
                            "full_content_file": full_content_file, "json_path": path}
            if key == "predictions":
                compact[key]["preview"] = [item[index] for index in _preview_indices(len(item))]
        # These are the actual proposals, including any invalid arguments. Do
        # not summarize them as if they were duplicate observation records.
        elif key in ("action", "arguments", "proposal", "program_action", "next_action"):
            compact[key] = item
        elif key == "paid_residuals" and isinstance(item, dict):
            examples = item.get('examples', [])
            compact[key] = {name: detail for name, detail in item.items() if name != 'examples'}
            compact[key]['example_count'] = len(examples)
            prior = previews.setdefault(key, [])
            if item in prior:
                compact[key]['examples'] = {'count': len(examples),
                    'full_content_file': full_content_file, 'json_path': [*path, 'examples']}
            else:
                compact[key]['examples'] = [
                    {**_row_preview(examples[index]), 'full_content_file': full_content_file,
                     'json_path': [*path, 'examples', index]}
                    for index in _preview_indices(len(examples))]
                prior.append(item)
        elif key == "paid_relation_diagnostic" and isinstance(item, dict):
            compact[key] = {name: detail for name, detail in item.items() if name != 'revisions'}
            # Restored paid results predate candidate attribution in this field.
            # Its sibling residuals come from the same score, unlike evaluated_model
            # which may name the retained incumbent. Leave original records intact.
            scored = value.get('paid_residuals') or {}
            if 'model' not in compact[key] and scored.get('model'):
                compact[key]['model'] = scored['model']
            if 'evaluation_scope' not in compact[key] and value.get('evaluation_scope'):
                compact[key]['evaluation_scope'] = value['evaluation_scope']
            compact[key]['field_meanings'] = dict(PAID_RELATION_FIELD_MEANINGS)
            compact[key]['evaluation_feedback'] = {
                'full_content_file': full_content_file, 'json_path': list(field_path)}
            compact[key]['revisions'] = []
            prior = previews.setdefault(key, [])
            repeated = item in prior
            if not repeated:
                prior.append(item)
            for revision_index, revision in enumerate(item.get('revisions', [])):
                examples = revision.get('examples', [])
                preview = {name: detail for name, detail in revision.items() if name != 'examples'}
                preview['example_count'] = len(examples)
                if repeated:
                    preview['examples'] = {'count': len(examples), 'full_content_file': full_content_file,
                        'json_path': [*path, 'revisions', revision_index, 'examples']}
                    compact[key]['revisions'].append(preview)
                    continue
                preview['examples'] = []
                for index in _preview_indices(len(examples)):
                    example = _row_preview(examples[index])
                    if 'historical_check' not in example and 'evidence_status' in example:
                        example['historical_check'] = {'status': example['evidence_status']}
                    location = [*path, 'revisions', revision_index, 'examples', index]
                    if 'terms' in example:
                        example['terms'] = {'count': len(example['terms']),
                            'full_content_file': full_content_file,
                            'json_path': [*location, 'terms']}
                    example.update(full_content_file=full_content_file, json_path=location)
                    preview['examples'].append(example)
                compact[key]['revisions'].append(preview)
        elif key == "native_condition_research" and isinstance(item, dict):
            compact[key] = _saved_context_reference(item, path, full_content_file)
            for name in ('round_index', 'compute_cost', 'finite_prior'):
                if name in item:
                    compact[key][name] = item[name]
            compact[key]['candidates'] = [{name: candidate[name] for name in
                ('model', 'source', 'fit_mse', 'evaluate_action') if name in candidate}
                for candidate in item.get('candidates', [])]
        elif key == "research_advice" and isinstance(item, dict):
            advice = {name: detail for name, detail in item.items()
                      if name not in ("decision", "evidence_scope")
                      and not (name == "evaluated_model" and detail == value.get("evaluated_model"))}
            if isinstance(item.get("decision"), dict):
                advice["decision"] = {name: detail for name, detail in item["decision"].items()
                                      if name in ("selected_action", "reason")}
            compact[key] = _compact_practice_feedback(advice, path, full_content_file, previews=previews)
        else:
            compact[key] = _compact_practice_feedback(item, path, full_content_file, previews=previews)
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


def _method_decision_preview(decision):
    """Retain callable methods when cumulative execution traces exceed the prompt."""
    for section in ('methods', 'research_options'):
        for index, method in enumerate(decision.get(section, [])):
            executions = method.get('current_executions', [])
            if len(json.dumps(executions)) <= 12000:
                continue
            path = [section, index, 'current_executions']
            latest = []
            for offset, execution in enumerate(executions[-2:], start=len(executions) - 2):
                latest.append({key: value if len(json.dumps(value)) <= 12000 else
                    _saved_context_reference(value, [*path, offset, key], 'method_decision.json')
                    for key, value in execution.items()})
            method['current_executions'] = {
                **_saved_context_reference(executions, path, 'method_decision.json'),
                'preview_start': len(executions) - len(latest), 'latest': latest}
    return decision


def load_code_context(workspace):
    """Load a saved request, expanding its repeated inherited file references."""
    context = json.loads((Path(workspace) / "research_context.json").read_text())
    inherited = context.get("session_knowledge", {}).get("files", {})
    for name, value in context.get("files", {}).items():
        if value == {"full_content_file": "research_context.json",
                     "json_path": ["session_knowledge", "files", name]}:
            context["files"][name] = inherited[name]
    return context


def prepare_code_context(workspace, request, *, schema, instructions):
    """Persist the full request and return the compact provider-facing copy."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    inherited = request.get("session_knowledge", {}).get("files", {})
    stored = request
    if inherited and "files" in request:
        stored = {**request, "files": dict(request["files"])}
        for name, value in request["files"].items():
            reference = {"full_content_file": "research_context.json",
                         "json_path": ["session_knowledge", "files", name]}
            if (name in inherited and value == inherited[name]
                    and len(json.dumps(value)) > len(json.dumps(reference))):
                stored["files"][name] = reference
    (workspace / "research_context.json").write_text(json.dumps(stored), encoding="utf-8")
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
        source = arguments.get('code')
        recent_source = (index >= len(events) - 8
                         and isinstance(source, str) and len(source) <= 12000)
        compact_arguments = {key: value for key, value in arguments.items()
                             if key != "sequence" and (key != "code" or latest or recent_source)}
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
        context = {"method_decision": _method_decision_preview(
            json.loads(request["files"]["method_decision.json"])), **context}
        for name in ("method_decision.json", "method_candidates.json", "resume_context.json"):
            if name in context["files"]:
                context["files"][name] = {"full_content_file": name}
    _current_online_material(context, request)
    if trajectory_records:
        _current_numeric_research(context, request)
    _reference_repeated_stdout(context, request)
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
    context = _bound_provider_context(context)

    def point_to_saved_source(value):
        if isinstance(value, dict):
            path = value.get('json_path', [])
            if (value.get('full_content_file') == 'research_context.json'
                    and len(path) == 2 and path[0] == 'files'
                    and isinstance(stored.get('files', {}).get(path[1]), dict)):
                reference = stored['files'][path[1]]
                if reference.get('json_path') == ['session_knowledge', 'files', path[1]]:
                    value['json_path'] = list(reference['json_path'])
            for item in value.values():
                point_to_saved_source(item)
        elif isinstance(value, list):
            for item in value:
                point_to_saved_source(item)

    point_to_saved_source(context)
    return context
