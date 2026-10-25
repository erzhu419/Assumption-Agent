"""Connect retained Episodes and frozen abstractions to Common's Program pool."""

from copy import deepcopy
from dataclasses import replace
import json
from math import isfinite, sqrt
from pathlib import Path
from types import SimpleNamespace

from .abstractions import AbstractTemplate
from .language import Program
from .models import Observation, SearchBudget
from .research_resources import allocate_complete_research
from .checkpoint_storage import _json_values_equal, published_checkpoint_text, write_json_checkpoint
from .research_episodes import (
    extract_episodes, resolve_program_sources, _canonical, _contains_reference, _path, _source_id,
    load_source_values, load_common_sources, _question_versions,
)
from .theory_induction import consolidate_theories
from .question_agenda import (
    effective_question_mission, mission_compute_remaining, debit_mission_compute,
    adopt_research_mission, question_work_basis,
)


LIBRARY_PATH = 'shared/theory_library.json'
INDEX_PATH = 'shared/theory_index.json'
_LIBRARY_MAPS = ('candidates', 'executions', 'material_cursors')
_LIBRARY_LISTS = ('episodes', 'materials', 'conflicts', 'unique_support_sources', 'transfers')
CONSOLIDATION_INTERFACE = """
run {operator:'theory_consolidation',theme:...,source_theory_ids:[...],
 theories:[ordinary PR2 declarations],materials:[{kind:...,source_ref:...}],
 episode_sources:[{archive:...,unit:...,member:...,json_path:['episodes',0]}],
 mission:{purpose:'theory_learning',theory_learning_allocation:{compute:...}},
 allocation:{compute:...,max_groups:...}} extracts retained Episodes and compiles
typed ordered Program families with source reconstruction and counterexample
checks. context_family_ref explicitly declares a structural comparison topic,
not shared physical support. counterexamples is keyed by source theory id; each
item contains source_ref, retained observation, context_ref and tolerance.
common_source_refs addresses original Common research_state/training JSON. Its
source_theory_ids select original local IDs; qualified IDs retain physical source
identity and original evidence addresses while avoiding cross-workspace overwrite.
Unknowns, failed episodes and numerical/text conflicts remain in the library.
Frozen candidates live in shared/theory_library.json; encoding costs include
the complete definition, selector, mappings, parameters and exceptions.
run {operator:'theory_transfer',library_path:'shared/theory_library.json',
 candidate_id:...,theory_id:...,mapping:{role_0:'u'},parameters:[...],
 role_signature:{u:{role:...,unit:...,time:...}},output_unit:...,context_ref:...,
 evidence_checks:[{source_ref:...,observation:...,context_ref:...,tolerance:...}]}
instantiates a frozen candidate and registers it in the existing PR2 pool.
Omit parameters to use the first frozen source's parameters, never target labels.
Checks use retained rows, not new acquisition. A failed target binding remains
refuted; source reconstructions and alternative candidates stay available.
Neither operator calls a model, buys data or grants independent confirmation.
Actual before/after revisions also supply typed Program contexts across unlike
after skeletons. episode_sources.action_cutoff is an inclusive original action
index; later outcomes remain referenced and never score the cutoff Program.
For a program_context_relation, theory_transfer takes component_theory_ids,
instantiates the frozen Definition on the finite typed pool and registers each
valid concrete Program. The relation is not asserted to be a before/after edit.
For a program_revision_relation, parent_theory_id fixes the actual before
Program. Optional fitting {operations:[...],risk_quantile:0.5,max_history:4,
max_structures:8} reuses the old grammar without fitting subprograms to world
labels: only open-hole constant slots are fitted inside the complete after.
The same configuration is theory_target.revision_fitting during automatic recall.
These are fits on declared paid evidence, never independent confirmation.
Paid checks may omit observation when source_ref addresses the retained row and
criterion={kind:'paid_mse',evaluated_rows:N,mse_threshold:tau,criterion_ref:...}.
Rows are resolved only in working memory; the complete checked prefix uses its
original mean squared error threshold. sqrt(N*tau) only triggers sufficient
single-row failure witnesses and does not replace the aggregate criterion.
research_route also recalls frozen objects automatically when theory_target
declares role_signature/output_unit/context_ref, using the remaining task
compute or an explicit allocation. Theory/method missions retain their funded
limits. Recall proposes optional candidates without transferring source support.
Typed families enumerate all finite injective AbstractTemplate mappings, checking
each role/unit/time contract without choosing one mapping as scientifically true.
research_route library_path selects a retained library for automatic recall;
the selected path and versions participate in caching and parent-plan validity.
"""


def _unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(deepcopy(value))
    return result


def _absolute_reference(reference, workspace):
    result = _canonical(reference)
    if result.get('file'):
        path = Path(result['file'])
        if not path.is_absolute() and workspace is None:
            raise ValueError('relative frozen source reference requires its original workspace')
        result['file'] = str((path if path.is_absolute() else Path(workspace) / path).resolve())
    return result


def _reference_workspace(reference, root=None):
    path = Path(reference['file'])
    origin = reference.get('workspace') or (root or {}).get('research_workspace') or (root or {}).get('workspace')
    if origin:
        if not Path(origin).is_absolute():
            raise ValueError('frozen source workspace must be absolute')
        return Path(origin)
    if path.parent.name == 'shared' and path.name in {'theory_library.json', 'theory_index.json'}:
        return path.parent.parent
    if path.name == 'research_state.json':
        return path.parent
    return None


def _anchor_file_references(value, workspace):
    # A frozen library repeats the same source filenames in many checks. Resolve
    # each actual path once during this read, rather than lstat every ancestor
    # again for each occurrence. The returned records remain independent.
    paths = {}

    def anchor(item):
        if isinstance(item, list):
            return [anchor(child) for child in item]
        if not isinstance(item, dict):
            return item if item is None or isinstance(item, (str, int, float, bool)) else deepcopy(item)
        result = {key: anchor(child) for key, child in item.items()}
        if result.get('file') and ('json_path' in result or 'field' in result):
            filename = result['file']
            if filename not in paths:
                paths[filename] = _absolute_reference({'file': filename, 'json_path': []}, workspace)['file']
            result['file'] = paths[filename]
            if 'field' in result:
                result['json_path'] = _path(result)
                result.pop('field')
        return result

    return anchor(value)


def _compact_library_fields(document):
    """Store repeated contracts and generated transfer payloads at one address.

    A reference without file stays within this document, including after a
    freeze or move. Runtime records remain expanded and independently mutable.
    """
    result, seen, seen_scopes, transfers = dict(document), [], [], []
    for index, original in enumerate(document.get('transfers', [])):
        record = dict(original)
        if 'check_basis' in record:
            previous = next((position for value, position in seen if value == record['check_basis']), None)
            if previous is None:
                seen.append((record['check_basis'], index))
            else:
                record.pop('check_basis')
                record['check_basis_ref'] = {'json_path': ['transfers', previous, 'check_basis']}
        if 'scope_refutation' in record:
            previous = next((position for value, position in seen_scopes
                if _json_values_equal(value, record['scope_refutation'])), None)
            if previous is None:
                seen_scopes.append((record['scope_refutation'], index))
            else:
                record = {('scope_refutation_ref' if key == 'scope_refutation' else key):
                    ({'json_path': ['transfers', previous, 'scope_refutation']}
                     if key == 'scope_refutation' else value) for key, value in record.items()}
        transfers.append(record)
    if 'transfers' in document:
        result['transfers'] = transfers
    indexed = {}
    for index, record in enumerate(transfers):
        identifier = record.get('theory_id', record.get('theory_ref', {}).get('id'))
        indexed.setdefault(identifier, []).append((index, record))
    if 'executions' in document:
        result['executions'] = {}
        for key, execution in document['executions'].items():
            entry = dict(execution)
            if 'generated_theories' in entry.get('result', {}):
                generated = []
                for original in entry['result']['generated_theories']:
                    item = dict(original)
                    for field in ('binding', 'checks'):
                        matched = next((index for index, record in indexed.get(item.get('theory_id'), [])
                            if field in item and record.get(field) == item[field]), None)
                        if matched is not None:
                            item.pop(field)
                            item[field + '_ref'] = {'json_path': ['transfers', matched, field]}
                    generated.append(item)
                entry['result'] = {**entry['result'], 'generated_theories': generated}
            result['executions'][key] = entry
    return result


def _read_library(reference, workspace, *, document=None, source_libraries=None, source_documents=None):
    reference = _absolute_reference(reference, workspace)
    if document is None:
        document = load_source_values([reference], workspace)[0]['value']
    document = _anchor_file_references(document, _reference_workspace(reference))
    if source_documents is not None:
        location = (reference['file'], tuple(_path(reference)), reference.get('workspace'))
        # Capture each layer's original constructor and transfer positions before
        # merge appends or removes list entries. These shallow containers share
        # already loaded immutable records, rather than copying their payloads.
        source_documents[location] = {'candidates': dict(document.get('candidates', {})),
            'transfers': list(document.get('transfers', [])),
            **({'source_ref': document['source_ref']} if 'source_ref' in document else {})}
    if 'source_ref' not in document:
        library = document
    else:
        library = _read_library(document['source_ref'], Path(reference['file']).parent,
                                source_libraries=source_libraries, source_documents=source_documents)
        library = _merge_library_document(library, document)
    if source_libraries is not None and not reference.get('json_path'):
        # The next thin layer updates the candidate map in place. Keep this
        # layer's constructor roster, without retaining another large history.
        source_libraries[reference['file']] = {'candidates': dict(library.get('candidates', {}))}
    return library


def _merge_library_document(library, document):
    for field in _LIBRARY_MAPS:
        library.setdefault(field, {}).update(deepcopy(document.get(field, {})))
    for identifier in document.get('removed_candidates', []):
        library['candidates'].pop(identifier, None)
    _merge_intake(library, {field: document.get(field, []) for field in _LIBRARY_LISTS
                            if field != 'transfers'})
    transfers = library.setdefault('transfers', [])
    transfers.extend(deepcopy(item) for item in document.get('transfers', []) if item not in transfers)
    for key in document.get('removed_execution_keys', []):
        library.setdefault('executions', {}).pop(key, None)
    removed_transfers = set(document.get('removed_transfer_indices', []))
    if removed_transfers:
        library['transfers'] = [item for index, item in enumerate(transfers)
                                if index not in removed_transfers]
    return library


def install_theory_library(agent, reference=None):
    """Read one frozen library and retain only target-local writes on disk."""
    local = agent.workspace / LIBRARY_PATH
    agent._theory_library_candidate_versions = None
    agent._theory_library_entry_references = None
    if reference is None and not local.is_file():
        agent._theory_library_base, agent._theory_library_source_ref = {}, None
        return {}
    selected = _absolute_reference(reference or {'file': LIBRARY_PATH, 'json_path': []}, agent.workspace)
    document = load_source_values([selected], agent.workspace)[0]['value']
    if selected['file'] == str(local) and not document.get('source_ref'):
        base, source = {}, None
    else:
        source = _absolute_reference(document['source_ref'], Path(selected['file']).parent) \
            if selected['file'] == str(local) else selected
        documents = {}
        base = _read_library(source, agent.workspace,
            document=document if source == selected else None, source_documents=documents)
        _retain_library_candidate_versions(agent, source, base, documents)
        agent._theory_library_entry_references = {
            'source_ref': deepcopy(source),
            'files': dict(agent._theory_library_candidate_versions['files']), 'entries': {}}
    agent._theory_library_base = base
    agent._theory_library_source_ref = source
    if source is None:
        return deepcopy(document)
    library = deepcopy(base)
    if source != selected:
        _merge_library_document(library,
            _anchor_file_references(document, _reference_workspace(selected)))
    return library


def _library_document(agent):
    library, source = agent.theory_consolidation, getattr(agent, '_theory_library_source_ref', None)
    if source is None:
        return library
    base = agent._theory_library_base
    document = {'source_ref': deepcopy(source)}
    removed = [identifier for identifier in base.get('candidates', {})
               if identifier not in library.get('candidates', {})]
    if removed:
        document['removed_candidates'] = removed
    removed_executions = [key for key in base.get('executions', {})
                          if key not in library.get('executions', {})]
    if removed_executions:
        document['removed_execution_keys'] = removed_executions
    removed_transfers = [index for index, item in enumerate(base.get('transfers', []))
                         if item not in library.get('transfers', [])]
    if removed_transfers:
        document['removed_transfer_indices'] = removed_transfers
    for field in _LIBRARY_MAPS:
        changes = {key: value for key, value in library.get(field, {}).items()
                   if key not in base.get(field, {}) or value != base[field][key]}
        if changes:
            document[field] = changes
    for field in _LIBRARY_LISTS:
        changes = [item for item in library.get(field, []) if item not in base.get(field, [])]
        if changes:
            document[field] = changes
    return document


def library_entry_reference(agent, field, key, *parts, documents=None, local_fields=None):
    """Address inherited checks at their source and new checks at their local index."""
    reference = {'file': LIBRARY_PATH, 'json_path': [field, key, *parts]}
    source = getattr(agent, '_theory_library_source_ref', None)
    if not source:
        return reference
    value = agent.theory_consolidation[field][key]
    base = agent._theory_library_base.get(field, [] if field in _LIBRARY_LISTS else {})
    if field in _LIBRARY_LISTS:
        if value not in base:
            # A caller may share this shallow list within one read-only batch.
            # The next batch starts fresh, including same-length local edits.
            if local_fields is not None and field in local_fields:
                local = local_fields[field]
            else:
                local = [item for item in agent.theory_consolidation.get(field, []) if item not in base]
                if local_fields is not None:
                    local_fields[field] = local
            reference['json_path'][1] = local.index(value)
            return reference
        source_key = base.index(value)
    elif key not in base or value != base[key]:
        return reference
    else:
        source_key = key
    # Retain only physical addresses of unchanged base values. The complete
    # payload already lives in _theory_library_base; do not cache a second tree.
    documents = {} if documents is None else documents
    retained = getattr(agent, '_theory_library_entry_references', None)
    if retained and retained['source_ref'] == source:
        changed = any((state.st_mtime_ns, state.st_size) != previous
            for filename, previous in retained['files'].items()
            for state in (Path(filename).stat(),))
        if changed:
            for location in list(documents):
                if location[0] in retained['files']:
                    documents.pop(location)
            retained['entries'].clear()
            retained['files'].clear()
        saved = retained['entries'].get((field, source_key))
        if saved is not None:
            return {**deepcopy(saved), 'json_path': [*_path(saved), *parts]}
    else:
        retained = None
    base_key = source_key
    # The frozen source may itself be a target delta. Follow its own addresses.
    while True:
        location = (source['file'], tuple(_path(source)), source.get('workspace'))
        if location not in documents:
            documents[location] = _anchor_file_references(
                load_source_values([source], agent.workspace)[0]['value'], _reference_workspace(source))
        document = documents[location]
        if retained is not None:
            state = Path(source['file']).stat()
            retained['files'][source['file']] = (state.st_mtime_ns, state.st_size)
        entries = document.get(field, [] if field in _LIBRARY_LISTS else {})
        if field in _LIBRARY_LISTS and value in entries:
            source_key = entries.index(value)
            break
        if field in _LIBRARY_MAPS and key in entries and entries[key] == value:
            source_key = key
            break
        if 'source_ref' not in document:
            raise ValueError('retained library entry is unavailable')
        source = _absolute_reference(document['source_ref'], Path(source['file']).parent)
    origin = {**deepcopy(source), 'json_path': [*_path(source), field, source_key]}
    if retained is not None:
        retained['entries'][field, base_key] = origin
    return {**deepcopy(origin), 'json_path': [*_path(origin), *parts]}


def _compact_index_programs(document):
    """Keep exact repeated frozen Programs at one address in this index file."""
    delta = 'source_ref' in document and 'entries' in document
    indexed = document['entries'] if delta else document
    prefix, seen = (['entries'] if delta else []), {}

    def compact(entry, path):
        result = {}
        for field, value in entry.items():
            if field == 'program':
                # Keep complete ordered JSON bytes, not a digest. Provenance
                # stays on each theory independently of this literal payload.
                encoded = json.dumps({'program': value}, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')
                previous = seen.get(encoded)
                local = {'json_path': previous} if previous is not None else None
                if local is not None and len(json.dumps({'program_ref': local},
                        ensure_ascii=False, separators=(',', ':')).encode('utf-8')) < len(encoded):
                    result['program_ref'] = local
                    continue
                seen.setdefault(encoded, [*path, 'program'])
            result[field] = ([compact(old, [*path, 'history', index])
                for index, old in enumerate(value)] if field == 'history' else value)
        return result

    compacted = {identifier: compact(entry, [*prefix, identifier])
                 for identifier, entry in indexed.items()}
    return {**document, 'entries': compacted} if delta else compacted


def _restore_index_programs(indexed, document, workspace):
    """Expand local Program fields before exposing an independently mutable index."""
    from .research_episodes import _source_path_value

    def restore(entry):
        result = {}
        for field, value in entry.items():
            if (field == 'program_ref' and 'program' not in entry
                    and isinstance(value, dict) and 'json_path' in value
                    and not value.get('file') and not value.get('archive')):
                result['program'] = deepcopy(_source_path_value(document, value['json_path'],
                    lambda saved: load_source_values([saved], workspace)[0]['value']))
            else:
                result[field] = ([restore(old) for old in value]
                                 if field == 'history' else value)
        return result

    return {identifier: restore(entry) for identifier, entry in indexed.items()}


def _read_index(reference, workspace):
    reference = _absolute_reference(reference, workspace)
    root = load_source_values([{**reference, 'json_path': []}], workspace)[0]['value']
    indexed = root
    for part in _path(reference):
        indexed = indexed[part]
    if 'source_ref' in indexed and 'entries' in indexed:
        merged = _read_index(indexed['source_ref'], Path(reference['file']).parent)
        merged.update(deepcopy(_restore_index_programs(indexed['entries'], root, workspace)))
        for identifier in indexed.get('removed_entries', []):
            merged.pop(identifier, None)
        return merged
    return _snapshot_index(_restore_index_programs(indexed, root, workspace), root, reference)


def _snapshot_index(indexed, root, reference):
    from .question_routing import resolved_theories

    origin = _reference_workspace(reference, root)

    def snapshot(identifier, entry):
        # Anchoring below already builds an independent nested record. Only
        # detach these top-level fields before resolving the retained source.
        item = dict(entry)
        history = item.pop('history', [])
        original = item.pop('source_ref', None)
        if original:
            proxy = SimpleNamespace(theory_index={identifier: {**item, 'source_ref': original}},
                questions=root.get('questions', root.get('online_questions', [])),
                relation_research=root.get('relation_research', {}), events=root.get('events', []),
                theory_consolidation={}, workspace=Path(reference['file']).parent)
            resolved, = resolved_theories(proxy)
            item = {key: value for key, value in resolved.items() if key not in {'source_ref', 'history'}}
            item['source_snapshot'] = {'source_ref': original, 'index_ref': deepcopy(reference),
                                       'version': entry.get('version', 1)}
        item = _anchor_file_references(item, origin)
        if history:
            item['history'] = [snapshot(identifier, old) for old in history]
        return item
    return {identifier: snapshot(identifier, entry) for identifier, entry in indexed.items()}


def install_theory_index(agent, reference=None):
    reference = _absolute_reference(reference, agent.workspace) if reference else None
    indexed = _read_index(reference, agent.workspace) if reference else {}
    agent._theory_index_source_ref = reference
    agent._theory_index_base = deepcopy(indexed)
    return indexed


def save_theory_index(agent):
    source = getattr(agent, '_theory_index_source_ref', None)
    base = getattr(agent, '_theory_index_base', {})
    changes = {key: value for key, value in agent.theory_index.items()
               if key not in base or value != base[key]}
    changes = _snapshot_index(changes, {'questions': agent.questions,
        'relation_research': agent.relation_research, 'events': agent.events},
        {'file': str(agent.workspace / 'research_state.json'), 'json_path': ['theory_index']})
    document = {'source_ref': deepcopy(source), 'entries': changes} if source else changes
    if source:
        removed = [key for key in base if key not in agent.theory_index]
        if removed:
            document['removed_entries'] = removed
    destination = agent.workspace / INDEX_PATH
    write_json_checkpoint(destination, _compact_index_programs(document), indent=None,
                          ensure_ascii=False, separators=(',', ':'))
    agent.files[INDEX_PATH] = published_checkpoint_text(agent, destination)


def _save(agent):
    destination = agent.workspace / LIBRARY_PATH
    write_json_checkpoint(destination, _compact_library_fields(_library_document(agent)),
                          indent=None, ensure_ascii=False, separators=(',', ':'))
    agent.files[LIBRARY_PATH] = published_checkpoint_text(agent, destination)


def _consolidation_return(question, theme, result, allowance):
    """Keep completed bounded work separate from its remaining author obligation."""
    if question is None:
        return result
    if not result.get('author_work'):
        question.pop('pending_author_work', None)
        if result['status'] != 'UNKNOWN':
            question['completed_agenda_basis'] = question_work_basis(question)
        return result
    from .question_continuation import _author_work

    question.pop('completed_agenda_basis', None)
    origin = question.get('origin')
    request = _author_work(question, 'unresolved_consolidation_author_work', 'typed_shared_structure')
    request.update(bounded_status=result['status'], bounded_allowance=allowance,
        agenda_basis=question_work_basis(question),
        author_work_ref={'file': result['library_path'],
                         'json_path': ['executions', theme, 'result', 'author_work']},
        bounded_result_ref={'file': result['library_path'], 'json_path': ['executions', theme, 'result']},
        required_contract={'source_refs': deepcopy(origin.get('source_refs', []) if isinstance(origin, dict) else []),
            'required_output': 'Author or refine the executable shared structure or its missing contracts; '
                               'retain source reconstructions and counterexamples.'})
    question['pending_author_work'] = deepcopy(request)
    return {**result, **request}


def _allocation(agent, arguments, question=None):
    if question is None and 'mission' in arguments:
        agent.research_mission = adopt_research_mission(agent.research_mission, arguments['mission'])
    mission = effective_question_mission(question, arguments.get('mission', agent.research_mission)
                                         if question is not None else agent.research_mission)
    funding = mission.get('allocation', mission.get('theory_learning_allocation', {}))
    if mission.get('purpose') not in {'theory_learning', 'method_learning'} or not funding:
        return None
    funding = funding.get('compute', 0) if isinstance(funding, dict) else funding
    requested = arguments.get('allocation', {}).get('compute', funding)
    compute = min(funding, requested, mission_compute_remaining(agent, mission))
    if type(compute) is not int or compute < 0:
        raise ValueError('theory allocation needs nonnegative integer compute')
    allocation = {'compute': compute, 'max_groups': arguments.get('allocation', {}).get('max_groups', 16)}
    for field in ('parameter_relations', 'source_pairs'):
        if field in arguments.get('allocation', {}):
            allocation[field] = deepcopy(arguments['allocation'][field])
    return allocation


def _merge_intake(library, intake):
    def same_origin(left, right):
        lref, rref = (_canonical(item.get('source_ref', {})) for item in (left, right))
        if lref != rref:
            return False
        global_reference = all(lref.get(key) for key in ('archive', 'unit', 'member'))
        return global_reference or left['source_id'] == right['source_id']

    def same_episode(left, right):
        if left['episode_id'] != right['episode_id']:
            return False
        if any(all(item.get('source_ref', {}).get(key) for key in ('archive', 'unit', 'member'))
               for item in (left, right)):
            return same_origin(left, right)
        return left['source_id'] == right['source_id']

    materials = library.setdefault('materials', [])
    for original in intake['materials']:
        if any(same_origin(item, original) and all(item.get(key) == original.get(key)
                for key in ('id', 'claim', 'original_kind')) for item in materials):
            continue
        entry = deepcopy(original)
        first = next((item for item in materials if same_origin(item, entry)), None)
        if first:
            entry.update(kind='duplicate', duplicate_of=first['id'], duplicate_source_id=first['source_id'])
        materials.append(entry)
    library['conflicts'] = _unique([*library.get('conflicts', []), *intake['conflicts']])
    origins = library.setdefault('unique_support_sources', [])
    for entry in intake['unique_support_sources']:
        if not any(same_origin(item, entry) for item in origins):
            origins.append(deepcopy(entry))
    episodes = library.setdefault('episodes', [])
    for episode in intake['episodes']:
        previous = next((index for index, old in enumerate(episodes) if same_episode(old, episode)), None)
        if previous is None:
            episodes.append(deepcopy(episode))
        else:
            episodes[previous] = deepcopy(episode)


def _retained_trigger_examples(agent, entries, retained):
    """Keep all saved sufficient failure witnesses in their original order.

    Whole-prefix improvement and its original criterion remain separate from
    individual residuals; paid observations live only in compilation memory.
    """
    references, work = [], []
    for entry in entries:
        comparison = entry.get('source_comparison')
        if not comparison:
            continue
        if comparison.get('status') == 'UNKNOWN':
            comparison['paid_mse_effect'] = {'status': 'UNKNOWN', 'reason': comparison['reason']}
            entry['trigger_obligations'] = [{'status': 'UNKNOWN', 'reason': comparison['reason'],
                'source_ref': deepcopy(entry['action_ref'])}]
            continue
        start = len(references)
        references.extend(comparison[key] for key in
            ('before_evaluation_ref', 'after_evaluation_ref', 'criterion_ref'))
        references.extend(comparison['paid_observation_refs'])
        work.append((entry, start, len(comparison['paid_observation_refs'])))
    values = resolve_program_sources(agent, references, retained)
    for entry, start, count in work:
        comparison = entry['source_comparison']
        before, after, threshold = values[start:start + 3]
        n = comparison['scope'].get('evaluated_rows')
        reason = 'retained_paid_mse_scope_or_criterion_unavailable'
        valid = (type(n) is int and n > 0 and isinstance(threshold, (int, float))
            and isfinite(threshold) and threshold >= 0 and comparison['scope'].get('same_paid_rows') is True
            and all(score.get('returncode') == 0 and score.get('loss_kind') == 'mse'
                and score.get('paid_residuals', {}).get('evaluated_rows') == n for score in (before, after)))
        scores = [score.get('mse') for score in (before, after)]
        if valid and all(isinstance(value, (int, float)) and isfinite(value) and value >= 0
                         for value in scores):
            change = scores[1] - scores[0]
            direction = 'improved' if change < 0 else 'worsened' if change > 0 else 'unchanged'
            comparison['paid_mse_effect'] = {
                'status': 'measured', 'direction': direction,
                'before_mse': scores[0], 'after_mse': scores[1], 'mse_change': change,
                'evaluated_rows': n, 'threshold': threshold,
                'before_criterion_status': 'passed' if scores[0] <= threshold else 'failed',
                'after_criterion_status': 'passed' if scores[1] <= threshold else 'failed',
                'before_evaluation_ref': deepcopy(comparison['before_evaluation_ref']),
                'after_evaluation_ref': deepcopy(comparison['after_evaluation_ref']),
                'criterion_ref': deepcopy(comparison['criterion_ref']),
                'scope': 'same_paid_rows_not_independent_confirmation'}
        else:
            comparison['paid_mse_effect'] = {'status': 'UNKNOWN',
                'reason': 'finite_saved_same_scope_paid_mse_required'}
        if valid:
            reason = 'saved_single_residual_paid_mse_failure_witness_unavailable'
            for index, residual in enumerate(before['paid_residuals'].get('examples', [])):
                error = residual.get('squared_error')
                if (not isinstance(error, (int, float)) or not isfinite(error) or error <= n * threshold
                        or type(residual.get('row_index')) is not int or not 0 <= residual['row_index'] < n):
                    continue
                for record_ref, record in zip(comparison['paid_observation_refs'], values[start + 3:start + 3 + count]):
                    indices = record.get('row_indices', [])
                    if residual['row_index'] not in indices:
                        continue
                    offset = indices.index(residual['row_index'])
                    observation = record['rows'][offset]
                    if any(observation.get(key, []) != residual.get(key, [])
                           for key in ('features', 'history', 'target')):
                        continue
                    row_ref = {**deepcopy(record_ref), 'json_path': [*_path(record_ref), 'rows', offset]}
                    prior_ref = comparison['before_evaluation_ref']
                    criterion = {**deepcopy(comparison), 'kind': 'paid_mse_single_residual_witness',
                        'threshold': threshold, 'evaluated_rows': n,
                        'evaluated_rows_ref': {**deepcopy(prior_ref),
                            'json_path': [*_path(prior_ref), 'paid_residuals', 'evaluated_rows']},
                        'before_residual_ref': {**deepcopy(prior_ref),
                            'json_path': [*_path(prior_ref), 'paid_residuals', 'examples', index]},
                        'paid_row_ref': row_ref, 'sufficiency': 'one_row_proves_paid_mse_failure_only'}
                    witness = {'source_ref': row_ref, 'observation': deepcopy(observation),
                               'tolerance': sqrt(n * threshold), 'criterion': criterion}
                    counterexamples = entry.setdefault('counterexamples', [])
                    if witness not in counterexamples:
                        counterexamples.append(witness)
                    reason = None
                    break
        if reason:
            entry['trigger_obligations'] = [{'status': 'UNKNOWN', 'reason': reason,
                'source_ref': deepcopy(entry['action_ref'])}]


def _revision_contract_sources(agent, entries, pool):
    """Address each actual endpoint's own typed constructor, without tree inference.

    The consumer matches these patterns under its SearchBudget. Private pattern
    values live only in this call; saved provenance contains their original refs.
    """
    endpoints = [endpoint for entry in entries for endpoint in entry.get('fit_state_refs', {}).values()
                 if endpoint.get('kind') == 'frozen_revision_export']
    references = _unique(endpoint['transfer_ref'] for endpoint in endpoints)
    transfers = {_canonical_key(reference): item['value'] for reference, item in zip(
        references, load_source_values(references, agent.workspace))}
    libraries, documents = {}, {}

    def constructor(endpoint, pattern_path):
        reference = endpoint['transfer_ref']
        transfer = transfers[_canonical_key(reference)]
        if transfer.get('theory_ref') != endpoint.get('theory_ref'):
            return {'reason': 'retained_constructor_theory_version_mismatch'}
        binding = transfer['binding']
        transfer_file = Path(_absolute_reference(reference, agent.workspace)['file'])
        origin = transfer_file.parent.parent if transfer_file.parent.name == 'shared' else transfer_file.parent
        selected = _absolute_reference({'file': binding['library_path'], 'json_path': []}, origin)
        key = _canonical_key(selected)
        if key not in libraries:
            proxy = SimpleNamespace(workspace=origin)
            proxy.theory_consolidation = install_theory_library(proxy, selected)
            libraries[key] = proxy
        proxy = libraries[key]
        identifier = binding['candidate_id']
        candidate = proxy.theory_consolidation.get('candidates', {}).get(identifier)
        if candidate is None or candidate.get('version') != binding['candidate_version']:
            return {'reason': 'retained_constructor_version_unavailable'}
        pattern = candidate
        for part in pattern_path:
            pattern = pattern.get(part) if isinstance(pattern, dict) else None
        if pattern is None:
            return {'reason': 'retained_constructor_pattern_unavailable'}
        original = _absolute_reference(library_entry_reference(
            proxy, 'candidates', identifier, documents=documents), origin)
        return {'_pattern': pattern, '_holes': candidate['holes'],
            '_bound_holes': binding.get('components', {}), 'source_ref': {
            'program_ref': deepcopy(endpoint['program_ref']),
            'pattern_ref': {**original, 'json_path': [*_path(original), *pattern_path]},
            'holes_ref': {**original, 'json_path': [*_path(original), 'holes']},
            'transfer_ref': deepcopy(reference),
            'components_ref': {**_canonical(reference), 'json_path': [*_path(reference), 'binding', 'components']}}}

    for entry in entries:
        states = entry.get('fit_state_refs', {})
        for side in ('before', 'after'):
            reference = _absolute_reference(entry[f'{side}_ref'], agent.workspace)
            endpoint = states.get(side, {})
            # A newly exported declaration already retains exact occurrence
            # contracts. Match its address, never another equal-valued Program.
            declarations = [value for value in pool.values() if value.get('program_source_ref')
                and _absolute_reference(value['program_source_ref'], agent.workspace) == reference]
            if len(declarations) == 1 and 'subtree_contracts' in declarations[0]:
                entry[f'{side}_subtree_contracts'] = deepcopy(declarations[0]['subtree_contracts'])
            if endpoint.get('kind') == 'frozen_revision_export':
                entry[f'_{side}_contract_source'] = constructor(endpoint, ['definition', 'body'])
            elif side == 'before' and states.get('after', {}).get('kind') == 'frozen_revision_export':
                after = states['after']
                transfer = transfers[_canonical_key(after['transfer_ref'])]
                source = endpoint.get('source_ref')
                parent = transfer.get('binding', {}).get('parent', {})
                if (source and source == parent.get('source_ref')
                        and endpoint.get('program_ref') == entry['before_ref']):
                    entry['_before_contract_source'] = constructor(after, ['before_pattern'])
                    entry['_before_contract_source'].get('source_ref', {}).update(
                        program_ref=deepcopy(entry['before_ref']))


def _canonical_key(reference):
    return json.dumps(_canonical(reference), sort_keys=True)


def _same_retained_parent_binding(agent, original, current):
    """Accept an exact binding or its equal ordered paid-row relocation only."""
    if original == current:
        return True
    if not isinstance(original, dict) or not isinstance(current, dict):
        return False
    left, right = original.get('fitting'), current.get('fitting')
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    old_refs, new_refs = left.get('evidence_refs', []), right.get('evidence_refs', [])
    if not old_refs or len(old_refs) != len(new_refs):
        return False
    def without_files(binding, fitting, references):
        return {**binding, 'fitting': {**fitting, 'evidence_refs': [
            {key: value for key, value in reference.items() if key != 'file'}
            for reference in references]}}
    if without_files(original, left, old_refs) != without_files(current, right, new_refs):
        return False
    try:
        values = resolve_program_sources(agent, [*old_refs, *new_refs])
    except (OSError, KeyError, IndexError, ValueError):
        return False
    return values[:len(old_refs)] == values[len(old_refs):]


def _retained_parent_contract(agent, parent):
    """Reuse this exported version's constructor; lexical matching is budgeted later."""
    if parent is None or parent.get('subtree_contracts'):
        return parent
    source = parent.get('_contract_source') or {}
    if all(key in source for key in ('_pattern', '_holes', '_bound_holes', 'source_ref')):
        # The scene already resolved this exact constructor in RAM. The normal
        # _component preparation still checks its Program, endpoint and units.
        return parent
    exports = [(identifier, meaning, item) for (identifier, meaning), (question, _) in
        _question_versions(agent).items() for item in question.get('revision_candidate_exports', [])
        if item.get('theory_ref', {}).get('id') == parent['id']]
    if not exports:
        return parent
    result = deepcopy(parent)

    def unavailable(reason):
        result['_contract_source'] = {'reason': reason}
        return result

    matches = [(identifier, meaning, item) for identifier, meaning, item in exports
               if item['theory_ref'].get('version') == parent.get('version', 1)]
    if len(matches) != 1:
        return unavailable('retained_parent_export_version_mismatch')
    identifier, meaning, exported = matches[0]
    program_ref = exported.get('program_source_ref', {})
    if (not parent['id'].startswith(f'frozen_recall:{identifier}@{meaning}:')
            or _path(program_ref) != ['theory_index', parent['id'], 'program']
            or program_ref.get('version') != parent.get('version', 1)
            or not exported.get('transfer_ref')):
        return unavailable('retained_parent_export_reference_mismatch')
    # A restored local library is a delta. Its old export's relative list
    # index belongs to the original physical document, not this thin shell.
    transfers, documents, local_fields = [], {}, {}
    for index, transfer in enumerate(agent.theory_consolidation.get('transfers', [])):
        if transfer.get('theory_ref') != exported['theory_ref']:
            continue
        reference = _absolute_reference(library_entry_reference(
            agent, 'transfers', index, documents=documents, local_fields=local_fields), agent.workspace)
        origin = _reference_workspace(reference) or Path(reference['file']).parent
        exported_transfer = _absolute_reference(exported['transfer_ref'], origin)
        if (reference['file'] == exported_transfer['file']
                and _path(reference) == _path(exported_transfer)
                and _same_retained_parent_binding(agent,
                    _anchor_file_references(transfer.get('binding'), origin),
                    _anchor_file_references(parent.get('consolidation_ref'), origin))):
            transfers.append((reference, origin))
    if len(transfers) != 1:
        return unavailable('retained_parent_transfer_reference_required')
    transfer_ref, origin = transfers[0]
    program_ref = _absolute_reference(program_ref, origin)
    if (parent.get('program_source_ref') is not None and
            _absolute_reference(parent['program_source_ref'], origin) != program_ref):
        return unavailable('retained_parent_program_reference_mismatch')
    entry = {'before_ref': program_ref, 'after_ref': program_ref, 'fit_state_refs': {'after': {
        'kind': 'frozen_revision_export', 'program_ref': program_ref,
        'theory_ref': deepcopy(exported['theory_ref']),
        'transfer_ref': transfer_ref}}}
    _revision_contract_sources(agent, [entry], {parent['id']: parent})
    result['program_source_ref'] = deepcopy(program_ref)
    result['_contract_source'] = entry['_after_contract_source']
    return result


def _revision_entries(agent, episodes, pool, retained):
    pending, references = [], []
    for episode in episodes:
        declaration = pool[episode['theory_id']]
        declared_ref = declaration.get('program_source_ref')
        if declared_ref:
            declared_ref = _absolute_reference(declared_ref, agent.workspace)
        # Retentions within one ordered pool comparison are intermediate
        # selections, not separate research actions. Keep that original ledger
        # in the Episode, but learn from the complete first-to-terminal step.
        steps = episode.get('source_revision_steps')
        changes = (episode.get('actual_changes', []) if steps is None else
                   [step for step in steps if step.get('status') == 'completed'
                    and step.get('retained_outcome') == 'parent_changed'])
        for index, change in enumerate(changes):
            if not change.get('before_ref'):
                continue
            after_ref = change['source_ref']
            actual_after = _absolute_reference(after_ref, agent.workspace)
            if declared_ref and not (
                    _contains_reference(actual_after, declared_ref)
                    and _contains_reference(declared_ref, actual_after)):
                continue
            entry = {key: deepcopy(episode[key]) for key in (
                'source_id', 'theory_id', 'role_signature', 'output_unit', 'context_ref',
                'context_family_ref', 'evidence_refs', 'cost_refs', 'failure_refs', 'source_cutoff',
                'counterexamples') if key in episode}
            entry.update(episode_id=f'{episode["episode_id"]}:{"step" if steps is not None else "revision"}{index}',
                source_ref=deepcopy(after_ref), before_ref=deepcopy(change['before_ref']),
                after_ref=deepcopy(after_ref), action_ref=deepcopy(change.get('action_ref')),
                source_outcomes={key: deepcopy(episode[key]) for key in (
                    'status', 'costs', 'cost_scope', 'independent_results', 'retained_later_outcome_refs',
                    'research_action_refs')
                    if key in episode})
            if steps is not None:
                entry['source_outcomes']['source_revision_steps'] = deepcopy(steps)
            if 'source_comparison' in change:
                entry['source_comparison'] = {**deepcopy(change['source_comparison']),
                    'scope': deepcopy(change.get('scope', {})),
                    'paid_observation_refs': deepcopy(change.get('evidence_refs', []))}
                if steps is not None:
                    entry['source_comparison']['research_step'] = {key: deepcopy(change[key]) for key in (
                        'kind', 'status', 'action_ref', 'declared_models_ref', 'comparison_pool_ref',
                        'member_retention_refs', 'retained_outcome', 'cost_refs', 'cost_scope', 'producer_link')
                        if key in change}
            if 'fit_state_refs' in change:
                entry['fit_state_refs'] = deepcopy(change['fit_state_refs'])
            pending.append(entry)
            references.extend([change['before_ref'], after_ref])
    values = resolve_program_sources(agent, references, retained)
    for index, entry in enumerate(pending):
        entry['before_program'], entry['after_program'] = values[index * 2:index * 2 + 2]
    _revision_contract_sources(agent, pending, pool)
    _retained_trigger_examples(agent, pending, retained)
    return pending


def _same_frozen_candidate(old, new):
    from .parent_reference_relations import parent_reference_contract

    fields = ('kind', 'instances', 'source_bindings', 'role_signature', 'parameter_paths',
              'scope', 'source_evidence_refs', 'source_outcomes', 'counterexample_checks',
              'blocked_transfer_bindings', 'learned_constraints', 'holes', 'before_pattern',
              'constraint_hypotheses', 'trigger_checks', 'parameter_relations',
              'purpose', 'source_change_kind', 'structural_change')
    if not all(old.get(key) == new.get(key) for key in fields):
        return False
    if parent_reference_contract(old) != parent_reference_contract(new):
        return False
    if new.get('definition') is not None:
        return all(old.get('definition', {}).get(key) == new['definition'].get(key)
                   for key in ('arguments', 'body', 'definitions'))
    return old.get('template', {}).get('pattern') == new.get('template', {}).get('pattern')


def _episode_file_versions(agent, references):
    versions = []
    for reference in references:
        if reference.get('archive') or not reference.get('file'):
            continue
        path = agent.workspace / reference['file']
        stat = path.stat()
        entry = {'file': reference['file'], 'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns}
        if entry not in versions:
            versions.append(entry)
    return versions


def _new_local_materials(agent, library):
    """Address each new local record once; leave its payload in the old store."""
    source = _source_id(agent)
    previous = library.get('material_cursors', {}).get(source, {})
    materials, cursor = [], {}
    for collection, kind, index_name in (
            ('observations', 'observation', 'record_index'),
            ('revisions', 'revision', 'index'), ('events', 'event', 'index')):
        records = getattr(agent, collection, [])
        cursor[collection] = len(records)
        for index in range(previous.get(collection, 0), len(records)):
            record = records[index]
            feedback = record.get('feedback', {}) if collection == 'events' else record
            failed = (feedback.get('error') or feedback.get('returncode', 0) != 0
                      or feedback.get('evidence_status') == 'refuted'
                      or feedback.get('status', '').endswith('_failed'))
            classification = ('measured' if collection == 'observations' else
                'error' if failed else 'execution' if feedback else 'unresolved')
            materials.append({'id': f'native:{kind}:{index}', 'kind': classification,
                              'source_ref': {'kind': kind, index_name: index}})
    return materials, source, cursor


def consolidate_research(agent, arguments, laboratory=None, question=None):
    from .question_routing import register_theories, resolved_theories

    object_type = arguments.get('object_type')
    if object_type not in (None, 'program_revision_relation'):
        raise ValueError('unsupported shared construction object_type')
    register_theories(agent, arguments.get('theories', []))
    allocation = _allocation(agent, arguments, question)
    if allocation is None:
        return {'status': 'research_deferred', 'reason': 'theory_allocation_missing', 'new_compute_cost': 0}
    common = load_common_sources(arguments.get('common_source_refs', []), agent.workspace)
    register_theories(agent, common['theories'])
    library = agent.theory_consolidation
    mission = effective_question_mission(question, arguments.get('mission', agent.research_mission)
                                         if question is not None else agent.research_mission)
    library.setdefault('candidates', {})
    library.setdefault('executions', {})
    new_materials, material_source, material_cursor = _new_local_materials(agent, library)
    selected = arguments.get('source_theory_ids',
        [theory['id'] for theory in common['theories']] if arguments.get('common_source_refs')
        else list(agent.theory_index))
    pool = {theory['id']: theory for theory in resolved_theories(agent)}
    theme = arguments.get('theme', (question or {}).get('id', 'retained_programs'))
    request_basis = {key: deepcopy(arguments.get(key)) for key in (
        'source_theory_ids', 'materials', 'episode_sources', 'common_source_refs',
        'counterexamples', 'context_family_ref', 'allocation', 'object_type')}
    request_basis.update(question_meaning=({'id': question['id'],
        'meaning_version': question.get('meaning_version', 1)} if question is not None else None),
        source_id=_source_id(agent), sources=[{key: deepcopy(pool[identifier].get(key)) for key in (
        'id', 'version', 'source_ref', 'source_status', 'compatible')} for identifier in selected if identifier in pool],
        paid_cutoff=[len(agent.observations), len(agent.rows)],
        revisions=[{key: deepcopy(revision.get(key)) for key in ('index', 'status', 'evidence_status', 'check_results')}
                   for revision in agent.revisions], max_groups=allocation['max_groups'],
        theory_allocation=deepcopy(mission.get('allocation', mission.get('theory_learning_allocation'))),
        episode_file_versions=_episode_file_versions(agent,
            [*arguments.get('episode_sources', []), *arguments.get('common_source_refs', [])]),
        source_revision_steps=[{'source_id': episode['source_id'],
            'question_ref': deepcopy(episode.get('question_ref')),
            'steps': deepcopy(episode['source_revision_steps'])}
            for episode in common['episodes'] if episode.get('source_revision_steps')],
        local_comparison_refs=[{'file': 'research_context.json', 'json_path': ['events', index]}
            for index, event in enumerate(agent.events)
            if event.get('feedback', {}).get('operator') == 'research_choice'],
        native_versions=[{'question_id': q['id'], 'meaning_version': q.get('meaning_version', 1),
            'rounds': [{key: deepcopy(round_.get(key)) for key in ('round_index', 'status')}
                       for round_ in q.get('native_condition_research', [])]} for q in agent.questions
                         if q.get('native_condition_research')])
    request_basis['revision_objectives'] = ['ordered_revision', 'cross_position_parent_reference_context']
    previous = library['executions'].get(theme)
    if (previous and previous['request_basis'] == request_basis and
            (previous['result']['status'] != 'UNKNOWN' or allocation['compute'] <= previous['allowance'])):
        if new_materials:
            _merge_intake(library, extract_episodes(agent, new_materials,
                excluded_theory_ids=common['theory_ids']))
            library.setdefault('material_cursors', {})[material_source] = material_cursor
            _save(agent)
        result = {**deepcopy(previous['result']), 'cache_hit': True, 'new_compute_cost': 0,
                'conflict_count': len(library['conflicts']),
                'support_source_count': len(library['unique_support_sources'])}
        return _consolidation_return(question, theme, result, previous['allowance'])
    from .research_episodes import load_episode_sources
    retained = load_episode_sources(arguments.get('episode_sources', []), agent.workspace)
    intake = extract_episodes(agent, [*new_materials, *arguments.get('materials', [])],
                              retained_episodes=retained, excluded_theory_ids=common['theory_ids'])
    for field in ('episodes', 'materials', 'conflicts', 'unique_support_sources'):
        intake[field].extend(common[field])
    _merge_intake(library, intake)
    library.setdefault('material_cursors', {})[material_source] = material_cursor
    episodes = []
    for episode in intake['episodes']:
        for identifier in episode['theory_ids']:
            if identifier not in selected:
                continue
            theory = pool[identifier]
            entry = {**deepcopy(episode), 'episode_id': f'{episode["episode_id"]}:{identifier}',
                     'theory_id': identifier, 'source_ref': deepcopy(theory.get('program_source_ref') or theory.get('source_ref') or {
                         'file': 'research_state.json', 'json_path': ['theory_index', identifier],
                         'version': theory.get('version', 1)}),
                     'evidence_refs': _unique([*episode['change_evidence'],
                         *theory.get('support', {}).get('evidence', [])]),
                     'counterexamples': deepcopy(arguments.get('counterexamples', {}).get(identifier, []))}
            for field in ('program', 'role_signature', 'output_unit', 'context_ref',
                          'context_family_ref', 'declared_scope_ref', 'fragment_contracts',
                          'exceptions', 'lookup_table'):
                if field in theory:
                    entry[field] = deepcopy(theory[field])
            if theory.get('source_status') == 'unavailable_retained_version':
                entry.pop('program', None)
            if 'context_family_ref' in arguments:
                entry['context_family_ref'] = deepcopy(arguments['context_family_ref'])
            episodes.append(entry)
    serial = len(library['candidates'])
    if object_type == 'program_revision_relation':
        from .revision_relations import consolidate_revision_relations
        revisions = _revision_entries(agent, episodes, pool, retained)
        computed = consolidate_revision_relations(revisions,
            allocation=allocation, candidate_id=f'consolidated:{serial}:relation', object_type=object_type)
    else:
        computed = consolidate_theories(episodes, allocation=allocation, candidate_id=f'consolidated:{serial}')
        revisions = _revision_entries(agent, episodes, pool, retained)
        if revisions:
            from .revision_relations import consolidate_revision_relations
            relations = consolidate_revision_relations(revisions, allocation={
                'compute': max(0, allocation['compute'] - computed['actual_compute_cost']),
                'max_groups': max(0, allocation['max_groups'] - len(computed['candidates']))},
                candidate_id=f'consolidated:{serial}:relation')
            computed['candidates'].extend(relations['candidates'])
            computed['excluded'].extend(relations['excluded'])
            computed['author_work'].extend(relations['author_work'])
            computed['parameter_relation_searches'] = relations.get('parameter_relation_searches', [])
            computed['subtree_contract_recovery'] = relations.get('subtree_contract_recovery', [])
            computed['actual_compute_cost'] += relations['actual_compute_cost']
            computed['budget_exhausted'] = computed['budget_exhausted'] or relations['budget_exhausted']
            if computed['budget_exhausted']:
                computed['status'] = 'UNKNOWN'
    computed['author_work'].extend(common.get('author_work', []))
    # Historical unknown boundaries remain in the Episode. Request missing
    # inputs only when this selected source has no consumed revision step.
    computed['author_work'].extend(_unique([
        {'type': 'AuthorWorkRequest', 'reason': step['reason'],
         'source_id': episode['source_id'], 'source_ref': deepcopy(step['action_ref'])}
        for episode in episodes for step in episode.get('source_revision_steps', [])
        if step.get('status') == 'UNKNOWN'
        if not any(entry['theory_id'] == episode['theory_id']
                   and entry['source_id'] == episode['source_id'] for entry in revisions)]))
    computed['author_work'].extend({**deepcopy(item), 'type': 'AuthorWorkRequest',
        'episode_id': entry['episode_id']} for entry in revisions for item in entry.get('trigger_obligations', []))
    if common.get('status') == 'UNKNOWN':
        computed['status'] = 'UNKNOWN'
    before_compute = agent.costs['compute_cost']
    agent.costs['compute_cost'] += computed['actual_compute_cost']
    debit_mission_compute(agent, mission, before_compute)
    references = []
    for candidate in computed['candidates']:
        prior = next((item for item in library['candidates'].values()
                      if _same_frozen_candidate(item, candidate)), None)
        if prior is not None:
            candidate = prior
        else:
            candidate['theme'] = theme
            candidate['version'] = 1
            library['candidates'][candidate['id']] = candidate
        references.append({'candidate_id': candidate['id'], 'version': 1,
            'kind': candidate['kind'], 'source_count': len(candidate['source_ids']),
            'encoding_gain_bytes': candidate.get('encoding_cost', {}).get('gain_bytes'),
            'active_library': candidate['active_library'],
            'source_reconstruction': 'exact', 'transfer_status': candidate['transfer_status']})
    result = {'status': computed['status'], 'library_path': LIBRARY_PATH,
        'frozen_candidates': references, 'episode_count': len(intake['episodes']),
        'source_count': len(_unique(entry['source_id'] for entry in episodes)),
        'conflict_count': len(library['conflicts']),
        'support_source_count': len(library['unique_support_sources']),
        'excluded': computed['excluded'], 'author_work': computed['author_work'],
        'parameter_relation_searches': computed.get('parameter_relation_searches', []),
        'subtree_contract_recovery': computed.get('subtree_contract_recovery', []),
        'budget_exhausted': computed['budget_exhausted'],
        'cache_hit': False, 'new_compute_cost': computed['actual_compute_cost'],
        'support_updated': False, 'new_observations': 0, 'llm_calls': 0}
    library['executions'][theme] = {'request_basis': request_basis,
        'allowance': allocation['compute'], 'result': deepcopy(result)}
    _save(agent)
    return _consolidation_return(question, theme, result, allocation['compute'])


def _library_reference(agent, path):
    reference = _absolute_reference({'file': path, 'json_path': []}, agent.workspace)
    if (agent.workspace / path).resolve().is_relative_to(agent.workspace.resolve()):
        reference['workspace'] = str(agent.workspace.resolve())
    return reference


def read_theory_library(agent, path=LIBRARY_PATH, *, source_libraries=None, source_documents=None):
    """Read the selected frozen source without replacing the working library."""
    if path == LIBRARY_PATH:
        return agent.theory_consolidation
    return _read_library(_library_reference(agent, path), agent.workspace,
                         source_libraries=source_libraries, source_documents=source_documents)


def _retain_library_candidate_versions(agent, reference, library, documents):
    # Installation already reads these exact source layers. Keep only the small
    # ordered roster; the complete immutable payload remains in the existing base.
    files = {}
    for filename, *_ in documents:
        state = Path(filename).stat()
        files[filename] = (state.st_mtime_ns, state.st_size)
    agent._theory_library_candidate_versions = {
        'reference': (reference['file'], tuple(_path(reference))), 'files': files,
        'versions': [{'id': identifier, 'version': candidate.get('version', 1)}
                     for identifier, candidate in library.get('candidates', {}).items()]}


def theory_library_candidate_versions(agent, path=LIBRARY_PATH):
    """Read routing metadata without repeatedly expanding an installed source.

    Local working objects remain live. Only the installed source's own roster is
    reused, and a changed thin view or source file is read through the old reader.
    """
    if path != LIBRARY_PATH:
        reference = _library_reference(agent, path)
        installed = getattr(agent, '_theory_library_source_ref', None)
        location = (reference['file'], tuple(_path(reference)))
        if installed and location == (installed['file'], tuple(_path(installed))):
            retained = getattr(agent, '_theory_library_candidate_versions', None)
            if retained and retained['reference'] == location:
                unchanged = True
                for filename, previous in retained['files'].items():
                    state = Path(filename).stat()
                    if (state.st_mtime_ns, state.st_size) != previous:
                        unchanged = False
                        break
                if unchanged:
                    return [dict(item) for item in retained['versions']]
            documents = {}
            library = read_theory_library(agent, path, source_documents=documents)
            _retain_library_candidate_versions(agent, reference, library, documents)
            return [dict(item) for item in agent._theory_library_candidate_versions['versions']]
    return [{'id': identifier, 'version': candidate.get('version', 1)}
            for identifier, candidate in read_theory_library(agent, path).get('candidates', {}).items()]


def _same_library_candidate(agent, previous_path, current_path, identifier, version, libraries):
    """A relocated library must still contain the exact frozen constructor."""
    if libraries is None:
        return False
    for path in (previous_path, current_path):
        if path not in libraries:
            absolute = _library_reference(agent, path)['file']
            libraries[path] = (libraries[absolute] if absolute in libraries else
                read_theory_library(agent, path, source_libraries=libraries))
    old = libraries[previous_path].get('candidates', {}).get(identifier)
    current = libraries[current_path].get('candidates', {}).get(identifier)
    return (old is not None and current is not None
        and old.get('id') == current.get('id') == identifier
        and old.get('version') == current.get('version') == version and old == current)


def _retained_binding_failures(agent, library, path, binding, *, documents=None):
    if path == LIBRARY_PATH:
        return []
    proxy = SimpleNamespace(workspace=agent.workspace, theory_consolidation=library,
        _theory_library_base=library, _theory_library_source_ref=_library_reference(agent, path))
    wanted = {key: value for key, value in binding.items() if key != 'library_path'}
    # Recall already retained each layer's original transfer list. Direct
    # transfers share a single document map across their own failure checks.
    documents = {} if documents is None else documents
    references = []
    for index, record in enumerate(library.get('transfers', [])):
        retained = {key: value for key, value in record['binding'].items() if key != 'library_path'}
        if record['status'] != 'refuted' or retained != wanted:
            continue
        if record.get('paid_mse_check', {}).get('status') == 'failed':
            references.append(library_entry_reference(proxy, 'transfers', index, 'paid_mse_check', documents=documents))
        for position, check in enumerate(record.get('checks', [])):
            if check['status'] == 'failed' and check.get('context_ref') == binding['context_ref']:
                references.append(library_entry_reference(proxy, 'transfers', index, 'checks', position, documents=documents))
        references.extend(deepcopy(record.get('retained_counterexample_refs', [])))
        references.extend(deepcopy(record.get('scope_refutation', {}).get('counterexample_refs', [])))
    return _unique(references)


def transfer_theory(agent, arguments, laboratory=None):
    path = arguments.get('library_path', LIBRARY_PATH)
    library = arguments.get('_source_library')
    if library is None:
        library = install_theory_library(agent) if path == LIBRARY_PATH else read_theory_library(agent, path)
    if path == LIBRARY_PATH:
        agent.theory_consolidation = library
    candidate = library['candidates'][arguments['candidate_id']]
    if candidate['kind'] in {'program_context_relation', 'program_revision_relation'}:
        return _transfer_relation(agent, arguments, candidate, path, source_library=library)
    if candidate['kind'] == 'shared_subexpression':
        return {'status': 'author_work_request', 'reason': 'parent_composition_required_for_shared_fragment'}
    template = AbstractTemplate.from_dict(candidate['template'])
    mapping = arguments['mapping']
    target_roles = arguments['role_signature']
    if set(mapping) != set(template.roles) or len(set(mapping.values())) != len(mapping):
        return {'status': 'author_work_request', 'reason': 'complete_injective_role_mapping_required'}
    for role, name in mapping.items():
        if target_roles.get(name) != candidate['role_signature'][role]:
            return {'status': 'author_work_request', 'reason': 'tested_role_unit_time_mapping_required'}
    if arguments['output_unit'] != candidate['output_unit']:
        return {'status': 'author_work_request', 'reason': 'executable_output_conversion_required'}
    parameters = arguments.get('parameters', candidate['instances'][0]['parameters'])
    context = arguments['context_ref']
    program = template.instantiate(mapping, parameters)
    binding = {'library_path': path, 'candidate_id': candidate['id'], 'candidate_version': candidate['version'],
               'mapping': deepcopy(mapping), 'parameters': list(parameters), 'context_ref': deepcopy(context)}
    return _checked_transfer(agent, arguments, candidate, program, binding,
        compute_allowance=arguments.get('allocation', {}).get('compute'),
        retained_counterexample_refs=_retained_binding_failures(agent, library, path, binding,
            documents=arguments.get('_source_documents')))


def _recall_work_groups(agent, candidates, library, retained_inputs, *, prefix=None):
    """Equal executable problems share a slice; scientific objects stay intact."""
    from .revision_relations import _revision_contract

    groups, contracts = [], []
    for identifier, candidate in candidates.items():
        if (candidate['kind'] not in {'program_context_relation', 'program_revision_relation'}
                or not candidate.get('definition')):
            groups.append([identifier])
            contracts.append(None)
            continue
        contract = _revision_contract(candidate)
        contract.update({key: candidate.get(key) for key in (
            'kind', 'purpose', 'source_change_kind', 'structural_change', 'parameter_relations',
            'blocked_transfer_bindings')})
        contract['definition']['definitions'] = candidate['definition'].get('definitions')
        contract['source_discrepancy'] = any(check['before']['failed']
                                           for check in candidate.get('trigger_checks', []))
        # Failed bindings affect admission even when source Programs agree.
        # Keep original records; only remove object names from this comparison.
        failures = []
        for source in (agent.theory_consolidation, library):
            for record in source.get('transfers', []):
                # Outcomes of this very search are shared paid work, not a
                # new object-specific applicability condition. Otherwise its
                # first failed fit would split its untouched aliases next time.
                if (source is agent.theory_consolidation and prefix is not None
                        and record.get('theory_id', '').startswith(prefix)):
                    continue
                binding = record.get('binding', {})
                if binding.get('candidate_id') == identifier and record.get('status') == 'refuted':
                    failure = {key: value for key, value in record.items()
                               if key not in {'theory_id', 'theory_ref', 'binding'}}
                    failure['binding'] = {key: value for key, value in binding.items()
                                          if key not in {'candidate_id', 'candidate_version'}}
                    if failure not in failures:
                        failures.append(failure)
        contract['failed_bindings'] = failures
        contract['retained_inputs'] = retained_inputs.get(identifier)
        index = next((index for index, old in enumerate(contracts)
                      if old is not None and old == contract), None)
        if index is None:
            groups.append([identifier])
            contracts.append(contract)
        else:
            groups[index].append(identifier)
    return groups


def recall_frozen_theories(agent, question, arguments):
    """Use the existing transfer operators on a typed new question, within budget.

    The current component pool is frozen before this pass; newly proposed
    Programs do not recursively become components of their own recall work.
    """
    from .question_routing import resolved_theories

    comparison_rows = arguments.get('allocation', {}).get('comparison_rows', 0)
    if type(comparison_rows) is not int or comparison_rows < 0:
        raise ValueError('comparison_rows must be a nonnegative integer')
    path = arguments.get('library_path', question.get('frozen_library_path', LIBRARY_PATH))
    # Old request bases keep their source paths. Resolve a relocated source
    # once per recall, sharing the already loaded current library with transfers.
    source_libraries, source_documents = {}, {}
    selected_library = read_theory_library(agent, path, source_libraries=source_libraries,
                                           source_documents=source_documents)
    source_libraries[path] = selected_library
    candidates = selected_library.get('candidates', {})
    if not candidates:
        return {'status': 'no_frozen_candidates', 'generated_theory_ids': [], 'new_compute_cost': 0}
    mission = effective_question_mission(question, agent.research_mission)
    if mission.get('purpose') in {'theory_learning', 'method_learning'}:
        allocation = _allocation(agent, arguments, question)
        if allocation is None:
            return {'status': 'research_deferred', 'reason': 'theory_allocation_missing',
                    'generated_theory_ids': [], 'new_compute_cost': 0}
    else:
        remaining = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
        requested = arguments.get('allocation', {}).get('compute', remaining)
        if type(requested) is not int or requested < 0:
            raise ValueError('recall allocation needs nonnegative integer compute')
        allocation = {'compute': min(requested, remaining)}
    target = question.get('theory_target') or {}
    prefix = f'frozen_recall:{question["id"]}@{question.get("meaning_version", 1)}:'
    retained_components = arguments.get('_retained_components')
    resolved = (retained_components if retained_components is not None else resolved_theories(agent))
    resolved_pool = {theory['id']: theory for theory in resolved}
    parent_id = arguments.get('parent_theory_id', target.get('parent_theory_id'))
    parent_snapshot = (deepcopy(arguments['_retained_parent']) if '_retained_parent' in arguments
                       else next((deepcopy(item) for item in resolved if item['id'] == parent_id), None))
    if parent_snapshot is not None:
        resolved_pool[parent_snapshot['id']] = parent_snapshot
    components = (list(resolved) if retained_components is not None else
                  [item for item in resolved if not item['id'].startswith(prefix)])
    checks = deepcopy(target.get('evidence_checks', arguments.get('evidence_checks', [])))
    basis = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1),
        'target': deepcopy(target), 'checks': checks, 'library_path': path,
        'parent_theory_id': arguments.get('parent_theory_id', target.get('parent_theory_id')),
        'fitting': deepcopy(arguments.get('fitting', target.get('revision_fitting'))),
        'method_ref': 'hegel_machine.cognition.abstractions.AbstractTemplate.mappings',
        'candidates': [{'id': key, 'version': item['version']} for key, item in candidates.items()],
        'components': [{key: deepcopy(item.get(key)) for key in (
            'id', 'version', 'source_ref', 'source_status', 'compatible')} for item in components]}
    if comparison_rows:
        basis['comparison_rows'] = comparison_rows
    if parent_id is not None:
        basis['parent'] = ({key: deepcopy(parent_snapshot.get(key)) for key in
            ('id', 'version', 'source_ref', 'source_status')} if parent_snapshot else None)
    def retained_work():
        result = []
        for identifier in candidates:
            execution = agent.theory_consolidation.get('executions', {}).get(
                f'relation_transfer:{prefix}{identifier}')
            if execution is None:
                continue
            cursor = execution.get('cursor') or {}
            result.append({'candidate_id': identifier,
                'search_completed': execution.get('search_completed', True),
                'layer': cursor.get('layer'), 'offset': cursor.get('offset'),
                'pending_fits': deepcopy(cursor.get('pending_fits', []))})
        return result

    basis['retained_work'] = retained_work()
    previous = question.get('frozen_recall_execution')
    same_numeric_work = previous and previous['basis'] == basis
    # Coverage is independent of numerical freshness. New paid rows, parents
    # and component versions still invalidate each transfer's actual work, but
    # do not repeatedly put the first frozen object ahead of all the others.
    def coverage_scope(value):
        typed = value.get('target') or {}
        return {key: value.get(key) for key in
                ('question_id', 'meaning_version', 'library_path')} | {
            'typed_scope': {key: typed.get(key) for key in
                ('role_signature', 'output_unit', 'context_ref', 'context_family_ref')}}

    execution_inputs = arguments.get('_retained_execution_inputs', {})
    groups = _recall_work_groups(agent, candidates, selected_library, execution_inputs, prefix=prefix)
    order = [group[0] for group in groups]
    groups_by_owner = {group[0]: group for group in groups}
    previous_coverage = coverage_scope(previous['basis']) if previous else None
    current_coverage = coverage_scope(basis)
    if previous and previous_coverage['library_path'] is not None and previous_coverage['library_path'] != path:
        shared = [item for item in previous['basis']['candidates'] if item['id'] in candidates]
        if shared and all(_same_library_candidate(agent, previous_coverage['library_path'], path,
                item['id'], item['version'], source_libraries) for item in shared):
            previous_coverage = {**previous_coverage, 'library_path': path}
    same_coverage = previous and previous_coverage == current_coverage
    same_coverage = same_coverage and all(item['id'] not in candidates or
        candidates[item['id']]['version'] == item['version'] for item in previous['basis']['candidates'])
    saved_cursor = previous.get('cursor', {}) if same_coverage else {}
    next_id = saved_cursor.get('next_candidate_id')
    prior_groups = (previous['result'].get('execution_work_groups', []) if same_coverage else [])
    previous_versions = {item['id']: item['version'] for item in
                         previous['basis']['candidates']} if previous else {}
    for group in prior_groups:
        previous_versions.update({item['id']: item['version']
                                  for item in group.get('execution_candidate_refs', [])})
    if next_id not in candidates and prior_groups:
        old_first = next((index for index, group in enumerate(prior_groups)
                          if next_id in group['candidate_ids']), 0)
        next_id = next((identifier for group in [*prior_groups[old_first:], *prior_groups[:old_first]]
                        for identifier in group['candidate_ids'] if identifier in candidates), None)
    # A removed name cannot erase work still represented by an unchanged alias.
    execution_candidates = dict(candidates)
    execution_ids = {group[0]: list(group) for group in groups}
    retained_execution_keys = {}
    for group in groups:
        for old in prior_groups:
            if not any(identifier in group and candidates[identifier]['version'] ==
                       previous_versions.get(identifier) for identifier in old['candidate_ids']):
                continue
            if old.get('execution_key'):
                retained_execution_keys[group[0]] = old['execution_key']
            old_ids = [item['id'] for item in old.get('execution_candidate_refs', [])] or old['candidate_ids']
            for identifier in old_ids:
                if identifier not in candidates and identifier not in execution_ids[group[0]]:
                    execution_ids[group[0]].append(identifier)
                    execution_candidates[identifier] = {'version': previous_versions[identifier]}
    first = next((index for index, group in enumerate(groups) if next_id in group), 0)
    ordered = [*order[first:], *order[:first]]
    object_cursor = {'next_candidate_id': ordered[0]}
    if (previous and previous['basis'] == basis and
            previous['result'].get('search_completed', previous['result']['status'] != 'UNKNOWN') and
            (previous['result']['status'] != 'UNKNOWN' or allocation['compute'] <= previous['allowance'])):
        return {**deepcopy(previous['result']), 'cache_hit': True, 'new_compute_cost': 0,
                'reserved_comparison_compute': 0}
    start = agent.costs['compute_cost']
    budget = SearchBudget(allocation['compute'])
    generated, work, author_work, excluded = [], [], [], []
    serviced, examined, pending_objects = [], [], []
    unfunded_objects = []
    initial_unfunded_objects, residual_reoffers = [], []
    deferred_objects = []
    execution_keys = dict(retained_execution_keys)
    complete_visits = []
    completed_visit_owner = None
    def retained_execution(identifier):
        executions = agent.theory_consolidation.get('executions', {})
        key = execution_keys.get(identifier, f'relation_transfer:{prefix}{identifier}')
        return executions.get(key, {})

    complete_visit_mode = (arguments.get('fitting', target.get('revision_fitting')) is not None
        and any(candidate['kind'] == 'program_revision_relation'
                for identifier, candidate in candidates.items()))
    object_budget = None
    preparation_required = 1
    incomplete = False

    def charge(units=1):
        nonlocal preparation_required
        if not budget.can_spend(units) or not object_budget.can_spend(units):
            preparation_required = max(preparation_required, object_budget.spent + units)
            return False
        budget.spend(units)
        object_budget.spend(units)
        agent.costs['compute_cost'] += units
        return True

    def transfer_inputs(inputs, identifier):
        inputs['library_path'] = path
        inputs['allocation'] = {'compute': object_budget.remaining,
                                **({'comparison_rows': comparison_rows} if comparison_rows else {})}
        # Reuse this recall's inputs only across cache hits. An actual transfer
        # may add versions or counterexamples which the next object must read.
        inputs['_source_library'] = (agent.theory_consolidation if path == LIBRARY_PATH
                                     else selected_library)
        inputs['_source_libraries'] = source_libraries
        inputs['_source_documents'] = source_documents
        retained = execution_inputs.get(identifier)
        if retained is not None:
            inputs.update(retained)
        elif retained_components is not None:
            inputs['_retained_components'] = components
            inputs['_retained_parent'] = parent_snapshot
        if '_grammar_preparation' in arguments:
            # The scene validates this shared pure pool once. Keep it in RAM,
            # separate from each parent's execution cursor and fitting work.
            inputs['_grammar_preparation'] = arguments['_grammar_preparation']
        if '_retained_fit_constructors' in arguments:
            inputs['_retained_fit_constructors'] = arguments['_retained_fit_constructors']
        if '_retained_components' in inputs:
            inputs['_resolved_pool'] = {item['id']: item for item in inputs['_retained_components']}
            if inputs.get('_retained_parent') is not None:
                inputs['_resolved_pool'][inputs['_retained_parent']['id']] = inputs['_retained_parent']
            inputs['_parent_snapshot'] = deepcopy(inputs.get('_retained_parent'))
        elif resolved_pool is not None:
            inputs['_resolved_pool'] = resolved_pool
        inputs['_execution_candidate_ids'] = execution_ids[identifier]
        inputs['_execution_candidates'] = execution_candidates

    def clear_transfer_inputs(inputs):
        for key in ('_source_library', '_source_libraries', '_source_documents', '_resolved_pool',
                    '_grammar_preparation', '_retained_fit_constructors'):
            inputs.pop(key, None)

    def prepare_visit(inputs, identifier):
        transfer_inputs(inputs, identifier)
        inputs['_prepare_complete_visit'] = True
        try:
            prepared = transfer_theory(agent, inputs)
        finally:
            inputs.pop('_prepare_complete_visit', None)
            clear_transfer_inputs(inputs)
        budget.spend(prepared['new_compute_cost'])
        object_budget.spend(prepared['new_compute_cost'])
        if '_prepared_relation_visit' in prepared:
            inputs['_prepared_relation_visit'] = prepared['_prepared_relation_visit']
        return prepared

    def apply(inputs, identifier, *, mapping=None, binding_index=None):
        nonlocal incomplete, resolved_pool
        transfer_inputs(inputs, identifier)
        try:
            applied = transfer_theory(agent, inputs)
        finally:
            clear_transfer_inputs(inputs)
        if retained_components is None and (not applied.get('cache_hit') or resolved_pool is not None and (
                len(resolved_pool) != len(agent.theory_index)
                or any(identifier not in resolved_pool
                       or theory.get('version', 1) != resolved_pool[identifier].get('version', 1)
                       for identifier, theory in agent.theory_index.items()))):
            resolved_pool = None
        budget.spend(applied['new_compute_cost'])
        budget.reserve += applied.get('reserved_comparison_compute', 0)
        object_budget.spend(applied['new_compute_cost'])
        object_budget.reserve += applied.get('reserved_comparison_compute', 0)
        entries = applied.get('generated_theories', [applied])
        item = {'candidate_id': identifier, 'status': applied['status'],
            'theory_ids': [entry['theory_id'] for entry in entries if entry.get('theory_id')],
            'new_compute_cost': applied['new_compute_cost']}
        if len(groups_by_owner[identifier]) > 1:
            item['shared_candidate_ids'] = list(groups_by_owner[identifier])
        execution_keys[identifier] = applied.get('execution_key', f'relation_transfer:{prefix}{identifier}')
        if applied.get('reserved_comparison_compute'):
            item['reserved_comparison_compute'] = applied['reserved_comparison_compute']
        if applied.get('reason'):
            item['reason'] = applied['reason']
        author_work.extend({**deepcopy(request), 'candidate_id': identifier}
                           for request in applied.get('author_work', []))
        if mapping is not None:
            item.update(mapping=deepcopy(mapping), binding_index=binding_index)
        work.append(item)
        generated.extend(entry['theory_id'] for entry in entries
                         if entry.get('status') == 'candidate_registered')
        unfinished = not applied.get('search_completed', applied['status'] != 'UNKNOWN')
        incomplete = incomplete or unfinished
        if unfinished:
            pending_objects.append(identifier)
        return applied

    roles = target.get('role_signature')
    if (not isinstance(roles, dict) or target.get('output_unit') is None
            or target.get('context_ref') is None
            or any(not isinstance(contract, dict) or any(key not in contract for key in ('role', 'unit', 'time'))
                   for contract in roles.values())):
        author_work.append({'type': 'AuthorWorkRequest', 'reason': 'typed_target_binding_required',
                            'required': ['role_signature', 'output_unit', 'context_ref']})
    else:
        scheduled = [(position, identifier, None) for position, identifier in enumerate(ordered)]
        for position, identifier, deferred in scheduled:
            candidate = candidates[identifier]
            complete_revision = complete_visit_mode and candidate['kind'] == 'program_revision_relation'
            prior = retained_execution(identifier)
            if deferred is not None:
                required = deferred['required_compute']
                if budget.remaining < required:
                    residual_reoffers.append({'candidate_id': identifier,
                        'status': 'minimum_still_unfunded', 'required_compute': required,
                        'available_compute': budget.remaining, 'new_compute_cost': 0})
                    continue
                allowance = max(required, budget.remaining // (len(deferred_objects) - position))
                unfunded_objects[:] = [item for item in unfunded_objects
                                       if item['candidate_id'] != identifier]
                pending_objects[:] = [item for item in pending_objects if item != identifier]
            else:
                allowance = (budget.remaining if complete_revision else
                             budget.remaining // (len(ordered) - position))
            object_budget = SearchBudget(allowance)
            object_cursor['next_candidate_id'] = identifier
            before_work = len(work)
            preparation_exhausted = False
            preparation_required = 1
            try:
                if not charge():
                    preparation_exhausted = True
                    incomplete = True
                    unfunded_objects.append({'candidate_id': identifier,
                        'required_compute': 1, 'allocated_compute': object_budget.limit})
                    continue
                examined.append(identifier)
                inputs = {'candidate_id': identifier, 'theory_id': prefix + identifier,
                    'role_signature': deepcopy(roles), 'output_unit': target['output_unit'],
                    'context_ref': deepcopy(target['context_ref']), 'evidence_checks': checks}
                if target.get('context_family_ref') is not None:
                    inputs['context_family_ref'] = deepcopy(target['context_family_ref'])
                if candidate['output_unit'] != target['output_unit']:
                    excluded.append({'candidate_id': identifier, 'reason': 'output_unit_mismatch'})
                    continue
                retained = execution_inputs.get(identifier)
                object_components = (retained.get('_retained_components', components)
                                     if retained is not None else components)
                if candidate['kind'] in {'program_context_relation', 'program_revision_relation'}:
                    preparation_required = (1 + len(candidate.get('fixed_role_signature', {}))
                                            + len(object_components))
                    prior = retained_execution(identifier)
                    minimum = prior.get('result', {}).get('minimum_compute_for_progress', 0)
                    same_object = same_numeric_work
                    if retained is not None and prior.get('request_basis'):
                        retained_args = {**inputs, **retained,
                            'parent_theory_id': arguments.get('parent_theory_id', target.get('parent_theory_id')),
                            'fitting': arguments.get('fitting', target.get('revision_fitting')),
                            'allocation': {'comparison_rows': comparison_rows} if comparison_rows else {}}
                        current_basis = _relation_request_basis(retained.get('_retained_parent'),
                            object_components, retained_args, candidate, path)
                        same_object = _same_relation_work(agent, prior['request_basis'], current_basis,
                                                         source_libraries=source_libraries)
                        saved_fits = _available_retained_fit_state(agent, candidate,
                                                                  retained.get('_retained_parent'))
                        if saved_fits is not None and minimum:
                            prior_result = prior.get('result', {})
                            minimum -= max(0, prior_result.get('retained_fit_recovery_compute', 0)
                                           - saved_fits['selection_compute'])
                            if saved_fits.get('index') is not None:
                                index_compute = prior_result.get('retained_fit_index_compute')
                                if index_compute is None and prior_result.get('retained_fit_reuse_count'):
                                    # Saved pre-cache executions built the same
                                    # complete index before restoring a fit.
                                    index_compute = saved_fits.get('index_compute', 0)
                                minimum -= index_compute or 0
                            minimum = max(1, minimum)
                    preparation = len(candidate.get('fixed_role_signature', {})) + len(object_components)
                    if (not complete_revision and same_object and minimum
                            and object_budget.remaining < preparation + minimum):
                        preparation_exhausted = True
                        incomplete = True
                        unfunded_objects.append({'candidate_id': identifier,
                            'required_compute': 1 + preparation + minimum,
                            'allocated_compute': object_budget.limit})
                        continue
                if candidate['kind'] == 'typed_program_family':
                    template = AbstractTemplate.from_dict(candidate['template'])
                    compatible_mappings, exhausted = 0, False
                    for index, mapping in enumerate(template.mappings(sorted(roles))):
                        if not charge():
                            exhausted = True
                            break
                        if (set(mapping) != set(template.roles)
                                or len(set(mapping.values())) != len(mapping)):
                            excluded.append({'candidate_id': identifier, 'mapping': deepcopy(mapping),
                                             'reason': 'complete_injective_role_mapping_required'})
                            continue
                        compatible = True
                        for role, name in mapping.items():
                            if not charge():
                                exhausted = True
                                break
                            if roles.get(name) != candidate['role_signature'][role]:
                                compatible = False
                        if exhausted:
                            break
                        if not compatible:
                            excluded.append({'candidate_id': identifier, 'mapping': deepcopy(mapping),
                                             'reason': 'role_unit_time_contract_mismatch'})
                            continue
                        compatible_mappings += 1
                        if not charge(template.pattern.size):
                            exhausted = True
                            break
                        bound = {**inputs, 'mapping': mapping,
                            'theory_id': inputs['theory_id'] if index == 0 else
                                         f'{inputs["theory_id"]}:binding:{index}'}
                        apply(bound, identifier, mapping=mapping, binding_index=index)
                    if exhausted:
                        preparation_exhausted = True
                        incomplete = True
                        continue
                    if not compatible_mappings:
                        author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                            'reason': 'typed_injective_role_mapping_required'})
                    continue
                elif candidate['kind'] in {'program_context_relation', 'program_revision_relation'}:
                    if candidate['kind'] == 'program_revision_relation':
                        inputs['parent_theory_id'] = arguments.get('parent_theory_id', target.get('parent_theory_id'))
                        inputs['_parent_snapshot'] = deepcopy(parent_snapshot)
                        if arguments.get('fitting', target.get('revision_fitting')) is not None:
                            inputs['fitting'] = deepcopy(arguments.get('fitting', target.get('revision_fitting')))
                        if not inputs['parent_theory_id']:
                            author_work.append({'candidate_id': identifier, **deepcopy(
                                target.get('parent_binding_obligation') or {
                                    'type': 'AuthorWorkRequest', 'reason': 'actual_revision_parent_required'})})
                            continue
                    fixed_roles = candidate.get('fixed_role_signature', {})
                    if not charge(len(fixed_roles)):
                        preparation_exhausted = True
                        incomplete = True
                        continue
                    if any(roles.get(name) != contract for name, contract in fixed_roles.items()):
                        author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                            'reason': 'fixed_context_role_binding_required'})
                        continue
                    if not charge(len(object_components)):
                        preparation_exhausted = True
                        incomplete = True
                        continue
                    inputs['component_theory_ids'] = [item['id'] for item in object_components
                        if item.get('program') and item.get('context_ref') == target['context_ref']]
                    if not inputs['component_theory_ids'] and candidate['kind'] != 'program_revision_relation':
                        author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                            'reason': 'typed_target_components_required'})
                        continue
                else:
                    author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                        'reason': 'parent_composition_required_for_shared_fragment'})
                    continue
                if complete_revision:
                    outer_preparation = object_budget.spent
                    prepared = prepare_visit(inputs, identifier)
                    quote = prepared.get('quote') or {}
                    visit = {'candidate_id': identifier,
                        'search_position': deepcopy(prepared.get('search_position')),
                        'outer_preparation_compute': outer_preparation,
                        'preparation_compute': prepared['new_compute_cost'],
                        'minimum_resources': {'compute_cost': object_budget.spent + quote.get('compute', 0)},
                        'quote': deepcopy(quote)}
                    complete_visits.append(visit)
                    if prepared.get('cache_hit'):
                        visit.update(status='cached_complete', new_compute_cost=object_budget.spent)
                        continue
                    if prepared.get('prepared_visit') is not None:
                        minimum = {'compute_cost': quote['compute']}
                        offered = allocate_complete_research(minimum, [],
                            {'compute_cost': budget.remaining})
                        if offered is None:
                            visit.update(status='contract_unaffordable',
                                available_compute=budget.remaining, new_compute_cost=object_budget.spent)
                            pending_objects.append(identifier)
                            continue
                        visit['allocated_resources'] = offered
                        # A complete visit buys this binding only. Surplus is
                        # returned to the outer research loop, never its frontier.
                        object_budget.limit = object_budget.spent + quote['compute']
                    else:
                        result = prepared.get('result') or {}
                        if result.get('budget_exhausted'):
                            # Keep paid grammar preparation and its real cursor;
                            # the prepared result contains no fitting outputs.
                            apply(inputs, identifier)
                            visit.update(status='preparation_incomplete',
                                new_compute_cost=object_budget.spent)
                            pending_objects.append(identifier)
                            continue
                    applied = apply(inputs, identifier)
                    completed = applied.get('completed_research_unit', False)
                    if prepared.get('prepared_visit') is None and not completed:
                        visit.update(status='frontier_exhausted' if applied.get('search_completed')
                            and not applied.get('reason') else 'no_applicable_unit',
                            reason=applied.get('reason'), new_compute_cost=object_budget.spent)
                        continue
                    visit.update(status='complete_visit_executed' if completed else 'complete_visit_incomplete',
                        new_compute_cost=object_budget.spent,
                        reserved_comparison_compute=object_budget.reserve)
                    completed_visit_owner = identifier
                    break
                apply(inputs, identifier)
            finally:
                if preparation_exhausted:
                    pending_objects.append(identifier)
                    if len(work) == before_work:
                        unfunded = next((item for item in unfunded_objects
                            if item['candidate_id'] == identifier), None)
                        if unfunded is None:
                            unfunded = {'candidate_id': identifier,
                                'required_compute': preparation_required,
                                'allocated_compute': object_budget.limit}
                            unfunded_objects.append(unfunded)
                        if deferred is None:
                            deferred_objects.append(deepcopy(unfunded))
                # A residual slice that cannot prepare the next object leaves
                # it first next time. A full first slice must still yield to
                # later objects, even if this object's minimum work is unfunded.
                if not preparation_exhausted or len(work) > before_work or position == 0:
                    serviced.append(identifier)
                    index = order.index(identifier)
                    object_cursor['next_candidate_id'] = order[(index + 1) % len(order)]
                if deferred is not None:
                    residual_reoffers.append({'candidate_id': identifier,
                        'status': 'transfer_called' if len(work) > before_work else
                                  'preparation_incomplete' if preparation_exhausted else 'checked',
                        'allocated_compute': object_budget.limit,
                        'required_compute': deferred['required_compute'],
                        'new_compute_cost': object_budget.spent,
                        'reserved_comparison_compute': object_budget.reserve})
                elif position == len(ordered) - 1:
                    # Only never-dispatched objects can reclaim the same pass's
                    # residual. Already attempted research keeps its one slice.
                    initial_unfunded_objects = deepcopy(unfunded_objects)
                    scheduled.extend((index, item['candidate_id'], item)
                                     for index, item in enumerate(deferred_objects))
        # Each call offers a bounded slice to every object. Unspent slices pass
        # forward; rotate the first position across calls, including when an
        # object's known atomic work cannot fit its current share.
        object_cursor['next_candidate_id'] = order[(order.index(completed_visit_owner) + 1) % len(order)] if (
            completed_visit_owner is not None) else order[(first + 1) % len(order)]
    unvisited = [identifier for identifier in ordered if identifier not in serviced]
    pending_objects = [alias for identifier in _unique([*pending_objects, *unvisited])
                       for alias in groups_by_owner[identifier]]
    incomplete = bool(pending_objects)
    result = {'status': 'UNKNOWN' if incomplete or any(item['status'] == 'UNKNOWN' for item in work)
              or (excluded and not work and not author_work)
              else 'author_work_request' if author_work and not work else 'completed',
        'generated_theory_ids': _unique(generated), 'work': work,
        'author_work': author_work, 'excluded': excluded,
        'object_cursor': deepcopy(object_cursor), 'examined_candidate_ids': examined,
        'pending_object_ids': pending_objects,
        'execution_work_groups': [{'candidate_ids': list(group),
            'execution_candidate_id': group[0], 'execution_key': execution_keys.get(
                group[0], f'relation_transfer:{prefix}{group[0]}'),
            'execution_candidate_refs': [{'id': identifier, 'version': execution_candidates[identifier]['version']}
                                         for identifier in execution_ids[group[0]]]}
            for group in groups],
        'unfunded_objects': unfunded_objects,
        'initial_unfunded_objects': initial_unfunded_objects,
        'residual_reoffers': residual_reoffers,
        'complete_research_visits': complete_visits,
        'search_completed': not incomplete and not unvisited,
        'new_compute_cost': agent.costs['compute_cost'] - start,
        'reserved_comparison_compute': budget.reserve,
        'cache_hit': False, 'support_updated': False, 'independent_confirmation': False}
    debit_mission_compute(agent, mission, start)
    # A separately completed pending fit changes available work even when the
    # next slice is smaller. Snapshot only its cursor metadata, not its trees.
    basis['retained_work'] = retained_work()
    question['frozen_recall_execution'] = {'basis': basis, 'allowance': allocation['compute'],
                                          'cursor': deepcopy(object_cursor), 'result': deepcopy(result)}
    return result


def _resolve_paid_checks(agent, examples):
    """Resolve explicitly declared paid-row references without persisting rows."""
    needed, references, retained = [], [], {}
    for index, example in enumerate(examples):
        criterion = example.get('criterion', {})
        if ('observation' not in example and example.get('source_ref')
                and criterion.get('kind') == 'paid_mse' and criterion.get('criterion_ref')):
            needed.append(index)
            references.extend((example['source_ref'], criterion['criterion_ref']))
    if references:
        try:
            values = resolve_program_sources(agent, references)
            retained = {index: values[2 * position:2 * position + 2]
                        for position, index in enumerate(needed)}
        except (KeyError, IndexError, OSError):
            # Preserve the existing per-example UNKNOWN behavior when one
            # retained address is unavailable; complete files read only once.
            pass
    result = []
    for index, example in enumerate(examples):
        item = deepcopy(example)
        if 'observation' in item:
            result.append(item)
            continue
        criterion = item.get('criterion', {})
        reason = 'retained_paid_row_and_mse_criterion_required'
        if (item.get('source_ref') and criterion.get('kind') == 'paid_mse'
                and criterion.get('criterion_ref')):
            try:
                observation, threshold = (retained[index] if index in retained else
                    resolve_program_sources(agent, [item['source_ref'], criterion['criterion_ref']]))
            except (KeyError, IndexError, OSError):
                observation, threshold = None, None
            n = criterion.get('evaluated_rows')
            if (isinstance(observation, dict) and 'features' in observation and 'target' in observation
                    and type(n) is int and n > 0 and type(threshold) in (int, float)
                    and isfinite(threshold) and threshold >= 0
                    and threshold == criterion.get('mse_threshold')):
                item.update(observation=deepcopy(observation), tolerance=sqrt(n * threshold))
            else:
                item['resolution_reason'] = reason
        else:
            item['resolution_reason'] = reason
        result.append(item)
    return result


def _paid_mse_result(examples, checks):
    """Consume the original aggregate gate using only already paid predictions."""
    aggregate = [item.get('criterion') for item in examples
                 if item.get('criterion', {}).get('kind') == 'paid_mse']
    if not aggregate:
        return None
    criterion = aggregate[0]
    result = {**deepcopy(criterion), 'status': 'UNKNOWN', 'scope': 'retained_paid_evaluation',
              'independent_confirmation': False}
    n = criterion.get('evaluated_rows')
    if (type(n) is not int or n <= 0 or len(examples) != n or len(checks) != n
            or len(aggregate) != n or any(item != criterion for item in aggregate)
            or len(_unique(item['source_ref'] for item in examples)) != n
            or any(check['status'] not in {'passed', 'failed'} for check in checks)):
        result['reason'] = 'complete_same_scope_paid_predictions_required'
        return result
    mse = sum((check['prediction'] - check['actual']) ** 2 for check in checks) / n
    result.update(mse=mse, status='passed' if mse <= criterion['mse_threshold'] else 'failed')
    return result


def _checked_transfer(agent, arguments, candidate, program, binding, *, compute_allowance=None,
                      retained_counterexample_refs=(), persist=True):
    from .question_routing import register_theories, scoped_numeric_refutation

    context, target_roles = arguments['context_ref'], arguments['role_signature']
    # A frozen failed binding cannot be made fresh by omitting its old checks.
    blocked = any(all(item.get(key) == binding.get(key) for key in ('mapping', 'parameters', 'context_ref'))
                  for item in candidate.get('blocked_transfer_bindings', []))
    preceding = [item for item in agent.theory_consolidation.get('transfers', []) if item['binding'] == binding]
    blocked = blocked or bool(retained_counterexample_refs) or any(item['status'] == 'refuted' for item in preceding)
    check_basis = deepcopy(arguments.get('evidence_checks', []))
    cached = next((item for item in reversed(preceding) if item.get('check_basis') == check_basis
                   and item['theory_id'] == arguments['theory_id']), None)
    available = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
    if compute_allowance is not None:
        available = min(available, max(0, compute_allowance))
    pending = cached and cached['status'] == 'UNKNOWN' and any(
        check['status'] == 'UNKNOWN' for check in cached['checks'])
    if cached and (not blocked or cached['status'] == 'refuted') and (not pending or not available):
        return {**deepcopy(cached), 'cache_hit': True, 'candidate_pool_changed': False,
                'new_compute_cost': 0, 'new_observations': 0, 'llm_calls': 0,
                'source_reconstruction_preserved': True}
    checks, spent, unknown = [], 0, False
    examples = _resolve_paid_checks(agent, arguments.get('evidence_checks', []))
    for index, example in enumerate(examples):
        if cached and cached['checks'][index]['status'] in {'passed', 'failed'}:
            checks.append(deepcopy(cached['checks'][index]))
            blocked = blocked or checks[-1]['status'] == 'failed'
            continue
        if example['context_ref'] != context:
            checks.append({'source_ref': deepcopy(example['source_ref']), 'status': 'UNKNOWN',
                           'reason': 'check_context_mismatch'})
            unknown = True
            continue
        if 'observation' not in example:
            checks.append({'source_ref': deepcopy(example.get('source_ref')), 'status': 'UNKNOWN',
                           'reason': example['resolution_reason']})
            unknown = True
            continue
        payload = example['observation']
        row = Observation(payload['features'], payload['target'], tuple(payload.get('history', [])))
        # Use the same budget-aware Program evaluator as ordinary question
        # prediction, preserving the actual public history binding.
        from .question_queries import evaluate_on_model
        expression = {'op': 'Readout', 'record_index': 0, 'step': 0,
            'binding': {'action': {'kind': 'retained_row'}, 'features': row.features,
                        'history': list(row.history)}}
        remaining = max(0, agent.config.compute_budget - agent.costs['compute_cost'])
        if compute_allowance is not None:
            remaining = min(remaining, max(0, compute_allowance - spent))
        answer = evaluate_on_model(expression, lambda value: program.predict(Observation(
            value['features'], 0., tuple(value['history']))), compute_budget=remaining)
        agent.costs['compute_cost'] += answer['compute_cost']
        spent += answer['compute_cost']
        check = {'source_ref': deepcopy(example['source_ref']), 'context_ref': deepcopy(context)}
        if answer['status'] != 'answered':
            checks.append({**check, 'status': 'UNKNOWN', 'reason': answer.get('reason')})
            unknown = True
            continue
        passed = abs(answer['value'] - row.target) <= example.get('tolerance', 1e-8)
        checks.append({**check, 'status': 'passed' if passed else 'failed',
            'prediction': answer['value'], 'actual': row.target, 'tolerance': example.get('tolerance', 1e-8)})
        blocked = blocked or not passed
    aggregate = _paid_mse_result(examples, checks)
    if aggregate:
        blocked = blocked or aggregate['status'] == 'failed'
        unknown = unknown or aggregate['status'] == 'UNKNOWN'
    identifier = arguments['theory_id']
    declaration = {'id': identifier, 'program': program.to_dict(),
        'family_id': candidate['id'], 'role_signature': deepcopy(target_roles),
        'output_unit': arguments['output_unit'], 'context_ref': deepcopy(context),
        'declared_scope_ref': deepcopy(context), 'parameter_paths': candidate.get('parameter_paths', []),
        'consolidation_ref': deepcopy(binding),
        'support': {'status': 'refuted' if blocked else 'candidate', 'evidence': [],
                    'context_ref': deepcopy(context)}, 'compatible': not blocked}
    if arguments.get('context_family_ref') is not None:
        declaration['context_family_ref'] = deepcopy(arguments['context_family_ref'])
    if 'subtree_contracts' in arguments:
        declaration['subtree_contracts'] = deepcopy(arguments['subtree_contracts'])
        declaration['subtree_contract_sources'] = deepcopy(arguments.get('subtree_contract_sources', {}))
    scoped_failure = scoped_numeric_refutation(agent, declaration)
    blocked = blocked or scoped_failure is not None
    if scoped_failure:
        declaration['compatible'] = False
        declaration['support']['status'] = 'refuted'
        declaration['scope_refutation'] = deepcopy(scoped_failure)
    old_version = agent.theory_index.get(identifier, {}).get('version')
    register_theories(agent, [declaration])
    status = 'refuted' if blocked else 'UNKNOWN' if unknown else 'candidate_registered'
    record = {'theory_id': identifier, 'theory_ref': {
                  'id': identifier, 'version': agent.theory_index[identifier]['version']},
              'binding': binding, 'status': status,
              'checks': checks, 'check_basis': check_basis,
              'support_updated': False, 'independent_confirmation': False}
    if aggregate:
        record['paid_mse_check'] = aggregate
    if retained_counterexample_refs:
        record['retained_counterexample_refs'] = deepcopy(retained_counterexample_refs)
    if scoped_failure:
        record['scope_refutation'] = deepcopy(scoped_failure)
    transfers = agent.theory_consolidation.setdefault('transfers', [])
    if record not in transfers:
        transfers.append(record)
    if persist:
        _save(agent)
    return {'status': status, 'theory_id': identifier, 'binding': binding, 'checks': checks,
            **({'paid_mse_check': aggregate} if aggregate else {}),
            **({'retained_counterexample_refs': deepcopy(retained_counterexample_refs)}
               if retained_counterexample_refs else {}),
            **({'scope_refutation': deepcopy(scoped_failure)} if scoped_failure else {}),
            'candidate_pool_changed': old_version != agent.theory_index[identifier]['version'], 'new_compute_cost': spent,
            'source_reconstruction_preserved': True, 'support_updated': False,
            'independent_confirmation': False, 'new_observations': 0, 'llm_calls': 0}


def _native_parent_components(agent, parent, available=()):
    """Expose already fitted siblings of this exact native parent by reference."""
    from .question_routing import register_theories
    reference = (parent or {}).get('source_ref', {})
    if reference.get('kind') != 'native_candidate' or not parent.get('program'):
        return []
    owner = next((question for question in agent.questions if question['id'] == reference['question_id']), None)
    meaning = reference.get('meaning_version', 1)
    if owner and owner.get('meaning_version', 1) != meaning:
        owner = next((old for old in owner.get('meaning_history', []) if old['meaning_version'] == meaning), None)
    if owner is None or owner.get('context_ref', parent['context_ref']) != parent['context_ref']:
        return []
    rounds = owner.get('native_condition_research', [])
    if not 0 <= reference['round_index'] < len(rounds):
        return []
    declarations, retained_ids = [], []
    round_ = rounds[reference['round_index']]
    registered = {item['theory_id']: item['version'] for item in
                  round_.get('native_theory_registration', {}).get('registered', [])}
    for index, source in enumerate(round_.get('candidates', [])):
        if index == reference['candidate_index'] or not source.get('program'):
            continue
        source_ref = {**deepcopy(reference), 'candidate_index': index}
        existing = next((item for item in available
            if item.get('source_ref') == source_ref and item.get('program') == source['program']
            and registered.get(item['id']) == item.get('version')), None)
        if existing:
            retained_ids.append(existing['id'])
            continue
        identifier = (f'native_component:{reference["question_id"]}@meaning{meaning}:'
                      f'round{reference["round_index"]}:candidate{index}')
        declarations.append({'id': identifier, 'source_ref': source_ref,
            **{key: deepcopy(parent[key]) for key in ('role_signature', 'output_unit', 'context_ref')},
            'support': {'status': 'candidate', 'evidence': []}})
    register_theories(agent, declarations)
    return [*retained_ids, *(item['id'] for item in declarations)]


def _relation_work_basis(basis):
    # Resource allowance is recorded, but does not change the pending problem.
    allocation = dict(basis.get('allocation') or {})
    allocation.pop('compute', None)
    result = {**basis, 'allocation': allocation}
    for key in ('parent', 'components'):
        values = result.get(key)
        if values is not None:
            cleaned = [{name: value for name, value in item.items()
                        if name not in {'compatible', 'source_status'}}
                       for item in (values if isinstance(values, list) else [values])]
            result[key] = cleaned if isinstance(values, list) else cleaned[0]
    return result


def _revision_check_contract(examples):
    return [{key: deepcopy(item.get(key)) for key in ('context_ref', 'tolerance')}
            | {'criterion': {key: deepcopy(value) for key, value in item.get('criterion', {}).items()
                             if key != 'criterion_ref'}} for item in examples]


def _same_relation_work(agent, previous, current, *, source_libraries=None, candidate_id=None):
    old, new = _relation_work_basis(previous), _relation_work_basis(current)
    if old.get('library_path') != new.get('library_path'):
        identifier = candidate_id or old.get('candidate_id')
        if (old.get('candidate_version') != new.get('candidate_version') or not
                _same_library_candidate(agent, old.get('library_path'), new.get('library_path'),
                    identifier, new.get('candidate_version'), source_libraries)):
            return False
        old = {**old, 'library_path': new['library_path']}
    if old == new:
        return True
    old_checks, new_checks = old.pop('evidence_checks', []), new.pop('evidence_checks', [])
    if old != new or len(old_checks or []) != len(new_checks or []):
        return False
    # A thin restore rebases addresses. Compare the resolved ordered problem,
    # while leaving both original request bases intact as provenance.
    resolved = _resolve_paid_checks(agent, [*(old_checks or []), *(new_checks or [])])
    if any('observation' not in item or item.get('resolution_reason') for item in resolved):
        return False
    values = []
    for item in resolved:
        value = deepcopy(item)
        # File relocation does not change a retained row's original position.
        # Keep its path even when two distinct observations have equal values.
        for reference in (value.get('source_ref'), value.get('criterion', {}).get('criterion_ref')):
            if reference is not None:
                reference.pop('file', None)
        values.append(value)
    return values[:len(old_checks or [])] == values[len(old_checks or []):]


def _shared_relation_progress(agent, arguments, candidate, basis, cache_key, executions):
    """Continue the furthest identical problem, retaining all original ledgers."""
    matches = []
    for identifier in arguments.get('_execution_candidate_ids', [candidate['id']]):
        key = (cache_key if identifier == candidate['id'] else
               cache_key.removesuffix(candidate['id']) + identifier)
        execution = executions.get(key)
        if not execution or not execution.get('request_basis'):
            continue
        peer = arguments.get('_execution_candidates', {}).get(identifier, candidate)
        if execution['request_basis'].get('candidate_version') != peer['version']:
            continue
        old = {**execution['request_basis'], 'candidate_id': candidate['id'],
               'candidate_version': candidate['version']}
        if _same_relation_work(agent, old, basis,
                source_libraries=arguments.get('_source_libraries'), candidate_id=identifier):
            matches.append((key, execution))
    if not matches:
        return None, []
    # Diagonal search positions are ordered progress, not a scientific score.
    key, selected = max(matches, key=lambda pair: (
        pair[1].get('search_completed', True),
        (pair[1].get('cursor') or {}).get('layer', 0),
        (pair[1].get('cursor') or {}).get('offset', 0),
        len((pair[1].get('cursor') or {}).get('fitting', {}).get('structures', []))))
    result = deepcopy(selected)
    result['execution_key'] = key
    if not result.get('search_completed', True) and result.get('cursor') is not None:
        pending = result['cursor'].setdefault('pending_fits', [])
        for _, execution in matches:
            for item in (execution.get('cursor') or {}).get('pending_fits', []):
                if item not in pending:
                    pending.append(deepcopy(item))
    return result, [key for key, _ in matches]


def _revision_fit_contract(value):
    from .revision_relations import _revision_contract

    contract = _revision_contract(value)
    contract.pop('parent_reference_relations')
    contract.update({key: deepcopy(value.get(key)) for key in (
        'kind', 'purpose', 'source_change_kind', 'structural_change',
        'parameter_relations', 'blocked_transfer_bindings')})
    contract['definition']['definitions'] = deepcopy(value['definition'].get('definitions'))
    return contract


def _revision_fit_source_versions(agent, candidate, versions, budget, *, source_library=None):
    """Share completed numerical work without sharing object search progress."""
    from .revision_relations import _spend

    result = dict(versions)
    contract = _revision_fit_contract(candidate)
    for library in (agent.theory_consolidation, source_library or {}):
        for identifier, peer in library.get('candidates', {}).items():
            if identifier in result:
                continue
            _spend(budget)
            if (peer.get('kind') == 'program_revision_relation' and peer.get('definition')
                    and _revision_fit_contract(peer) == contract):
                result[identifier] = peer['version']
    return result


def _retained_fit_cache_source(agent):
    library = agent.theory_consolidation
    transfers = library.get('transfers')
    return (id(library), id(transfers), len(transfers or []),
            tuple((key, item.get('version', 1)) for key, item in library.get('candidates', {}).items()))


def _available_retained_fit_state(agent, candidate, parent):
    """Peek at completed preparation when pricing an unchanged object's work."""
    cache = getattr(agent, '_retained_revision_fit_cache', None)
    if cache is None or parent is None or cache['source'] != _retained_fit_cache_source(agent):
        return None
    scope = {key: parent.get(key) for key in ('role_signature', 'output_unit', 'context_ref')}
    contract = _revision_fit_contract(candidate)
    return next((saved for saved in cache['queries']
        if saved['query']['versions'].get(candidate['id']) == candidate['version']
        and saved['query']['scope'] == scope and saved['query']['contract'] == contract), None)


def _retained_revision_fits(agent, candidate, parent, budget, *, execution_candidates=None,
                          source_library=None):
    """Read complete paid linear batches through their original constructors."""
    from .abstractions import AbstractTemplate
    from .language import Definition
    from .revision_relations import _Incomplete, _MissingContract, _match_parent, _assign_properties, _spend

    methods = ('initial', 'ridge_lstsq', 'stable_lstsq')
    grouped, documents, local_fields, rows_cache, records, versions = {}, {}, {}, {}, [], {}

    def version(identifier, number):
        if (identifier, number) in versions:
            return versions[identifier, number]
        indexed = agent.theory_index.get(identifier, {})
        item = next((item for item in (indexed, *indexed.get('history', []))
                     if item.get('version', 1) == number), None)
        if item is not None and not item.get('program'):
            item = _snapshot_index({identifier: {key: value for key, value in item.items()
                                               if key != 'history'}}, {'questions': agent.questions,
                'relation_research': agent.relation_research, 'events': agent.events},
                {'file': str(agent.workspace / 'research_state.json'),
                 'json_path': ['theory_index']})[identifier]
        versions[identifier, number] = item if item and item.get('program') else None
        return versions[identifier, number]

    recovery_started = budget.spent
    try:
        candidate_versions = _revision_fit_source_versions(agent, candidate,
            execution_candidates or {candidate['id']: candidate['version']}, budget,
            source_library=source_library)
        # Completed transfers are append-only during research. Keep the exact
        # typed query in memory; installing/removing a library or adding a fit
        # changes this source and requires a fresh recovery. Nothing is copied
        # into each object's cursor or checkpoint.
        selection_compute = budget.spent - recovery_started
        source = _retained_fit_cache_source(agent)
        cache = getattr(agent, '_retained_revision_fit_cache', None)
        if cache is None or cache['source'] != source:
            cache = agent._retained_revision_fit_cache = {'source': source, 'queries': []}
        query = {'versions': candidate_versions, 'contract': _revision_fit_contract(candidate),
                 'scope': {key: deepcopy(parent.get(key)) for key in
                           ('role_signature', 'output_unit', 'context_ref')}}
        for saved in cache['queries']:
            if saved['query'] == query:
                return saved['records']
        for index, transfer in enumerate(agent.theory_consolidation.get('transfers', [])):
            binding = transfer.get('binding', {})
            fit = binding.get('fitting', {})
            method, linear = fit.get('fitting_method'), fit.get('linear_diagnostics', {})
            if (binding.get('candidate_id') not in candidate_versions
                    or binding.get('candidate_version') != candidate_versions[binding['candidate_id']]
                    or binding.get('context_ref') != parent['context_ref']
                    or fit.get('search_completed') is not True or method not in methods
                    or linear.get('status') != 'computed'
                    or linear.get('evaluation_objective') != 'mean_squared_error'
                    or linear.get('ridge', {}).get('penalty') != 1e-9
                    or linear.get('stable', {}).get('penalty') != 0.):
                continue
            _spend(budget)
            identity = transfer.get('theory_ref', {})
            theory = version(identity.get('id'), identity.get('version', 1))
            if theory is None or any(theory.get(key) != parent.get(key) for key in
                                     ('role_signature', 'output_unit', 'context_ref')):
                continue
            reference = _absolute_reference(library_entry_reference(
                agent, 'transfers', index, documents=documents, local_fields=local_fields), agent.workspace)
            origin = _reference_workspace(reference) or agent.workspace
            bound = _anchor_file_references(binding, origin)
            bound['_check_contract'] = _revision_check_contract(transfer.get('check_basis', []))
            bound['_existing_check_status'] = transfer['status']
            fit_ref = {**reference, 'json_path': [*_path(reference), 'binding', 'fitting']}
            key = json.dumps({name: bound.get(name) for name in
                ('candidate_id', 'candidate_version', 'components', 'parent',
                 'constraint_hypothesis', 'context_ref')}, sort_keys=True)
            key += json.dumps({name: bound['fitting'].get(name) for name in
                               ('parameter_paths', 'evidence_refs')}, sort_keys=True)
            grouped.setdefault(key, {}).setdefault(method, (bound, theory, fit_ref))
        definition = Definition.from_dict(candidate['definition'])
        for group in grouped.values():
            if any(method not in group for method in methods):
                continue
            bound, initial, source_ref = group['initial']
            fit = bound['fitting']
            old_parent = bound.get('parent', {})
            retained_parent = version(old_parent.get('component_id'), old_parent.get('version', 1))
            if retained_parent is None:
                continue
            try:
                fixed, _ = _match_parent(Program.from_dict(candidate['before_pattern']),
                    Program.from_dict(retained_parent['program']).expanded(), candidate['holes'], budget)
                hypothesis = next(item for item in candidate['constraint_hypotheses']
                                  if item['id'] == bound['constraint_hypothesis'])
                if any(item['right']['name'] not in fixed for item in hypothesis['assignments']):
                    continue
                body = _assign_properties(definition.body, hypothesis['assignments'], fixed, budget)
                output = Program.from_dict(initial['program']).expanded()
                arguments, _ = _match_parent(body, output, candidate['holes'], budget,
                                              bindings=deepcopy(fixed))
                lexical = Definition(candidate['id'], definition.arguments, body)
                call = Program('call', name=lexical.definition_id, definition=lexical,
                    children=tuple(arguments[name] for name in lexical.arguments))
                paths = tuple(tuple(path) for path in fit['parameter_paths'])
                template = AbstractTemplate(f'{candidate["id"]}:open_parameters',
                    candidate['id'], call, (), (), paths)
                if (call.expanded() != output or tuple(fit['parameters']) != template.parameters
                        or any(lexical.arguments[path[0]] in fixed for path in paths)):
                    continue
                references = fit.get('evidence_refs', [])
                row_key = tuple(_canonical_key(item) for item in references)
                if row_key not in rows_cache:
                    _spend(budget, len(references))
                    values = resolve_program_sources(agent, references)
                    rows_cache[row_key] = [Observation(row['features'], row['target'],
                        tuple(row.get('history', []))) for row in values]
                rows = rows_cache[row_key]
                if not rows or any(group[method][0]['fitting'].get('fitted_rows') != len(rows)
                                   for method in methods):
                    continue
                outputs = []
                for method in methods:
                    binding, theory, reference = group[method]
                    saved = binding['fitting']
                    _spend(budget, len(paths))
                    if template.instantiate({}, saved['parameters']).expanded() != Program.from_dict(
                            theory['program']).expanded():
                        break
                    outputs.append({'method': method, 'parameters': saved['parameters'],
                        'training_loss': saved['training_loss'], 'source_ref': reference,
                        'theory_ref': {'id': theory['id'], 'version': theory.get('version', 1)},
                        'check_contract': binding['_check_contract'],
                        'check_status': binding['_existing_check_status']})
                if len(outputs) != len(methods):
                    continue
                records.append({'template': template, 'rows': rows, 'risk_quantile': .5,
                    **{key: deepcopy(initial[key]) for key in
                       ('role_signature', 'output_unit', 'context_ref')},
                    'source_ref': source_ref, 'outputs': outputs,
                    'report': {'status': fit['status'], 'search_completed': True,
                        'actual_fitted_rows': len(rows), 'eligible_fitting_rows': len(rows),
                        'parameter_identifiability': fit['parameter_identifiability'],
                        'candidate_methods': list(methods),
                        'linear_diagnostics': deepcopy(fit['linear_diagnostics'])}})
            except (KeyError, StopIteration, ValueError, OSError, _MissingContract):
                continue
        cache['queries'].append({'query': query, 'records': records,
            'selection_compute': selection_compute, 'recovery_compute': budget.spent - recovery_started})
    except _Incomplete:
        pass  # An incomplete recovery is never memoized as a complete query.
    return records


def _relation_request_basis(parent, components, arguments, candidate, path):
    """One identity for actual transfer and its per-object budget admission."""
    basis = {key: deepcopy(arguments.get(key)) for key in (
        'component_theory_ids', 'component_bindings', 'role_signature', 'output_unit',
        'context_ref', 'evidence_checks', 'allocation', 'parent_theory_id', 'fitting')}
    basis.update(library_path=path, candidate_id=candidate['id'], candidate_version=candidate['version'],
        parent={key: deepcopy(parent.get(key)) for key in ('id', 'version', 'source_ref', 'source_status')}
               if parent else None,
        components=[{key: deepcopy(component.get(key)) for key in (
            'id', 'version', 'source_ref', 'compatible', 'source_status')} for component in components])
    if parent and parent.get('_contract_source'):
        basis['parent']['contract_source'] = deepcopy(parent['_contract_source'].get(
            'source_ref', parent['_contract_source']))
    return basis


def _relation_transfer_components(agent, arguments, candidate):
    from .question_routing import resolved_theories

    retained_components = arguments.get('_retained_components')
    pool = ({item['id']: item for item in retained_components}
            if retained_components is not None else arguments.get('_resolved_pool'))
    if pool is None:
        pool = {theory['id']: theory for theory in resolved_theories(agent)}
    parent = (deepcopy(arguments['_retained_parent']) if '_retained_parent' in arguments else
              deepcopy(arguments['_parent_snapshot']) if '_parent_snapshot' in arguments
              else pool.get(arguments.get('parent_theory_id')))
    native_ids = []
    if candidate['kind'] == 'program_revision_relation':
        if retained_components is not None:
            # The original roster already records the automatically exposed
            # siblings. Later current versions cannot change a retained search.
            source = (parent or {}).get('source_ref', {})
            native_ids = [item['id'] for item in retained_components
                if source.get('kind') == 'native_candidate'
                and item.get('source_ref', {}).get('kind') == 'native_candidate'
                and all(item['source_ref'].get(key) == source.get(key) for key in (
                    'question_id', 'meaning_version', 'round_index'))
                and item['source_ref'].get('candidate_index') != source.get('candidate_index')]
        else:
            native_ids = _native_parent_components(agent, parent, pool.values())
        if retained_components is None and any(identifier not in pool or agent.theory_index[identifier].get('version', 1)
               != pool[identifier].get('version', 1) for identifier in native_ids):
            pool = {theory['id']: theory for theory in resolved_theories(agent)}
        parent = _retained_parent_contract(agent, parent)
    binding_ids = arguments.get('component_bindings')
    identifiers = _unique([*arguments.get('component_theory_ids', []), *native_ids])
    if binding_ids:
        identifiers = _unique([*identifiers, *binding_ids.values()])
    excluded, components = [], []
    for identifier in identifiers:
        component = pool.get(identifier)
        if component is None or not component.get('program'):
            excluded.append({'id': identifier, 'reason': 'component_unavailable'})
        elif arguments.get('context_ref') is not None and component.get('context_ref') != arguments['context_ref']:
            excluded.append({'id': identifier, 'reason': 'component_context_mismatch'})
        else:
            components.append(component)
    if retained_components is not None:
        components = list(retained_components)
    return {'pool': pool, 'parent': parent, 'native_ids': native_ids,
            'excluded': excluded, 'components': components}


def _transfer_relation(agent, arguments, candidate, path, *, source_library=None):
    from .revision_relations import (instantiate_relation, propose_relation, propose_revision,
        prepare_revision_visit, execute_revision_visit)

    prepared_transfer = arguments.get('_prepared_relation_visit')
    context = (prepared_transfer['context'] if prepared_transfer is not None else
               _relation_transfer_components(agent, arguments, candidate))
    pool, parent = context['pool'], context['parent']
    native_ids, excluded, components = (context['native_ids'], context['excluded'], context['components'])
    binding_ids = arguments.get('component_bindings')
    if not components and candidate['kind'] != 'program_revision_relation':
        return {'status': 'UNKNOWN', 'reason': 'no_typed_components_available',
            'generated_theories': [], 'excluded': excluded, 'new_compute_cost': 0,
            'new_observations': 0, 'llm_calls': 0, 'support_updated': False}
    allocation = deepcopy(arguments.get('allocation', {}))
    requested = allocation.get('compute', agent.config.compute_budget)
    if type(requested) is not int or requested < 0:
        raise ValueError('relation transfer compute must be a nonnegative integer')
    allowance = min(requested, max(0, agent.config.compute_budget - agent.costs['compute_cost']))
    direct_mission = None
    if 'mission' in arguments:
        agent.research_mission = adopt_research_mission(agent.research_mission, arguments['mission'])
        direct_mission = agent.research_mission
        allowance = min(allowance, mission_compute_remaining(agent, direct_mission))
    before_transfer = agent.costs['compute_cost']
    allocation['compute'] = allowance
    basis = _relation_request_basis(parent, components, arguments, candidate, path)
    cache_key = f'relation_transfer:{arguments["theory_id"]}'
    executions = agent.theory_consolidation.setdefault('executions', {})
    # Direct transfers also support a relocated frozen source; recalls provide
    # one shared map so this does not reparse a large source for every object.
    if '_source_libraries' not in arguments:
        arguments = {**arguments, '_source_libraries': {
            path: source_library if source_library is not None else agent.theory_consolidation}}
    previous, shared_progress_refs = _shared_relation_progress(
        agent, arguments, candidate, basis, cache_key, executions)
    same_work = previous is not None
    same_admission = same_work and all(previous['request_basis'].get(key) == basis.get(key)
                                      for key in ('parent', 'components'))
    if (same_admission and previous.get('search_completed', True) and
            (previous['result']['status'] != 'UNKNOWN' or allowance <= previous['allowance'])):
        return {**deepcopy(previous['result']), 'cache_hit': True, 'new_compute_cost': 0,
                'candidate_pool_changed': False, 'reserved_comparison_compute': 0,
                'execution_key': previous['execution_key']}
    source_documents = arguments.get('_source_documents')
    if source_documents is None:
        source_documents = {}
    original = library_entry_reference(agent, 'candidates', candidate['id'],
        documents=source_documents) if path == LIBRARY_PATH else (
        library_entry_reference(SimpleNamespace(workspace=agent.workspace,
            theory_consolidation=source_library, _theory_library_base=source_library,
            _theory_library_source_ref=_library_reference(agent, path)),
                'candidates', candidate['id'], documents=source_documents) if source_library is not None else {
                'file': path, 'json_path': ['candidates', candidate['id']]})
    cursor = previous.get('cursor') if same_work else None
    # Hold one complete retained check set inside the unchanged total allowance.
    allocation['compute'] = max(0, allowance - len(arguments.get('evidence_checks', [])))
    if candidate['kind'] == 'program_revision_relation':
        working_checks = _resolve_paid_checks(agent, arguments.get('evidence_checks', []))
        grammar = ({'grammar_preparation': arguments['_grammar_preparation']}
                   if '_grammar_preparation' in arguments else {})
        if prepared_transfer is not None:
            retained_fits = prepared_transfer['retained_fits']
            proposed = (execute_revision_visit(prepared_transfer['prepared_visit'], allocation=allocation)
                if prepared_transfer['prepared_visit'] is not None else
                {**prepared_transfer['result'], 'actual_compute_cost': 0})
            reuse_cost = 0
        else:
            reuse_budget = SearchBudget(allocation['compute'])
            versions = {identifier: arguments.get('_execution_candidates', {}).get(
                identifier, candidate)['version'] for identifier in
                arguments.get('_execution_candidate_ids', [candidate['id']])}
            fit_source = source_library
            if '_retained_fit_constructors' in arguments:
                # Removed proposal availability does not erase an already-paid
                # producer. These constructors are used only by the fit query.
                fit_source = {'candidates': {
                    **(source_library or {}).get('candidates', {}),
                    **arguments['_retained_fit_constructors']}}
            retained_fits = (_retained_revision_fits(agent, candidate, parent, reuse_budget,
                    execution_candidates=versions, source_library=fit_source)
                             if parent is not None and arguments.get('fitting') is not None else [])
            agent.costs['compute_cost'] += reuse_budget.spent
            allocation['compute'] -= reuse_budget.spent
            fit_state = next((saved for saved in getattr(agent, '_retained_revision_fit_cache', {}).get(
                'queries', []) if saved['records'] is retained_fits), None)
            if arguments.get('_prepare_complete_visit'):
                prepared = prepare_revision_visit(candidate, parent, components, allocation=allocation,
                    evidence_checks=working_checks, cursor=cursor,
                    fitting=arguments.get('fitting'), retained_fits=retained_fits,
                    retained_fit_state=fit_state, **grammar)
                agent.costs['compute_cost'] += prepared['actual_compute_cost']
                quote = dict(prepared.get('quote') or {})
                if quote:
                    checks = quote['maximum_outputs'] * len(arguments.get('evidence_checks', []))
                    quote.update(retained_check_compute=checks, compute=quote['compute'] + checks)
                completed = prepared.get('result') or {}
                if completed.get('completed_research_unit') and 'research_unit_position' in completed:
                    prepared['search_position'] = deepcopy(completed['research_unit_position'])
                return {**prepared, 'quote': quote,
                    'new_compute_cost': reuse_budget.spent + prepared['actual_compute_cost'],
                    'retained_fit_recovery_compute': reuse_budget.spent,
                    '_prepared_relation_visit': {'context': context,
                        'retained_fits': retained_fits, 'prepared_visit': prepared.get('prepared_visit'),
                        'result': prepared.get('result')}}
            proposed = propose_revision(candidate, parent, components, allocation=allocation,
                evidence_checks=working_checks, cursor=cursor,
                fitting=arguments.get('fitting'), retained_fits=retained_fits,
                retained_fit_state=fit_state, **grammar)
            reuse_cost = reuse_budget.spent
    elif binding_ids:
        available = {component['id']: component for component in components}
        if any(identifier not in available for identifier in binding_ids.values()):
            return {'status': 'UNKNOWN', 'reason': 'bound_component_unavailable',
                'generated_theories': [], 'excluded': excluded, 'new_compute_cost': 0,
                'new_observations': 0, 'llm_calls': 0, 'support_updated': False}
        proposed = instantiate_relation(candidate, {
            name: available[identifier] for name, identifier in binding_ids.items()}, allocation=allocation)
    else:
        proposed = propose_relation(candidate, components, allocation=allocation, cursor=cursor)
    if candidate['kind'] != 'program_revision_relation':
        reuse_cost = 0
    agent.costs['compute_cost'] += proposed['actual_compute_cost']
    comparison_reserve = proposed.get('reserved_comparison_compute', 0)
    if comparison_reserve:
        # The enclosing route owns restoration of its task configuration. Hold
        # these units through matching/selection as well as retained checks.
        agent.config = replace(agent.config,
            compute_budget=agent.config.compute_budget - comparison_reserve)
    generated, spent, changed = [], proposed['actual_compute_cost'] + reuse_cost, False
    retained_instances = {_canonical_key(output['source_ref']): output
        for record in (retained_fits if candidate['kind'] == 'program_revision_relation' else [])
        for output in record['outputs']}
    proposals = proposed['proposals']
    try:
        for index, proposal in enumerate(proposals):
            if (arguments.get('output_unit') is not None
                    and proposal['output_unit'] != arguments['output_unit']):
                excluded.append({'reason': 'relation_output_unit_mismatch'})
                continue
            position = proposal.get('search_position')
            if position is not None:
                identifier = f'{arguments["theory_id"]}:binding:{position["layer"]}:{position["offset"]}'
                if 'fit_index' in position:
                    identifier += f':fit:{position["fit_index"]}'
            else:
                identifier = arguments['theory_id'] if len(proposals) == 1 else f'{arguments["theory_id"]}:binding:{index}'
            binding = {'library_path': path, 'candidate_id': candidate['id'],
                'candidate_version': candidate['version'], 'components': deepcopy(proposal['bindings']),
                'context_ref': deepcopy(proposal['context_ref'])}
            if proposal.get('parent_binding'):
                binding.update(parent=deepcopy(proposal['parent_binding']),
                               constraint_hypothesis=proposal['constraint_hypothesis'])
            if 'fitting' in proposal:
                binding['fitting'] = deepcopy(proposal['fitting'])
            reuse_ref = proposal.get('fitting', {}).get('fit_reuse_output_ref')
            known = retained_instances.get(_canonical_key(reuse_ref)) if reuse_ref else None
            current = pool.get(known['theory_ref']['id']) if known else None
            if (current and current.get('program') == proposal['program']
                    and all(current.get(key) == proposal.get(key) for key in
                            ('role_signature', 'output_unit', 'context_ref'))
                    and known['check_contract'] == _revision_check_contract(arguments.get('evidence_checks', []))):
                status = ('refuted' if current.get('compatible') is False else known['check_status'])
                fit = binding['fitting']
                fit.pop('linear_diagnostics', None)
                fit.pop('evidence_refs', None)
                for field in ('linear_diagnostics', 'evidence_refs'):
                    fit[f'{field}_ref'] = {**deepcopy(reuse_ref), 'json_path': [*_path(reuse_ref), field]}
                generated.append({'theory_id': current['id'], 'version': current.get('version', 1),
                    'status': status, 'binding': binding, 'reused_existing_theory': True,
                    'existing_fitting_ref': deepcopy(reuse_ref),
                    'constraint_checks': deepcopy(proposal.get('constraint_checks', [])),
                    'source_reconstruction_preserved': True})
                # This is an old instance already in the complete comparison pool.
                released = allocation.get('comparison_rows', 0)
                comparison_reserve -= released
                if released:
                    agent.config = replace(agent.config, compute_budget=agent.config.compute_budget + released)
                continue
            inputs = {**arguments, 'theory_id': identifier,
                'role_signature': proposal['role_signature'], 'output_unit': proposal['output_unit'],
                'context_ref': proposal['context_ref']}
            if 'subtree_contracts' in proposal:
                inputs['subtree_contracts'] = deepcopy(proposal['subtree_contracts'])
                inputs['subtree_contract_sources'] = {position: {
                    'hole_contract_ref': {**deepcopy(original),
                        'json_path': [*_path(original), 'holes', origin['hole_index']]},
                    'component_binding': deepcopy(proposal['bindings'][origin['hole_name']])}
                    for position, origin in proposal.get('subtree_contract_sources', {}).items()}
            result = _checked_transfer(agent, inputs, candidate,
                Program.from_dict(proposal['program']), binding,
                compute_allowance=max(0, allowance - spent - comparison_reserve), persist=False,
                retained_counterexample_refs=_retained_binding_failures(
                    agent, source_library or {}, path, binding,
                    documents=source_documents))
            spent += result['new_compute_cost']
            changed = changed or result['candidate_pool_changed']
            generated.append({'theory_id': identifier, 'version': agent.theory_index[identifier]['version'],
                'status': result['status'], 'binding': binding,
                'constraint_checks': deepcopy(proposal.get('constraint_checks', [])),
                **({'parent_binding': deepcopy(proposal['parent_binding']),
                    'constraint_hypothesis': proposal['constraint_hypothesis']} if proposal.get('parent_binding') else {}),
                'checks': result.get('checks', []), 'source_reconstruction_preserved': True})
            if result.get('paid_mse_check'):
                generated[-1]['paid_mse_check'] = deepcopy(result['paid_mse_check'])
    except Exception:
        # A failed later candidate must not erase already completed checks.
        # The normal return below persists this bounded batch only once.
        if direct_mission is not None:
            debit_mission_compute(agent, direct_mission, before_transfer)
        _save(agent)
        raise
    unknown = proposed['budget_exhausted'] or proposed['status'] == 'UNKNOWN'
    status = ('UNKNOWN' if unknown else 'candidates_registered' if any(
        item['status'] == 'candidate_registered' for item in generated) else
        'refuted' if generated and all(item['status'] == 'refuted' for item in generated) else 'UNKNOWN')
    result = {'status': status, 'generated_theories': generated,
        'execution_key': cache_key,
        'candidate_pool_changed': changed, 'excluded': [*excluded, *proposed['excluded']],
        'new_compute_cost': spent, 'budget_exhausted': proposed['budget_exhausted'], 'cache_hit': False,
        'reserved_comparison_compute': comparison_reserve,
        'reason': proposed.get('reason'), 'parent_binding': proposed.get('parent_binding'),
        **({'parent_contract_recovery': deepcopy(proposed['parent_contract_recovery'])}
           if 'parent_contract_recovery' in proposed else {}),
        **({'parameter_condition_context': deepcopy(proposed['parameter_condition_context'])}
           if 'parameter_condition_context' in proposed else {}),
        'retained_native_component_ids': native_ids,
        'retained_complete_fit_batches_available': (len(retained_fits)
            if candidate['kind'] == 'program_revision_relation' else 0),
        'retained_fit_recovery_compute': reuse_cost,
        'retained_fit_index_compute': proposed.get('retained_fit_index_compute', 0),
        'retained_fit_reuse_count': proposed.get('retained_fit_reuse_count', 0),
        'reused_existing_theory_count': sum(item.get('reused_existing_theory', False) for item in generated),
        'author_work': ([{'type': 'AuthorWorkRequest', 'reason': proposed['reason'],
            **({'required': ['operations', 'risk_quantile', 'max_history', 'max_structures']}
               if proposed['reason'] in {'finite_revision_fitting_contract_required',
                   'whole_after_fitting_required_for_parameter_proposal'} else {})}]
            if proposed.get('reason') in {'typed_open_hole_components_or_fitting_contract_required',
                                         'finite_revision_fitting_contract_required',
                                         'whole_after_fitting_required_for_parameter_proposal'} else []),
        'incomplete_fit_positions': deepcopy(proposed.get('incomplete_fit_positions', [])),
        'cursor': deepcopy(proposed.get('cursor')), 'search_completed': proposed.get('search_completed', True),
        **{key: deepcopy(proposed[key]) for key in (
            'completed_research_unit', 'research_unit_position',
            'complete_research_unit_quote', 'preparation_compute_cost')
           if key in proposed},
        'support_updated': False, 'independent_confirmation': False, 'new_observations': 0, 'llm_calls': 0}
    if any(key != cache_key for key in shared_progress_refs):
        result['retained_execution_refs'] = shared_progress_refs
    if proposed.get('minimum_compute_for_progress'):
        result['minimum_compute_for_progress'] = (
            proposed['minimum_compute_for_progress'] + reuse_cost
            + len(arguments.get('evidence_checks', [])))
    executions[cache_key] = {'request_basis': basis, 'allowance': allowance, 'result': deepcopy(result)}
    executions[cache_key].update(cursor=deepcopy(proposed.get('cursor')),
                                 search_completed=proposed.get('search_completed', True))
    if direct_mission is not None:
        debit_mission_compute(agent, direct_mission, before_transfer)
    _save(agent)
    return result
