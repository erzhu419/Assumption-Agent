"""Connect retained Episodes and frozen abstractions to Common's Program pool."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from .abstractions import AbstractTemplate
from .language import Program
from .models import Observation, SearchBudget
from .research_episodes import (
    extract_episodes, resolve_program_sources, _canonical, _contains_reference, _path, _source_id,
    load_source_values,
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
    if isinstance(value, list):
        return [_anchor_file_references(item, workspace) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    anchored = {key: _anchor_file_references(item, workspace) for key, item in value.items()}
    if anchored.get('file') and ('json_path' in anchored or 'field' in anchored):
        anchored = _absolute_reference(anchored, workspace)
    return anchored


def _read_library(reference, workspace):
    reference = _absolute_reference(reference, workspace)
    document = load_source_values([reference], workspace)[0]['value']
    document = _anchor_file_references(document, _reference_workspace(reference))
    if 'source_ref' not in document:
        return deepcopy(document)
    library = _read_library(document['source_ref'], Path(reference['file']).parent)
    for field in _LIBRARY_MAPS:
        library.setdefault(field, {}).update(deepcopy(document.get(field, {})))
    for identifier in document.get('removed_candidates', []):
        library['candidates'].pop(identifier, None)
    _merge_intake(library, {field: document.get(field, []) for field in _LIBRARY_LISTS
                            if field != 'transfers'})
    transfers = library.setdefault('transfers', [])
    transfers.extend(deepcopy(item) for item in document.get('transfers', []) if item not in transfers)
    return library


def install_theory_library(agent, reference=None):
    """Read one frozen library and retain only target-local writes on disk."""
    local = agent.workspace / LIBRARY_PATH
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
        base = _read_library(source, agent.workspace)
    agent._theory_library_base = deepcopy(base)
    agent._theory_library_source_ref = source
    return deepcopy(document) if source is None else _read_library(selected, agent.workspace)


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


def library_entry_reference(agent, field, key, *parts):
    """Address inherited checks at their source and new checks at their local index."""
    reference = {'file': LIBRARY_PATH, 'json_path': [field, key, *parts]}
    source = getattr(agent, '_theory_library_source_ref', None)
    if not source:
        return reference
    value = agent.theory_consolidation[field][key]
    base = agent._theory_library_base.get(field, [] if field in _LIBRARY_LISTS else {})
    if field in _LIBRARY_LISTS:
        if value not in base:
            reference['json_path'][1] = _library_document(agent)[field].index(value)
            return reference
        source_key = base.index(value)
    elif key not in base or value != base[key]:
        return reference
    else:
        source_key = key
    # The frozen source may itself be a target delta. Follow its own addresses.
    while True:
        document = _anchor_file_references(load_source_values([source], agent.workspace)[0]['value'],
                                           _reference_workspace(source))
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
    return {**deepcopy(source), 'json_path': [*_path(source), field, source_key, *parts]}


def _read_index(reference, workspace):
    reference = _absolute_reference(reference, workspace)
    root = load_source_values([{**reference, 'json_path': []}], workspace)[0]['value']
    indexed = root
    for part in _path(reference):
        indexed = indexed[part]
    if 'source_ref' in indexed and 'entries' in indexed:
        merged = _read_index(indexed['source_ref'], Path(reference['file']).parent)
        merged.update(deepcopy(indexed['entries']))
        return merged
    return _snapshot_index(indexed, root, reference)


def _snapshot_index(indexed, root, reference):
    from .question_routing import resolved_theories

    def snapshot(identifier, entry):
        item = deepcopy(entry)
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
        if history:
            item['history'] = [snapshot(identifier, old) for old in history]
        return _anchor_file_references(item, _reference_workspace(reference, root))
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
    agent._save(INDEX_PATH, json.dumps(document, ensure_ascii=False, indent=2))


def _save(agent):
    agent._save(LIBRARY_PATH, json.dumps(_library_document(agent), ensure_ascii=False, indent=2))


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
    return {'compute': compute, 'max_groups': arguments.get('allocation', {}).get('max_groups', 16)}


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


def _revision_entries(agent, episodes, pool, retained):
    pending, references = [], []
    for episode in episodes:
        declaration = pool[episode['theory_id']]
        declared_ref = declaration.get('program_source_ref')
        for index, change in enumerate(episode.get('actual_changes', [])):
            if not change.get('before_ref'):
                continue
            after_ref = change['source_ref']
            if declared_ref and 'actions' in _path(declared_ref) and not (
                    _contains_reference(after_ref, declared_ref)
                    and _contains_reference(declared_ref, after_ref)):
                continue
            entry = {key: deepcopy(episode[key]) for key in (
                'source_id', 'theory_id', 'role_signature', 'output_unit', 'context_ref',
                'context_family_ref', 'evidence_refs', 'cost_refs', 'failure_refs', 'source_cutoff') if key in episode}
            entry.update(episode_id=f'{episode["episode_id"]}:revision{index}',
                source_ref=deepcopy(after_ref), before_ref=deepcopy(change['before_ref']),
                after_ref=deepcopy(after_ref), action_ref=deepcopy(change.get('action_ref')),
                source_outcomes={key: deepcopy(episode[key]) for key in (
                    'status', 'costs', 'cost_scope', 'independent_results', 'retained_later_outcome_refs')
                    if key in episode})
            pending.append(entry)
            references.extend([change['before_ref'], after_ref])
    values = resolve_program_sources(agent, references, retained)
    for index, entry in enumerate(pending):
        entry['before_program'], entry['after_program'] = values[index * 2:index * 2 + 2]
    return pending


def _same_frozen_candidate(old, new):
    fields = ('kind', 'instances', 'source_bindings', 'role_signature', 'parameter_paths',
              'scope', 'source_evidence_refs', 'source_outcomes', 'counterexample_checks',
              'blocked_transfer_bindings', 'learned_constraints', 'holes')
    if not all(old.get(key) == new.get(key) for key in fields):
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

    register_theories(agent, arguments.get('theories', []))
    allocation = _allocation(agent, arguments, question)
    if allocation is None:
        return {'status': 'research_deferred', 'reason': 'theory_allocation_missing', 'new_compute_cost': 0}
    library = agent.theory_consolidation
    mission = effective_question_mission(question, arguments.get('mission', agent.research_mission)
                                         if question is not None else agent.research_mission)
    library.setdefault('candidates', {})
    library.setdefault('executions', {})
    new_materials, material_source, material_cursor = _new_local_materials(agent, library)
    selected = arguments.get('source_theory_ids', list(agent.theory_index))
    pool = {theory['id']: theory for theory in resolved_theories(agent)}
    theme = arguments.get('theme', (question or {}).get('id', 'retained_programs'))
    request_basis = {key: deepcopy(arguments.get(key)) for key in (
        'source_theory_ids', 'materials', 'episode_sources', 'counterexamples', 'context_family_ref', 'allocation')}
    request_basis.update(source_id=_source_id(agent), sources=[{key: deepcopy(pool[identifier].get(key)) for key in (
        'id', 'version', 'source_ref', 'source_status', 'compatible')} for identifier in selected if identifier in pool],
        paid_cutoff=[len(agent.observations), len(agent.rows)],
        revisions=[{key: deepcopy(revision.get(key)) for key in ('index', 'status', 'evidence_status', 'check_results')}
                   for revision in agent.revisions], max_groups=allocation['max_groups'],
        theory_allocation=deepcopy(mission.get('allocation', mission.get('theory_learning_allocation'))),
        episode_file_versions=_episode_file_versions(agent, arguments.get('episode_sources', [])),
        native_versions=[{'question_id': q['id'], 'meaning_version': q.get('meaning_version', 1),
            'rounds': [{key: deepcopy(round_.get(key)) for key in ('round_index', 'status')}
                       for round_ in q.get('native_condition_research', [])]} for q in agent.questions
                         if q.get('native_condition_research')])
    previous = library['executions'].get(theme)
    if (previous and previous['request_basis'] == request_basis and
            (previous['result']['status'] != 'UNKNOWN' or allocation['compute'] <= previous['allowance'])):
        if new_materials:
            _merge_intake(library, extract_episodes(agent, new_materials))
            library.setdefault('material_cursors', {})[material_source] = material_cursor
            _save(agent)
        result = {**deepcopy(previous['result']), 'cache_hit': True, 'new_compute_cost': 0,
                'conflict_count': len(library['conflicts']),
                'support_source_count': len(library['unique_support_sources'])}
        return _consolidation_return(question, theme, result, previous['allowance'])
    from .research_episodes import load_episode_sources
    retained = load_episode_sources(arguments.get('episode_sources', []), agent.workspace)
    intake = extract_episodes(agent, [*new_materials, *arguments.get('materials', [])],
                              retained_episodes=retained)
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
        computed['actual_compute_cost'] += relations['actual_compute_cost']
        computed['budget_exhausted'] = computed['budget_exhausted'] or relations['budget_exhausted']
        if computed['budget_exhausted']:
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


def read_theory_library(agent, path=LIBRARY_PATH):
    """Read the selected frozen source without replacing the working library."""
    if path == LIBRARY_PATH:
        return agent.theory_consolidation
    return _read_library(_library_reference(agent, path), agent.workspace)


def _retained_binding_failures(agent, library, path, binding):
    if path == LIBRARY_PATH:
        return []
    proxy = SimpleNamespace(workspace=agent.workspace, theory_consolidation=library,
        _theory_library_base=library, _theory_library_source_ref=_library_reference(agent, path))
    wanted = {key: value for key, value in binding.items() if key != 'library_path'}
    references = []
    for index, record in enumerate(library.get('transfers', [])):
        retained = {key: value for key, value in record['binding'].items() if key != 'library_path'}
        if record['status'] != 'refuted' or retained != wanted:
            continue
        for position, check in enumerate(record.get('checks', [])):
            if check['status'] == 'failed' and check.get('context_ref') == binding['context_ref']:
                references.append(library_entry_reference(proxy, 'transfers', index, 'checks', position))
        references.extend(deepcopy(record.get('retained_counterexample_refs', [])))
        references.extend(deepcopy(record.get('scope_refutation', {}).get('counterexample_refs', [])))
    return _unique(references)


def transfer_theory(agent, arguments, laboratory=None):
    path = arguments.get('library_path', LIBRARY_PATH)
    library = (install_theory_library(agent) if path == LIBRARY_PATH else
               read_theory_library(agent, path))
    if path == LIBRARY_PATH:
        agent.theory_consolidation = library
    candidate = library['candidates'][arguments['candidate_id']]
    if candidate['kind'] == 'program_context_relation':
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
        retained_counterexample_refs=_retained_binding_failures(agent, library, path, binding))


def recall_frozen_theories(agent, question, arguments):
    """Use the existing transfer operators on a typed new question, within budget.

    The current component pool is frozen before this pass; newly proposed
    Programs do not recursively become components of their own recall work.
    """
    from .question_routing import resolved_theories

    path = arguments.get('library_path', question.get('frozen_library_path', LIBRARY_PATH))
    candidates = read_theory_library(agent, path).get('candidates', {})
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
    components = [item for item in resolved_theories(agent) if not item['id'].startswith(prefix)]
    checks = deepcopy(target.get('evidence_checks', arguments.get('evidence_checks', [])))
    basis = {'target': deepcopy(target), 'checks': checks, 'library_path': path,
        'method_ref': 'hegel_machine.cognition.abstractions.AbstractTemplate.mappings',
        'candidates': [{'id': key, 'version': item['version']} for key, item in candidates.items()],
        'components': [{key: deepcopy(item.get(key)) for key in (
            'id', 'version', 'source_ref', 'source_status', 'compatible')} for item in components]}
    previous = question.get('frozen_recall_execution')
    if (previous and previous['basis'] == basis and
            (previous['result']['status'] != 'UNKNOWN' or allocation['compute'] <= previous['allowance'])):
        return {**deepcopy(previous['result']), 'cache_hit': True, 'new_compute_cost': 0}
    start = agent.costs['compute_cost']
    budget = SearchBudget(allocation['compute'])
    generated, work, author_work, excluded = [], [], [], []
    incomplete = False

    def charge(units=1):
        if not budget.spend(units):
            return False
        agent.costs['compute_cost'] += units
        return True

    def apply(inputs, identifier, *, mapping=None, binding_index=None):
        nonlocal incomplete
        inputs['library_path'] = path
        inputs['allocation'] = {'compute': budget.remaining}
        applied = transfer_theory(agent, inputs)
        budget.spend(applied['new_compute_cost'])
        entries = applied.get('generated_theories', [applied])
        item = {'candidate_id': identifier, 'status': applied['status'],
            'theory_ids': [entry['theory_id'] for entry in entries if entry.get('theory_id')],
            'new_compute_cost': applied['new_compute_cost']}
        if mapping is not None:
            item.update(mapping=deepcopy(mapping), binding_index=binding_index)
        work.append(item)
        generated.extend(entry['theory_id'] for entry in entries
                         if entry.get('status') == 'candidate_registered')
        incomplete = incomplete or applied['status'] == 'UNKNOWN'

    roles = target.get('role_signature')
    if (not isinstance(roles, dict) or target.get('output_unit') is None
            or target.get('context_ref') is None
            or any(not isinstance(contract, dict) or any(key not in contract for key in ('role', 'unit', 'time'))
                   for contract in roles.values())):
        author_work.append({'type': 'AuthorWorkRequest', 'reason': 'typed_target_binding_required',
                            'required': ['role_signature', 'output_unit', 'context_ref']})
    else:
        for identifier, candidate in candidates.items():
            if not charge():
                incomplete = True
                break
            inputs = {'candidate_id': identifier, 'theory_id': prefix + identifier,
                'role_signature': deepcopy(roles), 'output_unit': target['output_unit'],
                'context_ref': deepcopy(target['context_ref']), 'evidence_checks': checks}
            if candidate['output_unit'] != target['output_unit']:
                excluded.append({'candidate_id': identifier, 'reason': 'output_unit_mismatch'})
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
                    incomplete = True
                    break
                if not compatible_mappings:
                    author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                        'reason': 'typed_injective_role_mapping_required'})
                continue
            elif candidate['kind'] == 'program_context_relation':
                fixed_roles = candidate.get('fixed_role_signature', {})
                if not charge(len(fixed_roles)):
                    incomplete = True
                    break
                if any(roles.get(name) != contract for name, contract in fixed_roles.items()):
                    author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                        'reason': 'fixed_context_role_binding_required'})
                    continue
                if not charge(len(components)):
                    incomplete = True
                    break
                inputs['component_theory_ids'] = [item['id'] for item in components
                    if item.get('program') and item.get('context_ref') == target['context_ref']]
                if not inputs['component_theory_ids']:
                    author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                        'reason': 'typed_target_components_required'})
                    continue
            else:
                author_work.append({'candidate_id': identifier, 'type': 'AuthorWorkRequest',
                                    'reason': 'parent_composition_required_for_shared_fragment'})
                continue
            apply(inputs, identifier)
    result = {'status': 'UNKNOWN' if incomplete else 'author_work_request' if author_work and not work
              else 'completed', 'generated_theory_ids': _unique(generated), 'work': work,
        'author_work': author_work, 'excluded': excluded,
        'new_compute_cost': agent.costs['compute_cost'] - start,
        'cache_hit': False, 'support_updated': False, 'independent_confirmation': False}
    debit_mission_compute(agent, mission, start)
    question['frozen_recall_execution'] = {'basis': basis, 'allowance': allocation['compute'],
                                          'result': deepcopy(result)}
    return result


def _checked_transfer(agent, arguments, candidate, program, binding, *, compute_allowance=None,
                      retained_counterexample_refs=()):
    from .question_routing import register_theories, scoped_numeric_refutation

    context, target_roles = arguments['context_ref'], arguments['role_signature']
    # A frozen failed binding cannot be made fresh by omitting its old checks.
    blocked = any(all(item.get(key) == binding.get(key) for key in ('mapping', 'parameters', 'context_ref'))
                  for item in candidate.get('blocked_transfer_bindings', []))
    preceding = [item for item in agent.theory_consolidation.get('transfers', []) if item['binding'] == binding]
    blocked = blocked or bool(retained_counterexample_refs) or any(item['status'] == 'refuted' for item in preceding)
    check_basis = deepcopy(arguments.get('evidence_checks', []))
    cached = next((item for item in reversed(preceding) if item['check_basis'] == check_basis), None)
    if cached and (not blocked or cached['status'] == 'refuted') and cached['theory_id'] == arguments['theory_id']:
        return {**deepcopy(cached), 'cache_hit': True, 'candidate_pool_changed': False,
                'new_compute_cost': 0, 'new_observations': 0, 'llm_calls': 0,
                'source_reconstruction_preserved': True}
    checks, spent, unknown = [], 0, False
    for example in arguments.get('evidence_checks', []):
        if example['context_ref'] != context:
            checks.append({'source_ref': deepcopy(example['source_ref']), 'status': 'UNKNOWN',
                           'reason': 'check_context_mismatch'})
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
    identifier = arguments['theory_id']
    declaration = {'id': identifier, 'program': program.to_dict(),
        'family_id': candidate['id'], 'role_signature': deepcopy(target_roles),
        'output_unit': arguments['output_unit'], 'context_ref': deepcopy(context),
        'declared_scope_ref': deepcopy(context), 'parameter_paths': candidate.get('parameter_paths', []),
        'consolidation_ref': deepcopy(binding),
        'support': {'status': 'refuted' if blocked else 'candidate', 'evidence': [],
                    'context_ref': deepcopy(context)}, 'compatible': not blocked}
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
    if retained_counterexample_refs:
        record['retained_counterexample_refs'] = deepcopy(retained_counterexample_refs)
    if scoped_failure:
        record['scope_refutation'] = deepcopy(scoped_failure)
    transfers = agent.theory_consolidation.setdefault('transfers', [])
    if record not in transfers:
        transfers.append(record)
    _save(agent)
    return {'status': status, 'theory_id': identifier, 'binding': binding, 'checks': checks,
            **({'retained_counterexample_refs': deepcopy(retained_counterexample_refs)}
               if retained_counterexample_refs else {}),
            **({'scope_refutation': deepcopy(scoped_failure)} if scoped_failure else {}),
            'candidate_pool_changed': old_version != agent.theory_index[identifier]['version'], 'new_compute_cost': spent,
            'source_reconstruction_preserved': True, 'support_updated': False,
            'independent_confirmation': False, 'new_observations': 0, 'llm_calls': 0}


def _transfer_relation(agent, arguments, candidate, path, *, source_library=None):
    from .question_routing import resolved_theories
    from .revision_relations import instantiate_relation, propose_relation

    pool = {theory['id']: theory for theory in resolved_theories(agent)}
    binding_ids = arguments.get('component_bindings')
    identifiers = arguments.get('component_theory_ids', [])
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
    if not components:
        return {'status': 'UNKNOWN', 'reason': 'no_typed_components_available',
            'generated_theories': [], 'excluded': excluded, 'new_compute_cost': 0,
            'new_observations': 0, 'llm_calls': 0, 'support_updated': False}
    allocation = deepcopy(arguments.get('allocation', {}))
    requested = allocation.get('compute', agent.config.compute_budget)
    if type(requested) is not int or requested < 0:
        raise ValueError('relation transfer compute must be a nonnegative integer')
    allowance = min(requested, max(0, agent.config.compute_budget - agent.costs['compute_cost']))
    allocation['compute'] = allowance
    basis = {key: deepcopy(arguments.get(key)) for key in (
        'component_theory_ids', 'component_bindings', 'role_signature', 'output_unit',
        'context_ref', 'evidence_checks', 'allocation')}
    basis.update(library_path=path, candidate_id=candidate['id'], candidate_version=candidate['version'],
        components=[{key: deepcopy(component.get(key)) for key in (
            'id', 'version', 'source_ref', 'compatible', 'source_status')} for component in components])
    cache_key = f'relation_transfer:{arguments["theory_id"]}'
    executions = agent.theory_consolidation.setdefault('executions', {})
    previous = executions.get(cache_key)
    if (previous and previous['request_basis'] == basis and
            (previous['result']['status'] != 'UNKNOWN' or allowance <= previous['allowance'])):
        return {**deepcopy(previous['result']), 'cache_hit': True, 'new_compute_cost': 0,
                'candidate_pool_changed': False}
    if binding_ids:
        available = {component['id']: component for component in components}
        if any(identifier not in available for identifier in binding_ids.values()):
            return {'status': 'UNKNOWN', 'reason': 'bound_component_unavailable',
                'generated_theories': [], 'excluded': excluded, 'new_compute_cost': 0,
                'new_observations': 0, 'llm_calls': 0, 'support_updated': False}
        proposed = instantiate_relation(candidate, {
            name: available[identifier] for name, identifier in binding_ids.items()}, allocation=allocation)
    else:
        proposed = propose_relation(candidate, components, allocation=allocation)
    agent.costs['compute_cost'] += proposed['actual_compute_cost']
    generated, spent, changed = [], proposed['actual_compute_cost'], False
    proposals = proposed['proposals']
    for index, proposal in enumerate(proposals):
        if (arguments.get('output_unit') is not None
                and proposal['output_unit'] != arguments['output_unit']):
            excluded.append({'reason': 'relation_output_unit_mismatch'})
            continue
        identifier = arguments['theory_id'] if len(proposals) == 1 else f'{arguments["theory_id"]}:binding:{index}'
        binding = {'library_path': path, 'candidate_id': candidate['id'],
            'candidate_version': candidate['version'], 'components': deepcopy(proposal['bindings']),
            'context_ref': deepcopy(proposal['context_ref'])}
        inputs = {**arguments, 'theory_id': identifier,
            'role_signature': proposal['role_signature'], 'output_unit': proposal['output_unit'],
            'context_ref': proposal['context_ref']}
        result = _checked_transfer(agent, inputs, candidate,
            Program.from_dict(proposal['program']), binding, compute_allowance=max(0, allowance - spent),
            retained_counterexample_refs=_retained_binding_failures(
                agent, source_library or {}, path, binding))
        spent += result['new_compute_cost']
        changed = changed or result['candidate_pool_changed']
        generated.append({'theory_id': identifier, 'version': agent.theory_index[identifier]['version'],
            'status': result['status'], 'binding': binding,
            'constraint_checks': deepcopy(proposal.get('constraint_checks', [])),
            'checks': result.get('checks', []), 'source_reconstruction_preserved': True})
    unknown = proposed['budget_exhausted'] or proposed['status'] == 'UNKNOWN'
    status = ('UNKNOWN' if unknown else 'candidates_registered' if any(
        item['status'] == 'candidate_registered' for item in generated) else
        'refuted' if generated and all(item['status'] == 'refuted' for item in generated) else 'UNKNOWN')
    result = {'status': status, 'generated_theories': generated,
        'candidate_pool_changed': changed, 'excluded': [*excluded, *proposed['excluded']],
        'new_compute_cost': spent, 'budget_exhausted': proposed['budget_exhausted'], 'cache_hit': False,
        'support_updated': False, 'independent_confirmation': False, 'new_observations': 0, 'llm_calls': 0}
    executions[cache_key] = {'request_basis': basis, 'allowance': allowance, 'result': deepcopy(result)}
    _save(agent)
    return result
