"""Read compact research episodes from retained execution and evidence records.

Programs, observations and analysis output stay in their original stores.  This
module records addresses and actual outcomes; a declaration is not adoption,
and another description of a paid record is not another experiment.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace


MATERIAL_KINDS = {
    'measured', 'execution', 'external_claim', 'author_interpretation', 'idea',
    'error', 'unresolved', 'duplicate',
}
SEMANTIC_FIELDS = ('question', 'kind', 'query_semantics', 'answer_type',
                   'context_ref', 'answer_resolution_ref')


def _ref(field, *path):
    return {'file': 'research_state.json', 'json_path': [field, *path]}


def _path(reference):
    if 'json_path' in reference:
        return reference['json_path']
    return [name if name else int(index) for name, index in
            re.findall(r'([A-Za-z_]\w*)|\[(\d+)\]', reference.get('field', ''))]


def _canonical(reference):
    result = deepcopy(reference)
    if 'field' in result:
        result['json_path'] = _path(result)
        result.pop('field')
    return result


def _child_ref(reference, *path):
    result = _canonical(reference)
    result['json_path'] = [*_path(result), *path]
    return result


def _contains_reference(parent, child):
    parent, child = _canonical(parent), _canonical(child)
    return (all(parent.get(key) == child.get(key) for key in ('archive', 'unit', 'member', 'file'))
            and _path(child)[:len(_path(parent))] == _path(parent))


def _source_suffix(reference):
    return ''.join(f'[{part}]' if isinstance(part, int) else ('.' if index else '') + part
                   for index, part in enumerate(_path(reference)))


def load_source_values(references, workspace):
    """Read small addressed JSON values without restoring an archive tree."""
    from .effective_history import HistoryStore

    payloads, loaded = {}, []
    for original in references:
        reference = _canonical(original)
        if reference.get('archive'):
            key = reference['archive'], reference['unit'], reference['member']
            if key not in payloads:
                store = HistoryStore(reference['archive'], readonly=True)
                try:
                    rows = store.connection.execute(
                        'SELECT metadata,sequence FROM entries WHERE unit=? ORDER BY ordinal',
                        (reference['unit'],))
                    for metadata, sequence in rows:
                        if json.loads(metadata)['name'] == reference['member']:
                            payloads[key] = json.loads(b''.join(store.read_sequence(sequence)))
                            break
                    else:
                        raise ValueError('retained Episode member is unavailable')
                finally:
                    store.close()
        else:
            path = Path(reference['file'])
            path = path if path.is_absolute() else Path(workspace) / path
            key = str(path.resolve())
            if key not in payloads:
                payloads[key] = json.loads(path.read_text())
        episode, parent = payloads[key], None
        for component in _path(reference):
            parent = episode
            episode = episode[component]
        loaded_entry = {'source_ref': reference, 'value': episode}
        if (isinstance(episode, dict) and 'task_id' in episode and isinstance(parent, list)
                and sum(item.get('task_id') == episode['task_id'] for item in parent) > 1):
            loaded_entry['ambiguous_task_id'] = True
        loaded.append(loaded_entry)
    return loaded


def load_episode_sources(references, workspace):
    return [{**{key: value for key, value in loaded.items() if key != 'value'},
             'episode': loaded['value']} for loaded in load_source_values(references, workspace)]


def load_common_sources(references, workspace):
    """Read actual Common sources using their existing index and Episode extractor.

    Source references address research_state.json or a saved Common result.
    Optional source_theory_ids names local IDs, including an explicit empty list.
    Qualified IDs distinguish containers; only original declared source identity
    determines independence. Programs and paid records stay at their old addresses.
    """
    from .language import Program
    from .question_routing import resolved_theories
    from .theory_consolidation import (
        LIBRARY_PATH, _absolute_reference, _anchor_file_references,
        install_theory_index, install_theory_library,
    )

    result = {key: [] for key in ('episodes', 'materials', 'conflicts', 'unique_support_sources',
                                  'theories', 'theory_ids', 'source_refs', 'author_work')}

    def identity(context):
        return (str(context.get('source_id') or context.get('episode'))
                if isinstance(context, dict) and (context.get('source_id') or context.get('episode')) else None)

    def append(field, entry):
        if entry not in result[field]:
            result[field].append(entry)

    for loaded in load_source_values(references, workspace):
        reference = _absolute_reference(loaded['source_ref'], workspace)
        append('source_refs', reference)
        value = loaded['value']
        training, training_ref = None, None
        if isinstance(value, dict) and 'online_questions' in value:
            training, training_ref = value, reference
            source_workspace = Path(value['workspace']).resolve()
            state_ref = {'file': str(source_workspace / 'research_state.json'), 'json_path': []}
            state = load_source_values([state_ref], workspace)[0]['value']
        elif isinstance(value, dict) and 'questions' in value and 'theory_index' in value:
            state, state_ref = value, {'file': reference['file'], 'json_path': _path(reference)}
            source_workspace = Path(reference['file']).parent
            path = source_workspace / 'training.json'
            if path.is_file():
                training_ref = {'file': str(path), 'json_path': []}
                training = load_source_values([training_ref], workspace)[0]['value']
        else:
            append('author_work', {'type': 'AuthorWorkRequest', 'reason': 'common_research_state_required',
                                   'source_ref': reference})
            continue
        observations = (training or {}).get('paid_observations')
        if observations is None:
            path = source_workspace / 'laboratory_observations.json'
            observations = load_source_values([{'file': str(path), 'json_path': []}], workspace)[0]['value']
        criterion_source = (training_ref if training is not None
            and 'confirmation_mse_threshold' in training.get('limits', {}) else state_ref)
        source_limits = (training if criterion_source is training_ref else state).get('limits', {})
        proxy = SimpleNamespace(workspace=source_workspace, questions=state['questions'],
            revisions=state.get('revisions', []), relation_research=state.get('relation_research', {}),
            events=(training or state).get('events', []), observations=observations,
            theory_index=state.get('theory_index', {}), theory_consolidation={},
            config=SimpleNamespace(**source_limits),
            _criterion_source_ref={'file': criterion_source['file'], 'json_path': [*_path(criterion_source),
                'limits', 'confirmation_mse_threshold']})
        snapshots = install_theory_index(proxy, {**state_ref,
            'json_path': [*_path(state_ref), 'theory_index']})
        if (source_workspace / LIBRARY_PATH).is_file():
            proxy.theory_consolidation = install_theory_library(proxy)
        resolved = {theory['id']: theory for theory in resolved_theories(proxy)}
        versions = _question_versions(proxy)
        requested = reference.get('source_theory_ids')
        selected = list(proxy.theory_index) if requested is None else requested
        native, groups = {}, {}
        for identifier in selected:
            declaration = proxy.theory_index.get(identifier, {})
            source = declaration.get('source_ref', {})
            if source.get('kind') != 'native_candidate':
                if requested is not None:
                    append('author_work', {'type': 'AuthorWorkRequest', 'reason': 'registered_native_source_required',
                        'source_ref': reference, 'theory_id': identifier})
                continue
            owner = source['question_id'], source.get('meaning_version', 1)
            native[identifier] = declaration
            groups.setdefault(owner, []).append(identifier)

        def anchored(value):
            if isinstance(value, list):
                return [anchored(item) for item in value]
            if not isinstance(value, dict):
                return deepcopy(value)
            entry = {key: anchored(item) for key, item in value.items()}
            if entry.get('file') == 'research_state.json':
                entry.update(file=state_ref['file'], json_path=[*_path(state_ref), *_path(entry)])
                entry.pop('field', None)
            elif entry.get('file') == 'research_context.json' and training_ref:
                entry.update(file=training_ref['file'], json_path=[*_path(training_ref), *_path(entry)])
                entry.pop('field', None)
            return _anchor_file_references(entry, source_workspace)

        namespace = state_ref['file']
        qualified = lambda identifier: f'{namespace}::{identifier}'
        # First retain the actual banks, failures and revisions even when the
        # caller explicitly selects no theories from this source.
        original_index = proxy.theory_index
        proxy.theory_index = {}
        declared_ids = {identity(item.get('context_ref')) for item in original_index.values()}
        declared_ids.discard(None)
        proxy.research_source_id = next(iter(declared_ids)) if len(declared_ids) == 1 else None
        base = extract_episodes(proxy)
        by_episode = {episode['episode_id']: episode for episode in base['episodes']}
        for owner, identifiers in groups.items():
            retained = versions.get(owner)
            source_ids = {identity(native[identifier].get('context_ref')) for identifier in identifiers}
            source_ids.discard(None)
            if retained:
                question, question_ref = retained
                for context in (question.get('context_ref'), question.get('theory_target', {}).get('context_ref')):
                    if identity(context):
                        source_ids.add(identity(context))
            reason = ('unavailable_retained_meaning' if retained is None else
                      'source_identity_required' if not source_ids else
                      'conflicting_source_identity' if len(source_ids) != 1 else None)
            eligible = []
            retained_failures = []
            for identifier in identifiers:
                theory = resolved.get(identifier, {})
                required = []
                if not reason and theory.get('program') and theory.get('source_status') != 'unavailable_retained_version':
                    for name, _ in Program.from_dict(theory['program']).observed_variables():
                        signatures = theory.get('role_signature') or {}
                        contract = signatures.get(name) if isinstance(signatures, dict) else None
                        required.extend(f'role_signature.{name}.{key}' for key in ('role', 'unit', 'time')
                                        if not isinstance(contract, dict) or contract.get(key) is None)
                    required.extend(key for key in ('output_unit', 'context_ref') if theory.get(key) is None)
                unavailable = not theory.get('program') or theory.get('source_status') == 'unavailable_retained_version'
                if reason or required or unavailable:
                    append('author_work', {'type': 'AuthorWorkRequest',
                        'reason': reason or ('native_program_unavailable' if unavailable else 'typed_source_contract_required'),
                        'theory_id': identifier, 'source_ref': reference, 'required': required})
                    continue
                snapshot = deepcopy(snapshots[identifier])
                snapshot['id'] = qualified(identifier)
                snapshot['episode_ref'] = {'question_id': owner[0], 'meaning_version': owner[1],
                                          'source_ref': anchored(question_ref)}
                for key in ('compatible', 'scope_refutation'):
                    if key in theory:
                        snapshot[key] = anchored(theory[key])
                for failed_ref in theory.get('scope_refutation', {}).get('counterexample_refs', []):
                    failed_ref = anchored(failed_ref)
                    if failed_ref not in snapshot.setdefault('counterexample_refs', []):
                        snapshot['counterexample_refs'].append(failed_ref)
                    if failed_ref not in retained_failures:
                        retained_failures.append(failed_ref)
                append('theories', snapshot)
                append('theory_ids', snapshot['id'])
                eligible.append(identifier)
            if eligible:
                proxy.theory_index = {identifier: original_index[identifier] for identifier in eligible}
                proxy.research_source_id = next(iter(source_ids))
                for episode in extract_episodes(proxy)['episodes']:
                    if any(identifier in eligible for identifier in episode['theory_ids']):
                        episode['failure_refs'].extend(retained_failures)
                        by_episode[episode['episode_id']] = episode
        for episode in by_episode.values():
            episode = anchored(episode)
            account_ref = training_ref if training is not None else state_ref
            episode['cost_refs'].append({**account_ref, 'json_path': [*_path(account_ref), 'costs']})
            episode['cost_scope'] = 'whole_Common_source_account_not_additive_per_question'
            if training is not None and 'status' in training:
                episode['source_outcome_ref'] = {**training_ref,
                    'json_path': [*_path(training_ref), 'status']}
                if training['status'] != 'completed':
                    episode['failure_refs'].append(deepcopy(episode['source_outcome_ref']))
            episode['episode_id'] = qualified(episode['episode_id'])
            episode['theory_ids'] = [qualified(identifier) for identifier in episode['theory_ids']]
            for program_ref in episode['program_refs']:
                if program_ref.get('theory_id'):
                    program_ref['theory_id'] = qualified(program_ref['theory_id'])
            append('episodes', episode)
        for field in ('materials', 'conflicts', 'unique_support_sources'):
            for entry in base[field]:
                entry = anchored(entry)
                if entry.get('episode_id'):
                    entry['episode_id'] = qualified(entry['episode_id'])
                append(field, entry)
    result['status'] = 'UNKNOWN' if result['author_work'] else 'completed'
    return result


def resolve_program_sources(agent, references, retained_episodes=None):
    """Reuse loaded Episodes or live state before reading another small source."""
    state = {'questions': agent.questions, 'revisions': agent.revisions,
             'theory_index': getattr(agent, 'theory_index', {}),
             'relation_research': getattr(agent, 'relation_research', {})}
    criterion = getattr(getattr(agent, 'config', None), 'confirmation_mse_threshold', None)
    if criterion is not None:
        state['limits'] = {'confirmation_mse_threshold': criterion}
    values, missing = [], []
    for original in references:
        reference = _canonical(original)
        retained = next((item for item in retained_episodes or []
                         if _contains_reference(item['source_ref'], reference)), None)
        if retained:
            value = retained['episode']
            path = _path(reference)[len(_path(retained['source_ref'])):]
        elif reference.get('file') == 'research_state.json' and not reference.get('archive'):
            value, path = state, _path(reference)
        elif reference.get('file') == 'research_context.json' and not reference.get('archive'):
            value, path = {'events': agent.events}, _path(reference)
        elif reference.get('file') == 'laboratory_observations.json' and not reference.get('archive'):
            value, path = agent.observations, _path(reference)
        else:
            values.append(None)
            missing.append((len(values) - 1, reference))
            continue
        for component in path:
            value = value[component]
        values.append(value)
    loaded = load_source_values([reference for _, reference in missing], agent.workspace)
    for (index, _), item in zip(missing, loaded):
        values[index] = item['value']
    return values


def _source_id(agent, context=None):
    context = context if isinstance(context, dict) else {}
    return str(context.get('source_id') or context.get('episode')
               or getattr(agent, 'research_source_id', None)
               or Path(agent.workspace).resolve())


def _question_versions(agent):
    versions = {}
    for index, question in enumerate(agent.questions):
        for old_index, old in enumerate(question.get('meaning_history', [])):
            version = old['meaning_version']
            versions[question['id'], version] = (
                old, _ref('questions', index, 'meaning_history', old_index))
        versions[question['id'], question.get('meaning_version', 1)] = (
            question, _ref('questions', index))
    return versions


def _material_source(agent, reference):
    """Normalize a supported reference to the existing immutable source address."""
    kind = reference.get('kind')
    if kind == 'observation':
        index = reference['record_index']
        if not 0 <= index < len(agent.observations):
            raise ValueError('material references an unavailable paid record')
        return {'file': 'laboratory_observations.json', 'json_path': [index]}
    if kind == 'revision':
        index = reference['index']
        if not 0 <= index < len(agent.revisions):
            raise ValueError('material references an unavailable revision')
        return _ref('revisions', index)
    if kind == 'event':
        index = reference['index']
        if not 0 <= index < len(agent.events):
            raise ValueError('material references an unavailable event')
        return {'file': 'research_context.json', 'json_path': ['events', index]}
    if kind == 'question':
        key = reference['question_id'], reference.get('meaning_version', 1)
        if key not in _question_versions(agent):
            raise ValueError('material references an unavailable question meaning')
        return _question_versions(agent)[key][1]
    if kind == 'theory':
        identifier = reference['theory_id']
        theory = agent.theory_index[identifier]
        version = reference.get('version', theory.get('version', 1))
        if version == theory.get('version', 1):
            return _ref('theory_index', identifier)
        for index, old in enumerate(theory.get('history', [])):
            if old['version'] == version:
                return _ref('theory_index', identifier, 'history', index)
        raise ValueError('material references an unavailable theory version')
    # External documents and already-addressed retained files are references,
    # not newly measured or executed evidence.
    if reference.get('file'):
        return _canonical(reference)
    if all(reference.get(key) is not None for key in ('archive', 'unit', 'member')):
        return _canonical(reference)
    raise ValueError('material needs an existing source_ref')


def _record_references(agent, indices):
    return [{'file': 'laboratory_observations.json', 'json_path': [index]}
            for index in sorted(set(indices)) if 0 <= index < len(agent.observations)]


def _new_episode(identifier, source_id, question, reference):
    return {'episode_id': identifier, 'source_id': source_id,
        'question_ref': deepcopy(reference),
        'original_question_semantics': {key: deepcopy(question[key])
            for key in SEMANTIC_FIELDS if key in question},
        'meaning_version': question.get('meaning_version', 1),
        'theory_ids': [], 'program_refs': [],
        'initial_candidates': [],
        'support_premises': deepcopy(question.get('required_premises', [])),
        'change_evidence': [], 'actual_changes': [], 'behavior_changes': [],
        'independent_results': [], 'cost_refs': [], 'failure_refs': [],
        'adoption': {'occurred': False, 'execution_refs': []},
        'status': question.get('status', 'candidate'),
        'unresolved_scope': deepcopy(question.get('pending_author_work')
            or question.get('unresolved_scope') or {'status': 'not_independently_confirmed'})}


def extract_episodes(agent, materials=None, retained_episodes=None, *, excluded_theory_ids=()):
    """Extract real question meanings and classify only the supplied materials.

    Returns small episode/material records without mutating the agent or reading
    model files. ``materials`` use ``kind``, ``source_ref`` and optional ``claim``;
    checked revision references reuse the saved numerical check verbatim.
    """
    episodes, versions = {}, _question_versions(agent)

    def episode(question_id, meaning=1):
        key = question_id, meaning
        if key not in episodes:
            question, reference = versions.get(key, (
                {'id': question_id, 'meaning_version': meaning, 'status': 'unresolved'},
                {'question_id': question_id, 'meaning_version': meaning,
                 'status': 'unavailable_retained_meaning'}))
            episodes[key] = _new_episode(f'{question_id}@meaning{meaning}',
                _source_id(agent, question.get('context_ref')), question, reference)
        return episodes[key]

    retained_materials, retained_by_task, source_refs_by_task = [], {}, {}
    for retained in retained_episodes or []:
        references = source_refs_by_task.setdefault(retained['episode']['task_id'], [])
        reference = _canonical(retained['source_ref'])
        if reference not in references:
            references.append(reference)
    processed_references = []
    for retained in retained_episodes or []:
        original, reference = retained['episode'], _canonical(retained['source_ref'])
        if reference in processed_references:
            continue
        processed_references.append(reference)
        task_id = original['task_id']
        ambiguous = retained.get('ambiguous_task_id') or len(source_refs_by_task[task_id]) > 1
        context = next((item.get('context_ref', {}) for item in getattr(agent, 'theory_index', {}).values()
            if item.get('episode_ref', {}).get('question_id') == task_id
            and (not ambiguous or _contains_reference(reference,
                item.get('episode_ref', {}).get('source_ref') or item.get('program_source_ref') or {}))), {})
        source_id = (context.get('source_id') or context.get('episode')
                     or f'{reference.get("member", reference.get("file"))}/{task_id}')
        suffix = f'@source:{_source_suffix(reference)}' if ambiguous else ''
        if ambiguous:
            source_id = f'{source_id}{suffix}'
        questions = original.get('questions', [])
        cutoff = reference.get('action_cutoff')
        question = {'meaning_version': 1, 'status': ('historical_prefix' if cutoff is not None
                    else original.get('status', 'unresolved'))}
        if questions:
            question['question'] = questions[0].get('question')
        current = _new_episode(f'{task_id}@meaning1{suffix}', str(source_id), question,
                               _child_ref(reference, 'questions'))
        current['source_ref'] = reference
        current['cost_refs'] = [_child_ref(reference, key) for key in ('observation_cost', 'compute_cost')
                                if key in original]
        actions = original.get('actions', [])
        if cutoff is None:
            current['costs'] = {key: original[key] for key in ('observation_cost', 'compute_cost') if key in original}
            current['unresolved_scope'] = {'stop_reason': original.get('stop_reason'),
                'status': 'historical_evaluation_only', 'source_ref': _child_ref(reference, 'stop_reason')}
        else:
            if type(cutoff) is not int or not 0 <= cutoff < len(actions):
                raise ValueError('action_cutoff must identify an actual retained action')
            current['source_cutoff'] = {'action_index_inclusive': cutoff}
            current['costs'] = {key: sum(action.get(key, 0) for action in actions[:cutoff + 1])
                                for key in ('observation_cost', 'compute_cost')}
            current['cost_scope'] = 'recorded_action_costs_through_cutoff'
            current['unresolved_scope'] = {'status': 'cutoff_program_not_independently_confirmed'}
            current['retained_later_outcome_refs'] = [_child_ref(reference, 'actions'),
                _child_ref(reference, 'evaluation'), _child_ref(reference, 'status')]
        current['program_refs'].append({'source_ref': _child_ref(reference, 'program'),
            'status': 'later_final_program_not_cutoff_program' if cutoff is not None else 'retained_final_program'})
        for index, action in enumerate(actions if cutoff is None else actions[:cutoff + 1]):
            action_ref = _child_ref(reference, 'actions', index)
            retained_materials.append((action_ref, 'execution'))
            details, kind = action.get('details', {}), action.get('action')
            if kind == 'observe':
                current['change_evidence'].append(action_ref)
                retained_materials.append((action_ref, 'measured'))
            if kind == 'revise' and 'before_program' in details and 'after_program' in details:
                before_ref = _child_ref(action_ref, 'details', 'before_program')
                after_ref = _child_ref(action_ref, 'details', 'after_program')
                if not current['initial_candidates']:
                    current['initial_candidates'].append(before_ref)
                if details['before_program'] != details['after_program']:
                    current['actual_changes'].append({'kind': 'program_changed',
                        'before_ref': before_ref, 'source_ref': after_ref,
                        'action_ref': action_ref, 'operation': details.get('operation')})
                current['adoption']['occurred'] = True
                current['adoption']['execution_refs'].append(action_ref)
            if kind in {'compare', 'prediction_check', 'reuse_check'}:
                result = {'kind': kind, 'source_ref': action_ref}
                result.update({key: deepcopy(details[key]) for key in ('accepted', 'adopt', 'loss',
                    'best_loss', 'revision_loss', 'parent_loss', 'selection_policy',
                    'validation_is_adaptive_development') if key in details})
                current['behavior_changes'].append(result)
                if details.get('accepted') is False or details.get('status', '').endswith('_failed'):
                    current['failure_refs'].append(action_ref)
            if action.get('error') or details.get('error') or details.get('status', '').endswith('_failed'):
                current['failure_refs'].append(action_ref)
        evaluation = original.get('evaluation')
        if evaluation:
            current['independent_results'].append({'kind': ('later_final_evaluation' if cutoff is not None
                                                           else 'retained_final_evaluation'),
                'source_ref': _child_ref(reference, 'evaluation'),
                **({'loss': evaluation.get('loss'), 'rows': evaluation.get('rows')} if cutoff is None else {}),
                'independent_confirmation': False,
                'scope': ('later outcome; not a score for the cutoff Program' if cutoff is not None
                          else 'saved final Episode evaluation; no new acquisition')})
        if cutoff is None and original.get('status') in {'failed', 'refuted', 'budget_exhausted'}:
            current['failure_refs'].append(_child_ref(reference, 'status'))
        episodes[task_id + suffix, 1] = current
        retained_by_task.setdefault(task_id, []).append((current, bool(ambiguous)))

    # Declarations supply typed interfaces and source addresses. They do not
    # manufacture executed changes, independent results or scientific support.
    for identifier, theory in getattr(agent, 'theory_index', {}).items():
        if identifier in excluded_theory_ids:
            continue
        source = theory.get('source_ref', {})
        owner = source.get('question_id') or theory.get('episode_ref', {}).get('question_id')
        meaning = source.get('meaning_version', theory.get('episode_ref', {}).get('meaning_version', 1))
        retained_choices = retained_by_task.get(owner, [])
        locator = theory.get('episode_ref', {}).get('source_ref') or theory.get('program_source_ref')
        matching = [item for item, _ in retained_choices if locator
                    and _contains_reference(item['source_ref'], locator)]
        unresolved_binding = bool(retained_choices) and not matching and (
            locator is not None or any(ambiguous for _, ambiguous in retained_choices))
        if matching:
            current = matching[0]
        elif unresolved_binding:
            key = f'unbound:{identifier}', meaning
            current = episodes.setdefault(key, _new_episode(f'unbound:{identifier}@meaning{meaning}',
                _source_id(agent, theory.get('context_ref')), {'status': 'UNKNOWN'},
                _ref('theory_index', identifier)))
            current['unbound_theory_refs'] = [_ref('theory_index', identifier)]
            current['unresolved_scope'] = {'status': 'UNKNOWN', 'reason': 'ambiguous_retained_episode',
                'requested_source_ref': deepcopy(locator),
                'retained_source_refs': [deepcopy(item['source_ref']) for item, _ in retained_choices]}
        elif owner:
            current = episode(owner, meaning)
        else:
            key = f'candidate:{identifier}', theory.get('version', 1)
            current = episodes.setdefault(key, _new_episode(
                f'candidate:{identifier}@version{theory.get("version", 1)}',
                _source_id(agent, theory.get('context_ref')), {},
                _ref('theory_index', identifier)))
        if not unresolved_binding:
            current['theory_ids'].append(identifier)
        if 'program' in theory:
            reference = (deepcopy(theory.get('program_source_ref'))
                         or _ref('theory_index', identifier, 'program'))
        else:
            reference = deepcopy(source)
        current['program_refs'].append({'theory_id': identifier, 'source_ref': reference,
            'version': theory.get('version', 1), 'status': 'retained_candidate'})
        current['initial_candidates'].append(_ref('theory_index', identifier))

    # Retain each old meaning, even if its latest conclusion was negative.
    for (identifier, meaning), (question, reference) in versions.items():
        if not any(question.get(field) for field in (
                'native_condition_research', 'revision_index', 'parent_use_history',
                'executed_pending_queries', 'research_return', 'semantic_answer',
                'check_results', 'failure_event_indices')):
            continue
        current = episode(identifier, meaning)
        previous_programs = []
        for round_index, round_ in enumerate(question.get('native_condition_research', [])):
            base = {**reference, 'json_path': [*reference['json_path'],
                    'native_condition_research', round_index]}
            current['change_evidence'].extend(_record_references(agent, round_.get('fit_record_indices', [])))
            programs = []
            for candidate_index, candidate in enumerate(round_.get('candidates', [])):
                program_ref = {**base, 'json_path': [*base['json_path'], 'candidates', candidate_index, 'program']}
                current['program_refs'].append({'source_ref': program_ref,
                    'status': candidate.get('status', round_.get('status', 'candidate'))})
                if round_index == 0:
                    current['initial_candidates'].append(program_ref)
                elif candidate.get('program') not in previous_programs:
                    current['actual_changes'].append({'kind': 'program_changed', 'source_ref': program_ref})
                programs.append(candidate.get('program'))
            previous_programs = programs
            if round_.get('status') in {'refuted', 'search_failed', 'search_incomplete'}:
                current['failure_refs'].append(base)
        for index, executed in enumerate(question.get('executed_pending_queries', [])):
            current['behavior_changes'].append({'kind': 'executed_parent_query',
                'source_ref': {**reference, 'json_path': [*reference['json_path'],
                    'executed_pending_queries', index]}})
        # Actual parent evaluation/refitting records adoption. A removed plan
        # records a behavior change; a prepared query still needs execution.
        for index, applied in enumerate(question.get('parent_use_history', [])):
            results = applied.get('results', [])
            execution_ref = {**reference, 'json_path': [*reference['json_path'],
                'parent_use_history', index]}
            if any(result.get('status') in {'parent_refitted', 'parent_component_evaluated'}
                   for result in results):
                current['adoption']['occurred'] = True
                current['adoption']['execution_refs'].append(execution_ref)
            if any(result.get('status') == 'parent_plan_stopped' and result.get('changed')
                   for result in results):
                current['behavior_changes'].append({'kind': 'parent_plan_stopped',
                    'source_ref': execution_ref})
        if question.get('failure_event_indices'):
            current['failure_refs'].extend({'file': 'research_context.json',
                'json_path': ['events', index]} for index in question['failure_event_indices'])

    for bank_id, bank in getattr(agent, 'relation_research', {}).items():
        if '@meaning' in bank_id:
            identifier, old_meaning = bank_id.rsplit('@meaning', 1)
            meaning = int(old_meaning)
        else:
            identifier = bank_id
            question = next((q for q in agent.questions if q['id'] == identifier), {})
            meaning = question.get('meaning_version', 1)
        current = episode(identifier, meaning)
        reference = _ref('relation_research', bank_id)
        current['change_evidence'].extend(_record_references(agent, bank.get('fit_record_indices', [])))
        for candidate_index, candidate in enumerate(bank.get('candidates', [])):
            current['program_refs'].append({'source_ref': _ref('relation_research', bank_id,
                'candidates', candidate_index, 'program'),
                'status': candidate.get('status', bank.get('status', 'candidate'))})
        if bank.get('status') == 'refuted':
            current['failure_refs'].append(reference)

    conflicts = []
    for index, revision in enumerate(getattr(agent, 'revisions', [])):
        identifier = revision.get('question_id')
        if not identifier:
            continue
        current = episode(identifier, revision.get('question_meaning_version', 1))
        reference = _ref('revisions', index)
        record_indices = revision.get('record_indices', [])
        current['change_evidence'].extend(_record_references(agent, record_indices))
        current['cost_refs'].append(reference)
        if revision.get('forecasts'):
            current['behavior_changes'].append({'kind': 'frozen_predictions', 'source_ref': reference})
            cutoff = revision.get('evidence_cutoff', {}).get('records')
            fresh = cutoff is not None and bool(record_indices) and all(i >= cutoff for i in record_indices)
            for forecast_index, forecast in enumerate(revision['forecasts']):
                if forecast.get('complete_experiment') and forecast.get('prospective_mse') is not None:
                    current['independent_results'].append({'kind': 'frozen_prospective_result',
                        'source_ref': _ref('revisions', index, 'forecasts', forecast_index),
                        'independent_from_fit': fresh,
                        'independent_confirmation': False,
                        'prospective_mse': forecast['prospective_mse'],
                        'record_refs': _record_references(agent, record_indices)})
        if (revision.get('error') or revision.get('readout_error')
                or revision.get('acquisition_error') or revision.get('evidence_status') == 'refuted'
                or revision.get('status', '').endswith('_failed')):
            current['failure_refs'].append(reference)
        authored = revision.get('authored_answer', {})
        for check_index, check in enumerate(revision.get('check_results', [])):
            if check.get('status') == 'violated' and authored.get('status') == 'supported':
                conflicts.append({'kind': 'claim_numeric_conflict', 'episode_id': current['episode_id'],
                    'claim_ref': _ref('revisions', index, 'authored_answer'),
                    'numeric_check_ref': _ref('revisions', index, 'check_results', check_index),
                    'authored_status': 'supported', 'check_status': 'violated',
                    'actual_value': check.get('value'), 'expected': check.get('expected')})

    # Export paths identify saved native Programs, independently of ranked slots.
    native_models = {}
    for identity, (question, reference) in versions.items():
        models = native_models.setdefault(identity, {})
        for round_index, round_ in enumerate(question.get('native_condition_research', [])):
            for candidate_index, candidate in enumerate(round_.get('candidates', [])):
                if candidate.get('model') and candidate.get('program') is not None:
                    models.setdefault(candidate['model'], []).append((candidate['program'],
                        _child_ref(reference, 'native_condition_research', round_index,
                                   'candidates', candidate_index, 'program')))

    # Real executed evaluation is distinct from a candidate declaration and
    # from a text note saying it was adopted.
    for index, event in enumerate(getattr(agent, 'events', [])):
        feedback, arguments = event.get('feedback', {}), event.get('arguments', {})
        identifier = feedback.get('research_question_id')
        meaning = feedback.get('question_meaning_version', 1)
        if not identifier:
            continue
        current = episode(identifier, meaning)
        reference = {'file': 'research_context.json', 'json_path': ['events', index]}
        if feedback.get('operator') == 'research_choice':
            models = native_models.get((identifier, meaning), {})
            for evaluation_index, evaluation in enumerate(feedback.get('evaluations', [])):
                decision = evaluation.get('retention_decision', {})
                incumbent, candidate = decision.get('incumbent') or {}, decision.get('candidate') or {}
                before = models.get(incumbent.get('source_model'), [])
                after = models.get(candidate.get('model'), [])
                paid_rows = candidate.get('evaluated_rows')
                if (decision.get('status') != 'candidate_retained'
                        or decision.get('same_paid_rows') is not True
                        or evaluation.get('returncode') != 0
                        or len(before) != 1 or len(after) != 1
                        or before[0][0] == after[0][0]
                        or type(paid_rows) is not int or paid_rows <= 0
                        or incumbent.get('evaluated_rows') != paid_rows
                        or feedback.get('paid_rows') != paid_rows):
                    continue
                records, rows = [], 0
                for record_index, record in enumerate(getattr(agent, 'observations', [])):
                    rows += len(record.get('rows', []))
                    if rows > paid_rows:
                        break
                    records.append(record_index)
                    if rows == paid_rows:
                        break
                if rows != paid_rows:
                    continue
                action_ref = _child_ref(reference, 'feedback', 'evaluations', evaluation_index,
                                        'retention_decision')
                evidence_refs = _record_references(agent, records)
                cost_refs = [_child_ref(reference, 'feedback', key) for key in
                             ('compute_cost', 'python_execution_seconds') if key in feedback]
                # Reuse the actual same-prefix score, not a prediction rerun or
                # the cached last adoption summary. Missing checks stay unknown.
                before_scores = [(position, score) for position, score in enumerate(feedback['evaluations'])
                    if score.get('path') == incumbent.get('source_model')
                    and score.get('returncode') == 0 and score.get('loss_kind') == 'mse'
                    and score.get('mse') == incumbent.get('mse')
                    and score.get('paid_residuals', {}).get('evaluated_rows') == paid_rows]
                comparison = {'status': 'UNKNOWN', 'reason': 'unique_saved_before_paid_evaluation_required'}
                if len(before_scores) == 1:
                    comparison = {'status': 'UNKNOWN', 'reason': 'saved_after_paid_mse_evaluation_required'}
                    if (evaluation.get('loss_kind') == 'mse'
                            and evaluation.get('paid_residuals', {}).get('evaluated_rows') == paid_rows):
                        comparison = {'status': 'UNKNOWN', 'reason': 'original_paid_mse_criterion_required'}
                        if getattr(getattr(agent, 'config', None), 'confirmation_mse_threshold', None) is not None:
                            comparison = {'before_evaluation_ref': _child_ref(reference,
                                'feedback', 'evaluations', before_scores[0][0]),
                                'after_evaluation_ref': _child_ref(reference, 'feedback', 'evaluations', evaluation_index),
                                'criterion_ref': deepcopy(getattr(agent, '_criterion_source_ref',
                                    _ref('limits', 'confirmation_mse_threshold')))}
                current['actual_changes'].append({'kind': 'retained_program_replaced',
                    'before_ref': deepcopy(before[0][1]), 'source_ref': deepcopy(after[0][1]),
                    'action_ref': action_ref, 'evidence_refs': evidence_refs, 'cost_refs': cost_refs,
                    'source_comparison': comparison,
                    'cost_scope': 'whole_research_choice_not_additive_per_revision',
                    'scope': {'decision_scope': decision.get('scope'),
                        'same_paid_rows': True, 'evaluated_rows': paid_rows,
                        'paid_cutoff': {'records': len(records), 'rows': paid_rows},
                        'record_indices': records,
                        'before_evaluation_scope': incumbent.get('evaluation_scope'),
                        'after_evaluation_scope': candidate.get('evaluation_scope'),
                        'independent_confirmation': False}})
                current['change_evidence'].extend(evidence_refs)
                current['cost_refs'].extend(cost_refs)
                current['adoption']['occurred'] = True
                current['adoption']['execution_refs'].append(action_ref)
        if feedback.get('error') or feedback.get('returncode', 0) != 0:
            current['failure_refs'].append(reference)
        if (event.get('action') in {'evaluate', 'finish'} and feedback.get('returncode') == 0
                and ('mse' in feedback or feedback.get('fresh_confirmation'))):
            current['adoption']['occurred'] = True
            current['adoption']['execution_refs'].append(reference)
        confirmation = feedback.get('fresh_confirmation')
        if confirmation is not None:
            current['independent_results'].append({'kind': 'independent_confirmation',
                'source_ref': {**reference, 'json_path': [*reference['json_path'], 'feedback', 'fresh_confirmation']},
                'independent_confirmation': True, 'passed': confirmation.get('passed')})

    intake, seen = [], []
    for index, material in enumerate(materials or []):
        classification = material.get('kind', 'unresolved')
        if classification not in MATERIAL_KINDS:
            raise ValueError(f'unknown MaterialIntake kind: {classification}')
        original_ref = material['source_ref']
        reference = _material_source(agent, original_ref)
        # Evidence categories require the corresponding actual record. A cited
        # article is still an external claim if its author calls it measured.
        retained_kind = any(ref == reference and kind == classification for ref, kind in retained_materials)
        if classification == 'measured' and original_ref.get('kind') != 'observation' and not retained_kind:
            classification = 'external_claim'
        if classification == 'execution' and original_ref.get('kind') not in {'event', 'revision'} and not retained_kind:
            classification = 'author_interpretation'
        duplicate = next((entry for entry in seen if entry['source_ref'] == reference), None)
        entry = {'id': material.get('id', f'material:{index}'), 'kind': 'duplicate' if duplicate else classification,
            'original_kind': classification, 'source_ref': reference,
            'source_id': _source_id(agent), 'claim': deepcopy(material.get('claim')),
            'adds_independent_support': False}
        if duplicate:
            entry['duplicate_of'] = duplicate['id']
        else:
            seen.append(entry)
        if original_ref.get('kind') == 'revision':
            revision = agent.revisions[original_ref['index']]
            for check_index, check in enumerate(revision.get('check_results', [])):
                claim = material.get('claim')
                if (check.get('status') == 'violated' and isinstance(claim, dict)
                        and claim.get('status') == 'supported'):
                    conflicts.append({'kind': 'claim_numeric_conflict', 'material_id': entry['id'],
                        'claim_ref': {'material_id': entry['id'], 'field': 'claim'},
                        'numeric_check_ref': _ref('revisions', original_ref['index'], 'check_results', check_index),
                        'authored_status': 'supported', 'check_status': 'violated',
                        'actual_value': check.get('value'), 'expected': check.get('expected')})
        intake.append(entry)

    for current in episodes.values():
        # Keep source addresses once without expanding their payloads.
        for field in ('change_evidence', 'failure_refs', 'cost_refs'):
            unique = []
            for reference in current[field]:
                if reference not in unique:
                    unique.append(reference)
            current[field] = unique
    support_sources = []
    for current in episodes.values():
        for reference in current['change_evidence']:
            source = {'source_id': (current['source_id'] if current.get('source_ref') else _source_id(agent)),
                      'source_ref': reference}
            if source not in support_sources:
                support_sources.append(source)
    for entry in intake:
        if entry['source_ref'].get('file') == 'laboratory_observations.json':
            source = {'source_id': _source_id(agent), 'source_ref': entry['source_ref']}
            if source not in support_sources:
                support_sources.append(deepcopy(source))
    return {'episodes': list(episodes.values()), 'materials': intake, 'conflicts': conflicts,
        'unique_support_sources': support_sources,
        'support_count_scope': 'distinct retained paid records; not independent confirmations'}
