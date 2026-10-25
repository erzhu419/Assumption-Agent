"""Bind an actual retained fixed native export through its original addresses."""

from copy import deepcopy
from pathlib import Path

from .language import Program
from .question_routing import register_theories
from .research_episodes import _child_ref, load_source_values
from .temporal_operator_agent import export_temporal_program
from .theory_consolidation import _absolute_reference, _snapshot_index


def bind_retained_native_parent(agent, question):
    """Register a scoped ordinary Program, without fitting or inheriting support."""
    target = question.get('theory_target') or {}
    retained = agent.evaluated_model or {}
    selected = retained.get('source_model')
    retained_ref = {'file': 'research_state.json', 'json_path': ['evaluated_model']}

    def request(reason, **details):
        return {'status': 'author_work_request', 'type': 'AuthorWorkRequest',
            'reason': reason, 'source_model': selected, 'source_ref': retained_ref,
            'new_compute_cost': 0, 'support_updated': False, **details}

    if not selected or not retained.get('model'):
        return request('actual_retained_model_required')
    reason = 'retained_native_catalog_unavailable'
    try:
        catalog_ref = {'file': str((agent.workspace / 'source_catalog.json').resolve()), 'json_path': []}
        catalog, = load_source_values([catalog_ref], agent.workspace)
        matches = []
        for index, source in enumerate(catalog['value']):
            for position, mapping in enumerate(source.get('model_sources', [])):
                if mapping.get('model') == selected:
                    matches.append((source, mapping, _child_ref(catalog_ref, index, 'model_sources', position)))
        if len(matches) != 1:
            return request('unique_retained_native_catalog_mapping_required', matching_entries=len(matches))
        source, mapping, mapping_ref = matches[0]
        if (not source.get('state_ref') or not mapping.get('source_ref', {}).get('file')
                or not mapping.get('source_model')):
            return request('retained_native_source_addresses_required', catalog_ref=mapping_ref)
        state_ref = _absolute_reference(source['state_ref'], agent.workspace)
        export_ref = _absolute_reference(mapping['source_ref'], Path(state_ref['file']).parent)
        reason = 'retained_native_source_unavailable'
        loaded, = load_source_values([state_ref], agent.workspace)
        state = loaded['value']
        candidates = []
        for qi, current in enumerate(state.get('questions', state.get('online_questions', []))):
            versions = [(current, ['questions', qi])]
            versions.extend((old, ['questions', qi, 'meaning_history', hi])
                            for hi, old in enumerate(current.get('meaning_history', [])))
            for owner, owner_path in versions:
                for ri, round_ in enumerate(owner.get('native_condition_research', [])):
                    for ci, candidate in enumerate(round_.get('candidates', [])):
                        if candidate.get('model') == mapping['source_model']:
                            candidates.append((owner, owner_path, ri, ci, candidate))
        if len(candidates) != 1:
            return request('unique_original_native_export_required', matching_candidates=len(candidates),
                           catalog_ref=mapping_ref)
        owner, owner_path, ri, ci, candidate = candidates[0]
        entries = [(identifier, entry) for identifier, entry in state.get('theory_index', {}).items()
            if entry.get('source_ref') == {'kind': 'native_candidate', 'question_id': owner['id'],
                'meaning_version': owner.get('meaning_version', 1), 'round_index': ri, 'candidate_index': ci}]
        if len(entries) != 1:
            return request('unique_registered_source_native_required', matching_entries=len(entries),
                           catalog_ref=mapping_ref)
        source_id, entry = entries[0]
        index_ref = _child_ref(state_ref, 'theory_index')
        native = _snapshot_index({source_id: entry}, state, index_ref)[source_id]
        program = Program.from_dict(native['program'])
        roles, wanted = native.get('role_signature', {}), target.get('role_signature', {})
        names = program.required_features()
        if (not isinstance(roles, dict) or not isinstance(wanted, dict)
                or target.get('context_ref') is None or native.get('output_unit') is None
                or target.get('output_unit') != native['output_unit']
                or any(not isinstance(roles.get(name), dict)
                       or any(roles[name].get(key) is None for key in ('role', 'unit', 'time'))
                       or wanted.get(name) != roles[name] for name in names)):
            return request('retained_native_target_contract_mismatch', catalog_ref=mapping_ref)
        reason = 'retained_native_export_unavailable'
        original = Path(export_ref['file']).read_bytes()
        imported = (agent.workspace / selected).read_bytes()
        snapshot = (agent.workspace / agent._evaluated_summary()['model']).read_bytes()
        if original != imported or imported != snapshot:
            return request('retained_native_export_bytes_changed', catalog_ref=mapping_ref)
        if original != export_temporal_program(program, feature_names=tuple(roles)).encode('utf-8'):
            return request('retained_native_export_not_fixed', catalog_ref=mapping_ref)
        identifier = (f'retained_native_parent:{question["id"]}@meaning{question.get("meaning_version", 1)}:'
                      f'{state_ref["file"]}::{source_id}')
        declaration = {'id': identifier, 'program': native['program'],
            'program_source_ref': _child_ref(state_ref, *owner_path,
                'native_condition_research', ri, 'candidates', ci, 'program'),
            'source_snapshot': {**native['source_snapshot'], 'theory_id': source_id,
                'catalog_ref': mapping_ref, 'export_ref': export_ref},
            'role_signature': deepcopy(roles), 'output_unit': native['output_unit'],
            'context_ref': deepcopy(target['context_ref']),
            'support': {'status': 'candidate', 'evidence': [], 'independent_confirmation': False}}
        if target.get('context_family_ref') is not None:
            declaration['context_family_ref'] = deepcopy(target['context_family_ref'])
        previous = agent.theory_index.get(identifier)
        if (previous and all(previous.get(key) == declaration[key] for key in
                ('program', 'program_source_ref', 'source_snapshot', 'role_signature', 'output_unit', 'context_ref'))):
            # Retain any actual local support/refutation added since registration.
            declaration = {key: deepcopy(value) for key, value in previous.items()
                           if key not in {'version', 'history'}}
        register_theories(agent, [declaration])
        return {'status': 'bound', 'theory_id': identifier,
            'version': agent.theory_index[identifier]['version'], 'source_ref': declaration['program_source_ref'],
            'source_version': entry.get('version', 1), 'catalog_ref': mapping_ref,
            'new_compute_cost': 0, 'support_updated': False}
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        return request(reason, detail=str(error))
