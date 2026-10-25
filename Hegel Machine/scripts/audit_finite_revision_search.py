"""Post-hoc typed joint-contract classification of actual saved proposals.

Targets are audit inputs only.  No synthesis, fitting, world prediction or
candidate selection occurs here.  A first complete paired witness stops the
search for that target; an unfinished bounded audit stays UNKNOWN.
"""

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.revision_relations import (
    _Incomplete, _MissingContract, _bound_proposal, _match_parent, _prepare, _spend,
)


def _program(entry):
    value = entry['program']
    return (value if isinstance(value, Program) else Program.from_dict(value)).expanded()


def _reference(entry):
    return deepcopy(entry.get('program_source_ref') or entry.get('source_ref'))


def _identity(entry):
    return {'id': entry.get('id'), 'version': entry.get('version'),
            'source_ref': deepcopy(entry.get('source_ref')),
            'program_source_ref': _reference(entry)}


def _entry_summary(entry):
    return {'theory': _identity(entry), 'scientific_status': entry.get('source_status'),
            'support_status': entry.get('support', {}).get('status'),
            'compatible': entry.get('compatible'),
            'scope_refutation': deepcopy(entry.get('scope_refutation')),
            'transfer_ref': deepcopy(entry.get('transfer_ref'))}


def _family(entry, budget):
    if entry.get('context_family_ref') is not None:
        return entry['context_family_ref'], None
    declaration = entry.get('contextual_scope')
    reference = entry.get('contextual_scope_ref')
    if declaration is None and reference:
        _spend(budget)
        declaration = load_source_values([reference], Path(reference['file']).parent)[0]['value']
    if (declaration and declaration.get('context_ref') == entry.get('context_ref')
            and (reference or declaration.get('source_ref'))):
        return declaration.get('context_family_ref'), reference or declaration['source_ref']
    return None, None


def _declared_parent(parent, entry, budget):
    _spend(budget)
    declared = entry.get('consolidation_ref', {}).get('parent')
    if not declared:
        return {'status': 'UNKNOWN', 'exact': False,
                'reason': 'saved_generation_parent_required', 'resolved_parent': _identity(parent)}
    exact = (declared.get('component_id') == parent.get('id')
             and declared.get('version') == parent.get('version')
             and declared.get('source_ref') == parent.get('source_ref'))
    # No Program is inferred from an equal identifier.  The passed parent is
    # the caller's resolved actual component; compare the historical snapshot
    # independently, when supplied, and expose both physical references.
    historical = parent.get('historical_before')
    same_program = None
    if historical is not None:
        _spend(budget, _program(parent).size + _program(historical).size)
        same_program = _program(parent) == _program(historical)
    return {'status': 'matched_resolved_identity' if exact else 'UNKNOWN', 'exact': exact,
        **({'reason': 'declared_generation_parent_identity_differs'} if not exact else {}),
        'declared': deepcopy(declared), 'resolved_parent': _identity(parent),
        'historical_before_ref': _reference(historical) if historical else None,
        'resolved_program_equal_historical_before': same_program,
        'binding_contains_independent_program_snapshot': False}


def _scope_checks(parent, entry, target, budget):
    _spend(budget)
    if target.get('kind') != 'program_revision_relation':
        raise _MissingContract('joint_revision_target_required')
    if not any(check.get('before', {}).get('failed') is True
               for check in target.get('trigger_checks', [])):
        raise _MissingContract('source_measured_discrepancy_required')
    if target.get('parameter_relations') or target.get('blocked_bindings'):
        raise _MissingContract('additional_parameter_or_binding_domain_requires_instance_checker')
    if target.get('trigger_obligations'):
        raise _MissingContract('unresolved_source_trigger_obligation')
    if parent.get('context_ref') is None or parent['context_ref'] != entry.get('context_ref'):
        raise _MissingContract('actual_before_after_context_mismatch')
    if any(item.get('output_unit') != target.get('output_unit') for item in (parent, entry)):
        raise _MissingContract('actual_before_after_output_unit_mismatch')
    scope = target.get('scope', {})
    if set(scope) - {'context_ref', 'context_family_ref'}:
        raise _MissingContract('unsupported_additional_scope_contract')
    family, family_source = _family(parent, budget)
    after_family, after_family_source = _family(entry, budget)
    for key, value in scope.items():
        actual = (family, after_family) if key == 'context_family_ref' else (
            parent.get(key), entry.get(key))
        if value is None or any(item != value for item in actual):
            raise _MissingContract('actual_context_family_or_scope_required')
    for name, contract in target.get('fixed_role_signature', {}).items():
        _spend(budget)
        if any(item.get('role_signature', {}).get(name) != contract for item in (parent, entry)):
            raise _MissingContract('actual_fixed_role_unit_time_mismatch')
    return {'status': 'passed', 'context_ref': deepcopy(parent['context_ref']),
            'context_family_ref': deepcopy(family),
            'parent_context_family_source_ref': deepcopy(family_source),
            'after_context_family_source_ref': deepcopy(after_family_source),
            'output_unit': target['output_unit'], 'measured_source_failure_gate': True}


def _matched(pattern, actual, holes, budget, bindings=None, paths=None):
    try:
        result, result_paths = _match_parent(pattern, actual, holes, budget,
                                             bindings=bindings, paths=paths)
        return {'status': 'matched'}, result, result_paths
    except _MissingContract as error:
        return {'status': 'not_matched', 'reason': str(error)}, None, None


def _prepared(entry, budget):
    # Only the actual component's declarations supply units here.  Target hole
    # units never fill a missing actual internal unit.
    program = _program(entry)
    source = entry.get('subtree_contract_source')
    if source is None:
        return program, _prepare(entry, program, budget)
    keys = ('candidate', 'binding', 'program')
    missing = [key for key in keys if key not in source]
    values = {key: source[key] for key in keys if key in source}
    if missing:
        refs = [source[f'{key}_ref'] for key in missing]
        _spend(budget, len(refs))
        resolved = load_source_values(refs, Path(refs[0]['file']).parent)
        values.update((key, value['value']) for key, value in zip(missing, resolved))
    original = _program({'program': values['program']})
    _spend(budget, original.size + program.size)
    candidate, binding = values['candidate'], values['binding']
    if (original != program or candidate.get('id') != binding.get('candidate_id')
            or candidate.get('version') != binding.get('candidate_version')):
        raise _MissingContract('actual_original_constructor_binding_or_program_mismatch')
    record = {**entry, 'after_ref': _reference(entry), '_after_contract_source': {
        '_pattern': Definition.from_dict(candidate['definition']).body.to_dict(),
        '_holes': candidate['holes'], '_bound_holes': binding.get('components', {}),
        'source_ref': {'program_ref': _reference(entry),
            'original_program_ref': deepcopy(source['program_ref']),
            'pattern_ref': {**deepcopy(source['candidate_ref']), 'json_path': [
                *source['candidate_ref'].get('json_path', []), 'definition', 'body']},
            'binding_ref': deepcopy(source['binding_ref'])}}}
    return program, _prepare(record, program, budget, side='after')


def _pair(parent, entry, target, before_program, after_program,
          before_prepared, after_prepared, budget):
    scope = _scope_checks(parent, entry, target, budget)
    before_pattern = Program.from_dict(target['before_pattern'])
    after_pattern = Definition.from_dict(target['definition']).body
    check, bindings, before_paths = _matched(before_pattern, before_program,
                                            target['holes'], budget)
    if bindings is None:
        return {'status': 'not_matched', 'stage': 'before', 'reason': check['reason']}
    before_paths = deepcopy(before_paths)
    check, bindings, after_paths = _matched(after_pattern, after_program, target['holes'],
                                           budget, bindings=bindings, paths={})
    if bindings is None:
        return {'status': 'not_matched', 'stage': 'shared_after', 'reason': check['reason']}
    if set(bindings) != {hole['name'] for hole in target['holes']}:
        raise _MissingContract('complete_joint_hole_binding_required')
    components, refs = {}, {}
    for hole in target['holes']:
        _spend(budget)
        name = hole['name']
        if name in before_paths:
            side, owner, prepared, path = 'before', parent, before_prepared, before_paths[name]
        else:
            side, owner, prepared, path = 'after', entry, after_prepared, after_paths[name]
        actual = prepared['contracts'][tuple(path)]
        if actual.get('output_unit') is None:
            raise _MissingContract('actual_bound_subtree_unit_unknown')
        if actual.get('value_type') != hole.get('value_type'):
            raise _MissingContract('actual_bound_subtree_type_mismatch')
        if name in before_paths and name in after_paths:
            other = after_prepared['contracts'][tuple(after_paths[name])]
            if other.get('output_unit') is None:
                raise _MissingContract('actual_shared_after_subtree_unit_unknown')
            if any(actual.get(key) != other.get(key) for key in
                   ('value_type', 'output_unit', 'time_contract')):
                raise _MissingContract('actual_shared_subtree_contract_conflict')
        # _bound_proposal validates units/time/roles using these actual prepared
        # contracts, and retains shared binding equality from _match_parent.
        components[name] = {'id': f'{owner["id"]}:{side}:{path}',
            'version': owner.get('version'), 'source_ref': _reference(owner),
            'program': bindings[name].to_dict(), '_program': bindings[name],
            '_times': actual['time_contract'], 'output_unit': actual['output_unit'],
            'role_signature': deepcopy(owner['role_signature']),
            'context_ref': deepcopy(owner['context_ref'])}
        refs[name] = {'side': side, 'program_ref': _reference(owner), 'path': path,
            'expanded_nodes': bindings[name].size, 'output_unit': actual['output_unit'],
            'time_contract': deepcopy(actual['time_contract'])}
    hypotheses = target.get('constraint_hypotheses') or [{
        'id': 'learned_constraints', 'constraints': [], 'assignments': []}]
    checks = []
    for index, hypothesis in enumerate(hypotheses):
        _spend(budget)
        if hypothesis.get('assignments'):
            raise _MissingContract('instance_property_assignment_not_supported')
        variant = {**target, 'learned_constraints': [*target.get('learned_constraints', []),
                                                   *hypothesis['constraints']]}
        proposal, constraint_checks = _bound_proposal(variant, components, budget)
        checks.append({'hypothesis_index': index, 'checks': constraint_checks})
        if proposal is not None and Program.from_dict(proposal['program']) == after_program:
            return {'status': 'matched', 'before_exact': True, 'after_exact': True,
                'shared_bindings': True, 'scope': scope, 'binding_refs': refs,
                'constraint_checks': checks, 'hypothesis_index': index,
                'claim': 'actual_saved_instance_not_generic_family_equivalence',
                'scientific_status_changed': False}
    return {'status': 'not_matched', 'stage': 'constraints',
            'constraint_checks': checks, 'reason': 'actual_instance_constraint_violated'}


def classify_saved_proposals(parent, entries, targets, allocation=None):
    """Find one complete actual paired witness per frozen target, in order.

    ``parent``/``entries`` are real resolved theory records.  Each entry keeps
    its original consolidation_ref.parent. ``targets`` contains dictionaries
    with candidate/reference, and is never sent to a generator.  Optional
    parent.historical_before is another actual resolved snapshot; it is checked
    separately from the declared generation parent, never substituted for it.
    """
    allocation = {'compute': 5000} if allocation is None else deepcopy(allocation)
    budget = SearchBudget(allocation['compute'])
    records, results, exhausted = [], [], False
    prepared_entries = {}
    before_program, before_prepared = None, None
    for target_index, target_entry in enumerate(targets):
        target = target_entry['candidate']
        result = {'target_ref': deepcopy(target_entry['reference']), 'status': 'UNKNOWN',
                  'attempted_count': 0, 'witness_classification_index': None}
        results.append(result)
        for entry_index, entry in enumerate(entries):
            record = {'target_index': target_index, 'target_ref': deepcopy(target_entry['reference']),
                'entry_index': entry_index, **_entry_summary(entry)}
            records.append(record)
            result['attempted_count'] += 1
            started = budget.spent
            try:
                after_program = _program(entry)
                after_pattern = Definition.from_dict(target['definition']).body
                record['after_only_match'], _, _ = _matched(after_pattern, after_program,
                                                            target['holes'], budget)
                record['declared_generation_parent'] = _declared_parent(parent, entry, budget)
                record['historical_before_pair_contract'] = {'status': 'not_requested'}
                if record['after_only_match']['status'] != 'matched':
                    record['generation_parent_pair_contract'] = {
                        'status': 'not_matched', 'stage': 'after_syntax'}
                    continue
                if not record['declared_generation_parent']['exact']:
                    record['generation_parent_pair_contract'] = {
                        'status': 'UNKNOWN', 'reason': 'actual_declared_parent_not_resolved_exactly'}
                    continue
                if before_program is None:
                    before_program, before_prepared = _prepared(parent, budget)
                if entry_index not in prepared_entries:
                    prepared_entries[entry_index] = _prepared(entry, budget)
                after_program, after_prepared = prepared_entries[entry_index]
                record['generation_parent_pair_contract'] = _pair(parent, entry, target,
                    before_program, after_program, before_prepared, after_prepared, budget)
                historical = parent.get('historical_before')
                if historical is not None:
                    try:
                        if (record['declared_generation_parent']['resolved_program_equal_historical_before']
                                and all(parent.get(key) == historical.get(key) for key in
                                        ('role_signature', 'output_unit', 'context_ref', 'context_family_ref',
                                         'contextual_scope', 'contextual_scope_ref'))):
                            record['historical_before_pair_contract'] = {
                                **deepcopy(record['generation_parent_pair_contract']),
                                'reused_exact_same_actual_before_program_and_contract': True}
                        else:
                            hp, prepared = _prepared(historical, budget)
                            record['historical_before_pair_contract'] = _pair(historical, entry, target,
                                hp, after_program, prepared, after_prepared, budget)
                    except _MissingContract as error:
                        record['historical_before_pair_contract'] = {'status': 'UNKNOWN', 'reason': str(error)}
                if record['generation_parent_pair_contract']['status'] == 'matched':
                    result.update(status='actual_paired_witness_found',
                                  witness_classification_index=len(records) - 1)
                    break
            except _MissingContract as error:
                record['generation_parent_pair_contract'] = {'status': 'UNKNOWN', 'reason': str(error)}
            except _Incomplete:
                record['generation_parent_pair_contract'] = {
                    'status': 'UNKNOWN', 'reason': 'posthoc_budget_exhausted'}
                exhausted = True
                break
            finally:
                record['actual_compute_cost'] = budget.spent - started
        result['unstarted_count'] = len(entries) - result['attempted_count']
        if exhausted:
            break
    for target_entry in targets[len(results):]:
        results.append({'target_ref': deepcopy(target_entry['reference']), 'status': 'UNKNOWN',
                        'attempted_count': 0, 'unstarted_count': len(entries),
                        'reason': 'not_started_budget_exhausted'})
    return {'status': 'completed' if all(item['status'] == 'actual_paired_witness_found'
                                       for item in results) else 'UNKNOWN',
        'allocation': allocation, 'actual_compute_cost': budget.spent,
        'budget_exhausted': exhausted, 'last_denied_request': budget.last_denied_request,
        'targets': results, 'classifications': records,
        'attempted_count': len(records), 'unstarted_count': len(entries) * len(targets) - len(records),
        'finite_frontier': 'original target order; original saved entry order; stop each target at first complete actual paired witness',
        'absence_or_generic_equivalence_established': False,
        'target_syntax_used_by_generator': False, 'new_predictions': 0, 'new_fits': 0,
        'new_observations': 0, 'provider_calls': 0, 'scientific_status_changed': False}
