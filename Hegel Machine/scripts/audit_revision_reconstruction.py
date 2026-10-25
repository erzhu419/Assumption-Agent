"""Bounded symbolic reconstruction certificates for saved revision objects.

This is a representation audit, not proposal search or a world evaluator.  A
certificate uses an existing joint constructor, with any compound argument
itself reconstructed by an existing Definition.  Target syntax supplies the
question and its declared domain; it is not new scientific evidence.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_relations import (
    _Incomplete, _MissingContract, _check_generated_units, _instantiate,
    _match_parent, _occurrence_contracts, _prepare, _spend, _substitute,
)
from hegel_machine.cognition.theory_consolidation import _read_library


# These seven current operations use precisely the recursive maximum rule in
# Program.minimum_history_depth.  In particular, an unresolved arg is NEVER
# evaluated using Program.arg's default depth zero.
CURRENT_MAX_OPS = {'add', 'arg', 'const', 'feature', 'ge', 'if', 'mul'}
EVIDENCE_FIELDS = {
    'id', 'version', 'theme', 'source_ids', 'source_contexts', 'source_outcomes',
    'source_bindings', 'source_reconstruction', 'source_evidence_refs',
    'counterexample_refs', 'counterexample_checks', 'trigger_checks', 'encoding_cost',
    'status', 'frozen', 'active_library', 'construction_only', 'transfer_status',
    'support_updated', 'global_support', 'independent_confirmation',
    'scientific_support_added', 'parameter_relation_search',
}
STRUCTURE_FIELDS = {
    'before_pattern', 'definition', 'holes', 'parent_holes', 'learned_constraints',
    'constraint_hypotheses',
}


def child_ref(reference, *parts):
    return {**deepcopy(reference), 'json_path': [*reference.get('json_path', []), *parts]}


def node_ref(reference, side, path):
    parts = ['before_pattern'] if side == 'before' else ['definition', 'body']
    for index in path:
        parts.extend(('children', index))
    return child_ref(reference, *parts)


def source_gate(candidate):
    return any(item.get('before', {}).get('failed') is True
               for item in candidate.get('trigger_checks', []))


def executable_contract(candidate):
    result = {key: deepcopy(value) for key, value in candidate.items()
              if key not in EVIDENCE_FIELDS}
    # The ordinary structural consumer has always accepted the absent purpose;
    # parameter-relation purpose remains inside parameter_relations unchanged.
    result.setdefault('purpose', 'structural_revision')
    result.setdefault('parameter_relations', [])
    result['measured_source_gate'] = source_gate(candidate)
    if isinstance(result.get('definition'), dict):
        result['definition'].pop('definition_id', None)
        result['definition'].pop('source_concept_id', None)
    return result


def _count_metadata(value, budget):
    """Charge actual scalar/container contract visits, not disk-byte reads."""
    _spend(budget)
    if isinstance(value, dict):
        for item in value.values():
            _count_metadata(item, budget)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _count_metadata(item, budget)


def _same_metadata(left, right, budget):
    """Charge the equality traversal actually needed, stopping at a mismatch."""
    _spend(budget)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_metadata(value, right[key], budget) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_metadata(a, b, budget) for a, b in zip(left, right))
    return left == right


def _base_contract(candidate):
    return {key: value for key, value in executable_contract(candidate).items()
            if key not in STRUCTURE_FIELDS}


def _provenance(reference):
    return {name: child_ref(reference, name) for name in (
        'source_bindings', 'source_reconstruction', 'source_outcomes',
        'trigger_checks', 'counterexample_refs')}


def _target_argument(name, candidate, reference):
    index = next(index for index, hole in enumerate(candidate['holes']) if hole['name'] == name)
    return {'kind': 'target_argument', 'name': name,
            'contract_ref': child_ref(reference, 'holes', index)}


def _known_units(candidate):
    return {Program('arg', name=hole['name']): hole.get('output_unit')
            for hole in candidate['holes'] if hole.get('output_unit') is not None}


def _times(program, candidate, budget):
    _spend(budget)
    if program.op not in CURRENT_MAX_OPS:
        raise _MissingContract('unsupported_symbolic_time_operation')
    if program.op == 'arg':
        value = next((hole.get('time_contract') for hole in candidate['holes']
                      if hole['name'] == program.name), None)
        if not value:
            raise _MissingContract('target_argument_time_contract_required')
        return list(value)
    if program.op == 'feature':
        value = candidate.get('fixed_role_signature', {}).get(program.name, {}).get('time')
        if value is None:
            raise _MissingContract('feature_time_contract_required')
        return [value]
    values = []
    for child in program.children:
        for value in _times(child, candidate, budget):
            if value not in values:
                values.append(value)
    return values


def _typed_binding(hole, program, target, budget):
    _spend(budget)
    if hole.get('value_type') != 'scalar':
        raise _MissingContract('unsupported_original_hole_value_type')
    for node in _walk(program):
        _spend(budget)
        if node.op == 'arg':
            contract = next((item for item in target['holes'] if item['name'] == node.name), {})
            if contract.get('value_type') != 'scalar':
                raise _MissingContract('target_argument_scalar_contract_required')
        elif node.op in {'feature', 'lag'}:
            contract = target.get('fixed_role_signature', {}).get(node.name, {})
            if any(contract.get(key) is None for key in ('role', 'unit', 'time')):
                raise _MissingContract('feature_role_unit_time_contract_required')
    roles = target.get('fixed_role_signature', {})
    occurrences, _ = _occurrence_contracts(program, program, target['holes'], budget)
    prepared = _prepare({'role_signature': roles, 'output_unit': hole.get('output_unit'),
        'context_ref': target.get('scope'), 'subtree_contracts': occurrences}, program, budget)
    known = _known_units(target)

    def literal_units(node, path=()):
        _spend(budget)
        if node.op == 'const':
            value = prepared['contracts'][path]['output_unit']
            if value is not None:
                if node in known and known[node] != value:
                    raise _MissingContract('literal_occurrences_require_distinct_units')
                known[node] = value
        for index, child in enumerate(node.children):
            literal_units(child, (*path, index))

    literal_units(program)
    # Constants receive only the existing typed compiler's propagated context.
    # Compound expressions remain transparent to the independent recursive
    # checker, which still rejects feature/argument/product/branch conflicts.
    unit = _check_generated_units(program, hole.get('output_unit'), roles, known, budget)
    if unit is None or unit != hole.get('output_unit'):
        raise _MissingContract('symbolic_binding_unit_mismatch_or_unknown')
    clocks = _times(program, target, budget)
    # A literal constant is timeless, as in the existing anti-unifier.  An arg
    # with missing clock never takes this branch: _times requires its contract.
    if clocks and clocks != hole.get('time_contract'):
        raise _MissingContract('symbolic_binding_time_mismatch')
    return {'status': 'passed', 'output_unit': unit, 'time_contract': clocks,
            'literal_timeless': not clocks}


def _walk(program):
    yield program
    for child in program.children:
        yield from _walk(child)


def _depth_reference(reference):
    if reference.get('property') != 'minimum_history_depth':
        raise _MissingContract('unsupported_symbolic_property')
    if reference.get('object') == 'program':
        return 'program'
    if reference.get('object') == 'hole':
        return reference['name']
    raise _MissingContract('unsupported_symbolic_property_object')


def _declared_depths(constraints, budget):
    """Read the saved fixed-program/equal-hole domain, not observed examples."""
    values, equalities = {}, []
    for constraint in constraints:
        _spend(budget)
        left = _depth_reference(constraint['left'])
        if constraint['op'] == 'fixed':
            value = constraint['value']
            if type(value) is not int or value < 0 or (left in values and values[left] != value):
                raise _MissingContract('inconsistent_or_unknown_declared_depth')
            values[left] = value
        elif constraint['op'] == 'equal':
            equalities.append((left, _depth_reference(constraint['right'])))
        else:
            raise _MissingContract('unsupported_symbolic_constraint')
    # This intentionally supports only the actually learned star of equalities
    # to the program, not a new general implication/constraint solver.
    for left, right in equalities:
        _spend(budget)
        if right != 'program' or right not in values:
            raise _MissingContract('unsupported_depth_equality_domain')
        if left in values and values[left] != values[right]:
            raise _MissingContract('inconsistent_declared_depth')
        values[left] = values[right]
    return values


def _symbolic_depth(program, domain, budget):
    _spend(budget)
    if program.op == 'arg':
        if program.name not in domain:
            raise _MissingContract('unconstrained_argument_depth')
        return domain[program.name]
    if program.op not in CURRENT_MAX_OPS:
        raise _MissingContract('unsupported_symbolic_depth_operation')
    if program.op == 'if' and program.children[0].op == 'arg':
        # A legal scalar hole may be an available guard.  That changes the
        # native if depth rule after substitution, so max is not a certificate.
        raise _MissingContract('symbolic_if_condition_history_rule_unknown')
    return max((_symbolic_depth(child, domain, budget) for child in program.children), default=0)


def _depth_dependencies(program, budget):
    """For these current max-rule nodes, max(depths)==0 iff all inputs are0."""
    _spend(budget)
    if program.op == 'arg':
        return {program.name}
    if program.op not in CURRENT_MAX_OPS:
        raise _MissingContract('unsupported_symbolic_depth_operation')
    if program.op == 'if' and program.children[0].op == 'arg':
        raise _MissingContract('symbolic_if_condition_history_rule_unknown')
    return set().union(*(_depth_dependencies(child, budget) for child in program.children))


def _hypotheses(candidate):
    result = candidate.get('constraint_hypotheses')
    return result if result else [{'id': 'learned_constraints', 'constraints': [], 'assignments': []}]


def _prove_constraints(original, target, bindings, after, budget, *, require_reverse=True):
    proofs = []
    for target_index, target_hypothesis in enumerate(_hypotheses(target)):
        _spend(budget)
        if target_hypothesis.get('assignments'):
            raise _MissingContract('symbolic_property_assignments_not_supported')
        domain = _declared_depths([*target.get('learned_constraints', []),
                                  *target_hypothesis['constraints']], budget)
        target_depth = _symbolic_depth(after, domain, budget)
        if domain.get('program') != target_depth:
            raise _MissingContract('declared_program_depth_not_established')
        selected = None
        for old_index, hypothesis in enumerate(_hypotheses(original)):
            _spend(budget)
            if hypothesis.get('assignments'):
                continue
            residual = lambda value: {key: item for key, item in value.items()
                                      if key not in {'id', 'constraints', 'assignments'}}
            if not _same_metadata(residual(hypothesis), residual(target_hypothesis), budget):
                continue
            checks = []
            for constraint in [*original.get('learned_constraints', []), *hypothesis['constraints']]:
                _spend(budget)

                def value(reference):
                    name = _depth_reference(reference)
                    return target_depth if name == 'program' else _symbolic_depth(bindings[name], domain, budget)

                left = value(constraint['left'])
                if constraint['op'] == 'fixed':
                    right = constraint['value']
                elif constraint['op'] == 'equal':
                    right = value(constraint['right'])
                else:
                    raise _MissingContract('unsupported_symbolic_constraint')
                checks.append(left == right)
            if all(checks):
                selected = {'target_hypothesis_index': target_index,
                    'original_hypothesis_index': old_index, 'constraint_count': len(checks),
                    'target_declared_program_depth': target_depth,
                    'argument_depths_come_from_target_constraints': True,
                    'unbound_arg_default_depth_used': False,
                    'target_domain_implies_composed_old_domain': True,
                    'status': 'entailed_for_nested_constructor'}
                if not require_reverse:
                    break
                old_domain = _declared_depths([*original.get('learned_constraints', []),
                                               *hypothesis['constraints']], budget)
                # The reverse implication concerns this wired composition,
                # not the entire more general old constructor family.
                zero_inputs = set()
                for name, depth in old_domain.items():
                    if depth != 0:
                        raise _MissingContract('reverse_depth_domain_outside_zero_max_fragment')
                    zero_inputs.update(_depth_dependencies(
                        after if name == 'program' else bindings[name], budget))
                new_names = {hole['name'] for hole in target['holes']}
                if (set(domain) != {'program', *new_names}
                        or any(value != 0 for value in domain.values())
                        or zero_inputs != new_names):
                    raise _MissingContract('reverse_target_depth_domain_not_established')
                selected = {'target_hypothesis_index': target_index,
                    'original_hypothesis_index': old_index, 'constraint_count': len(checks),
                    'target_declared_program_depth': target_depth,
                    'argument_depths_come_from_target_constraints': True,
                    'unbound_arg_default_depth_used': False,
                    'target_domain_implies_composed_old_domain': True,
                    'composed_old_domain_implies_target_domain': True,
                    'nonnegative_depth_max_zero_dependencies': sorted(zero_inputs),
                    'status': 'equivalent_composed_domains'}
                break
        if selected is None:
            raise _MissingContract('old_constraint_domain_not_entailed')
        proofs.append(selected)
    return proofs


def _construct_argument(program, target, target_ref, constructors, budget, *, ancestry=()):
    """Give a replayable old-call DAG; never import a compound target as a seed."""
    _spend(budget)
    if program.op == 'arg':
        return _target_argument(program.name, target, target_ref)
    # Current objects' compound arguments are smaller instances of an old
    # Definition.  Primitive leaves must already occur in an old constructor.
    if not program.children:
        for entry in constructors:
            body = Definition.from_dict(entry['candidate']['definition']).body
            for node in _walk(body):
                _spend(budget)
                if node == program and node.op != 'arg':
                    return {'kind': 'existing_constructor_leaf', 'constructor_ref': entry['reference'],
                            'op': program.op, 'value': program.value, 'name': program.name, 'lag': program.lag}
        raise _MissingContract('non_argument_leaf_not_in_old_constructor')
    for entry in constructors:
        candidate = entry['candidate']
        body = Definition.from_dict(candidate['definition']).body
        if body.size > program.size or (entry['reference'], program) in ancestry:
            continue
        try:
            bindings, _ = _match_parent(body, program, candidate['holes'], budget)
        except _MissingContract:
            continue
        used = set(bindings)
        # A Definition lists all joint holes, even before-only holes absent from
        # its after body.  Bind those unused arguments only to an already
        # declared target arg with the identical scalar/unit/time contract.
        for hole in candidate['holes']:
            if hole['name'] in bindings:
                continue
            matching = None
            for actual in target['holes']:
                _spend(budget)
                if all(actual.get(key) == hole.get(key) for key in ('value_type', 'output_unit', 'time_contract')):
                    matching = Program('arg', name=actual['name'])
                    break
            if matching is None:
                raise _MissingContract('unused_definition_argument_contract_unavailable')
            bindings[hole['name']] = matching
        typed = {hole['name']: _typed_binding(hole, bindings[hole['name']], target, budget)
                 for hole in candidate['holes']}
        restored = _instantiate(candidate, bindings, budget)
        if restored != program:
            raise _MissingContract('nested_definition_exact_expansion_failed')
        constraint_proof = _prove_constraints(candidate, target, bindings, restored,
                                               budget, require_reverse=False)
        arguments = {name: _construct_argument(value, target, target_ref, constructors, budget,
                        ancestry=(*ancestry, (entry['reference'], program)))
                     for name, value in bindings.items()}
        return {'kind': 'old_definition_call', 'constructor_ref': entry['reference'],
            'definition_ref': child_ref(entry['reference'], 'definition'),
            'arguments': arguments, 'unused_arguments': sorted(set(bindings) - used),
            'argument_contracts': {hole['name']: {key: deepcopy(hole[key]) for key in
                ('value_type', 'output_unit', 'time_contract')} for hole in candidate['holes']},
            'typed_bindings': typed, 'old_constraint_domain_entailed': constraint_proof,
            'exact_ordered_expansion': True}
    raise _MissingContract('compound_argument_old_constructor_certificate_unavailable')


def _reverse_typed_domain(dag, original, target, budget):
    required = {}

    def visit(node, contract):
        _spend(budget)
        if node['kind'] == 'target_argument':
            required.setdefault(node['name'], []).append(contract)
        elif node['kind'] == 'old_definition_call':
            for name, child in node['arguments'].items():
                visit(child, node['argument_contracts'][name])

    for hole in original['holes']:
        visit(dag[hole['name']], {key: deepcopy(hole[key]) for key in
                                ('value_type', 'output_unit', 'time_contract')})
    for hole in target['holes']:
        _spend(budget)
        expected = {key: hole.get(key) for key in ('value_type', 'output_unit', 'time_contract')}
        if hole['name'] not in required or any(value != expected for value in required[hole['name']]):
            raise _MissingContract('reverse_target_typed_domain_not_established')
    return {'status': 'equivalent_composed_typed_domain',
            'all_target_arguments_constrained_by_original_call_arguments': True,
            'covered_argument_names': [hole['name'] for hole in target['holes']],
            'unit_aliases_assumed': False}


def certify_reconstruction(original, target, constructors, budget, *, original_ref, target_ref):
    """Prove this saved target family is an instance of an existing constructor."""
    old_base, new_base = _base_contract(original), _base_contract(target)
    _count_metadata(old_base, budget)
    _count_metadata(new_base, budget)
    if old_base != new_base:
        raise _MissingContract('executable_scope_role_or_gate_contract_differs')
    if not source_gate(original):
        raise _MissingContract('measured_source_failure_gate_not_satisfied')
    before = Program.from_dict(target['before_pattern'])
    after = Definition.from_dict(target['definition']).body
    old_before = Program.from_dict(original['before_pattern'])
    old_after = Definition.from_dict(original['definition']).body
    for value, pattern in ((original, old_before), (target, before)):
        arguments = {node.name for node in _walk(pattern) if node.op == 'arg'}
        _spend(budget, pattern.size)
        if arguments != set(value['parent_holes']):
            raise _MissingContract('parent_holes_differ_from_actual_before_arguments')
        if set(Definition.from_dict(value['definition']).arguments) != {
                hole['name'] for hole in value['holes']}:
            raise _MissingContract('definition_arguments_differ_from_declared_holes')
    bindings, _ = _match_parent(old_before, before, original['holes'], budget)
    _match_parent(old_after, after, original['holes'], budget, bindings=bindings)
    if set(bindings) != {hole['name'] for hole in original['holes']}:
        raise _MissingContract('original_joint_hole_not_bound')
    # _substitute is a pure tree operation; explicitly charge its node visits.
    _spend(budget, old_before.size)
    if _substitute(old_before, bindings) != before or _instantiate(original, bindings, budget) != after:
        raise _MissingContract('ordered_joint_reconstruction_failed')
    typed = {hole['name']: _typed_binding(hole, bindings[hole['name']], target, budget)
             for hole in original['holes']}
    constraints = _prove_constraints(original, target, bindings, after, budget)
    dag = {name: _construct_argument(value, target, target_ref, constructors, budget)
           for name, value in bindings.items()}
    typed_domain = _reverse_typed_domain(dag, original, target, budget)
    return {'status': 'certified_old_constructor_instance', 'original_ref': original_ref,
        'target_ref': target_ref, 'before_exact': True, 'after_exact': True,
        'shared_symbolic_bindings': True, 'typed_bindings': typed,
        'old_constraint_domain_entailed': constraints, 'typed_domain_equivalence': typed_domain,
        'composition_dag': dag,
        'source_failure_gate': True, 'original_evidence_refs': _provenance(original_ref),
        'target_evidence_refs': _provenance(target_ref),
        'classification': 'derived_reconstructible_with_old_closure',
        'whole_original_family_equivalent_to_target': False,
        'objects_or_classes_merged': False,
        'world_equivalence_or_support_added': False, 'finite_old_proposal_generation_established': False}


def audit_reconstruction(available_objects, targets, allocation, *, old_constructors=None,
                         material_refs=None):
    """Independent per-arm bounded audit; entries contain candidate/reference."""
    budget = SearchBudget(allocation['compute'])
    constructors = available_objects if old_constructors is None else old_constructors
    outputs, exhausted = [], False
    started = time.monotonic()
    for target_entry in targets:
        target_started = budget.spent
        target, reference = target_entry['candidate'], target_entry['reference']
        result = {'target_ref': reference, 'status': 'UNKNOWN', 'attempts': []}
        outputs.append(result)
        try:
            target_contract = executable_contract(target)
            for entry in available_objects:
                before = budget.spent
                actual_contract = executable_contract(entry['candidate'])
                exact = _same_metadata(actual_contract, target_contract, budget)
                result['attempts'].append({'method': 'exact_available_contract',
                    'candidate_ref': entry['reference'], 'exact': exact,
                    'actual_compute': budget.spent - before})
                if exact:
                    result.update(status='certified_available_contract', certificate={
                        'status': 'exact_available_contract', 'original_ref': entry['reference'],
                        'target_ref': reference, 'source_failure_gate': source_gate(target),
                        'target_evidence_refs': _provenance(reference),
                        'world_equivalence_or_support_added': False})
                    break
            if result['status'] != 'UNKNOWN':
                continue
            for entry in constructors:
                before = budget.spent
                try:
                    certificate = certify_reconstruction(entry['candidate'], target, constructors,
                        budget, original_ref=entry['reference'], target_ref=reference)
                except _MissingContract as error:
                    result['attempts'].append({'method': 'old_joint_symbolic_constructor',
                        'candidate_ref': entry['reference'], 'reason': str(error),
                        'status': 'UNKNOWN_not_a_certificate', 'actual_compute': budget.spent - before})
                    continue
                result['attempts'].append({'method': 'old_joint_symbolic_constructor',
                    'candidate_ref': entry['reference'], 'status': 'certified',
                    'actual_compute': budget.spent - before})
                result.update(status='certified_old_constructor_instance', certificate=certificate)
                break
        except _Incomplete:
            result['reason'] = 'bounded_reconstruction_budget_exhausted'
            exhausted = True
            break
        finally:
            result['actual_compute'] = budget.spent - target_started
    for target_entry in targets[len(outputs):]:
        outputs.append({'target_ref': target_entry['reference'], 'status': 'UNKNOWN',
                        'reason': 'not_started_budget_exhausted', 'attempts': []})
    completed = all(item['status'].startswith('certified') for item in outputs)
    return {'status': 'completed' if completed else 'UNKNOWN', 'allocation': deepcopy(allocation),
        'actual_new_compute': budget.spent, 'budget_exhausted': exhausted,
        'last_denied_request': budget.last_denied_request,
        'preparation_wall_seconds': time.monotonic() - started, 'targets': outputs,
        'material_refs': deepcopy(material_refs or {}), 'all_old_materials_retained': True,
        'frontier': 'available exact contracts in order, then original joint constructors in order; first complete certificate stops each target',
        'incomplete_frontier_is_non_reconstructibility': False,
        'representation_given_target_syntax': True, 'new_predictions': 0, 'new_fits': 0,
        'new_observations': 0, 'provider_calls': 0, 'successor_executed': False}


def _read_reference(reference):
    value = json.loads(Path(reference['file']).read_text())
    for part in reference.get('json_path', []):
        value = value[part]
    return value


def run_registered(registration_path, condition, output):
    registration_path, output = Path(registration_path), Path(output)
    if output.exists():
        raise ValueError('Preserve the previous receipt; output must be new')
    registration = json.loads(registration_path.read_text())
    comparison = registration['reconstruction_comparison']
    frontier = comparison['concrete_frontier']
    library_ref = frontier['old_constructor_library_ref']
    library = _read_library(library_ref, registration_path.parent)
    old = [{'candidate': library['candidates'][key],
            'reference': child_ref(library_ref, 'candidates', key)}
           for key in frontier['old_constructor_order']]
    targets = [{'candidate': _read_reference(item['representative_ref']),
                'reference': item['representative_ref']} for item in comparison['frozen_groups']]
    choices = {item['name']: item for item in comparison['conditions']}
    selected = choices[condition]
    available = old + (targets if condition == 'old_plus_two_frozen_groups' else [])
    receipt = audit_reconstruction(available, targets, {'compute': selected['budget_compute']},
        old_constructors=old, material_refs={
            'registration_ref': {'file': str(registration_path.resolve()), 'json_path': []},
            'old_constructor_library_ref': library_ref,
            'old999_archive_ref': comparison['old999_archive_ref'],
            'old_closure_ref': comparison['old_closure_ref'],
            'old_full_D2_preservation_ref': comparison['old_full_D2_preservation_ref']})
    receipt.update(condition=condition, available_object_count=len(available),
        old_constructor_count=len(old), direct_target_availability_is_positive_control=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--condition', required=True,
                        choices=('complete_old_capabilities', 'old_plus_two_frozen_groups'))
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    result = run_registered(arguments.registration, arguments.condition, arguments.output)
    print(json.dumps({key: result[key] for key in
                      ('condition', 'status', 'actual_new_compute', 'budget_exhausted')}))
