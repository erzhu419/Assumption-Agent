"""Bounded correspondence of saved theories, without merging their identities.

Program comparison keeps the native ordered tree, branch and history semantics.
Its result proposes reuse/refitting/rebinding; it never issues a confirmation.
"""
from copy import deepcopy
from math import isfinite

from .abstractions import AbstractTemplate
from .language import Program
from .scale_bindings import bind_measurement_arguments


class _BudgetExhausted(Exception):
    pass


class _Different(Exception):
    pass


class _Unresolved(Exception):
    pass


class _Budget:
    def __init__(self, available):
        self.remaining = max(0, int(available))
        self.nodes = self.behavior = 0

    def take(self, *, behavior=False):
        if not self.remaining:
            raise _BudgetExhausted
        self.remaining -= 1
        if behavior:
            self.behavior += 1
        else:
            self.nodes += 1


def _program(record):
    value = record.get('program')
    if isinstance(value, AbstractTemplate):
        return value.pattern
    if isinstance(value, Program):
        return value
    return Program.from_dict(value) if isinstance(value, dict) else None


def _parameter_paths(record):
    value = record.get('program')
    paths = record.get('parameter_paths', value.parameter_paths if isinstance(value, AbstractTemplate) else [])
    return {tuple(path) for path in paths}


def _compare_trees(source, target, budget, source_paths, target_paths, *, max_expansion_depth):
    features, reverse, parameters = {}, {}, []

    def visit(left, right, lpath=(), rpath=(), lenv=None, renv=None, ldepth=0, rdepth=0):
        budget.take()
        lenv, renv = lenv or {}, renv or {}
        if left.op == 'arg' and left.name in lenv:
            node, path, environment, depth = lenv[left.name]
            return visit(node, right, path, rpath, environment, renv, depth, rdepth)
        if right.op == 'arg' and right.name in renv:
            node, path, environment, depth = renv[right.name]
            return visit(left, node, lpath, path, lenv, environment, ldepth, depth)
        if left.op == 'call' or right.op == 'call':
            if left.op == 'call':
                if ldepth >= max_expansion_depth:
                    raise _Unresolved('definition_expansion_limit')
                lenv = {name: (child, None if lpath is None else (*lpath, i), lenv, ldepth)
                        for i, (name, child) in enumerate(zip(left.definition.arguments, left.children))}
                left, lpath, ldepth = left.definition.body, None, ldepth + 1
            if right.op == 'call':
                if rdepth >= max_expansion_depth:
                    raise _Unresolved('definition_expansion_limit')
                renv = {name: (child, None if rpath is None else (*rpath, i), renv, rdepth)
                        for i, (name, child) in enumerate(zip(right.definition.arguments, right.children))}
                right, rpath, rdepth = right.definition.body, None, rdepth + 1
            return visit(left, right, lpath, rpath, lenv, renv, ldepth, rdepth)
        if left.op != right.op or len(left.children) != len(right.children):
            raise _Different('ordered_operator_mismatch')
        if left.lag != right.lag:
            raise _Different('history_or_window_mismatch')
        if left.op in {'feature', 'lag'}:
            old = features.get(left.name)
            if old is not None and old != right.name:
                raise _Different('inconsistent_variable_mapping')
            if right.name in reverse and reverse[right.name] != left.name:
                raise _Different('non_injective_variable_mapping')
            features[left.name], reverse[right.name] = right.name, left.name
        elif left.name != right.name:
            raise _Different('ordered_operator_attribute_mismatch')
        if left.value != right.value:
            if (left.op != 'const' or lpath not in source_paths
                    or (target_paths and rpath not in target_paths)):
                raise _Different('undeclared_constant_mismatch')
            parameters.append({'source_path': list(lpath), 'target_path': list(rpath) if rpath is not None else None,
                               'source_value': left.value, 'target_value': right.value,
                               'constraint': 'requires_refit'})
        for i, (lchild, rchild) in enumerate(zip(left.children, right.children)):
            visit(lchild, rchild, None if lpath is None else (*lpath, i),
                  None if rpath is None else (*rpath, i), lenv, renv, ldepth, rdepth)

    visit(source, target)
    return features, parameters


def _conversion(value, source_unit, target_unit):
    return (isinstance(value, dict) and value.get('from') == source_unit
            and value.get('to') == target_unit
            and isinstance(value.get('scale'), (int, float)) and isfinite(value['scale'])
            and value['scale'] != 0
            and isinstance(value.get('offset'), (int, float)) and isfinite(value['offset']))


def _compile_measurement_binding(program, mapping, declaration):
    """Read target-unit inputs in source units, then convert the source output.

    A conversion declares target = scale * source + offset. Substitution uses
    the existing measurement binder, including its historical/reset semantics.
    """
    arguments = {}
    conversions = declaration.get('unit_conversions', {})
    output = declaration.get('output_conversion')
    if output and any(name == '__target__' for name, _ in program.observed_variables()):
        # The native language's historical response has the output's unit.
        mapping = {**mapping, '__target__': '__target__'}
        conversions = {**conversions, '__target__': output}
    for source, target in mapping.items():
        value = Program('feature', name=target)
        conversion = conversions.get(source)
        if conversion:
            if conversion['offset'] != 0:
                value = Program('sub', children=(value, Program('const', value=conversion['offset'])))
            if conversion['scale'] != 1:
                value = Program('mul', children=(Program('const', value=1 / conversion['scale']), value))
        arguments[source] = value
    bound = bind_measurement_arguments(program.expanded(), arguments)
    if output:
        if output['scale'] != 1:
            bound = Program('mul', children=(Program('const', value=output['scale']), bound))
        if output['offset'] != 0:
            bound = Program('add', children=(bound, Program('const', value=output['offset'])))
    return bound


def _mapping_options(source, target, features, allowed):
    source_roles, target_roles = source.get('role_signature', {}), target.get('role_signature', {})
    for left, right in features.items():
        if not all(key in source_roles.get(left, {}) and key in target_roles.get(right, {})
                   for key in ('role', 'unit', 'time')):
            raise _Unresolved('missing_role_unit_or_time_contract')
        if source_roles[left]['time'] != target_roles[right]['time']:
            raise _Different('time_role_mismatch')
    if 'output_unit' not in source or 'output_unit' not in target:
        raise _Unresolved('missing_output_unit_contract')
    options = []
    for declaration in [{}, *allowed]:
        valid = True
        needs_binding = source.get('context_ref') != target.get('context_ref')
        for left, right in features.items():
            lrole, rrole = source_roles[left], target_roles[right]
            if declaration.get('feature_map', {}).get(left, right) != right:
                valid = False
            if lrole['role'] != rrole['role']:
                if declaration.get('role_map', {}).get(lrole['role']) != rrole['role']:
                    valid = False
                needs_binding = True
            conversion = declaration.get('unit_conversions', {}).get(left)
            if lrole['unit'] != rrole['unit'] or conversion is not None:
                if not _conversion(conversion, lrole['unit'], rrole['unit']):
                    valid = False
                needs_binding = needs_binding or lrole['unit'] != rrole['unit'] or bool(
                    conversion and (conversion.get('scale') != 1 or conversion.get('offset') != 0))
            if left != right:
                needs_binding = True
        output = declaration.get('output_conversion')
        if source['output_unit'] != target['output_unit'] or output is not None:
            if not _conversion(output, source['output_unit'], target['output_unit']):
                valid = False
            needs_binding = needs_binding or source['output_unit'] != target['output_unit'] or bool(
                output and (output.get('scale') != 1 or output.get('offset') != 0))
        if not valid:
            continue
        mapping = {**deepcopy(declaration), 'feature_map': deepcopy(features)}
        tested = (declaration.get('status') == 'tested'
                  and bool(declaration.get('evidence') or declaration.get('test_evidence'))
                  and declaration.get('context_ref') == target.get('context_ref'))
        option = {'mapping': mapping, 'needs_binding': needs_binding, 'tested': tested}
        if option not in options:
            options.append(option)
    if not options:
        raise _Different('no_allowed_role_or_unit_mapping')
    return options


def _premises(source, target):
    required = target.get('required_premises', target.get('premise_refs', []))
    current = target.get('premise_statuses', {})
    missing = []
    references = deepcopy(source.get('premise_refs', []))
    references.extend(item for item in required if item not in references)
    for reference in references:
        name = reference.get('id', reference.get('question_id')) if isinstance(reference, dict) else reference
        wanted = next((item for item in required if (
            item.get('id', item.get('question_id')) if isinstance(item, dict) else item) == name), None)
        status = current.get(name, wanted if isinstance(wanted, dict) else {})
        if (wanted is None or status.get('status') not in {'supported', 'empirical_support', 'supported_in_scope'}
                or (isinstance(reference, dict) and any(
                    key in reference and status.get(key) != reference[key]
                    for key in ('version', 'meaning_version', 'answer_version')))):
            missing.append(deepcopy(reference))
    return missing


def _applicability(source, target, options):
    support = source.get('support', {})
    if source.get('context_ref') is None or target.get('context_ref') is None:
        return 'UNKNOWN_MATCH', 'missing_context_contract'
    if (support.get('status') == 'contradicted_in_scope' and support.get('evidence')
            and support.get('context_ref') == target.get('context_ref')):
        return 'PROPOSE_REVISION', 'executed_mismatch_in_requested_context'
    if any(option['needs_binding'] for option in options) and not any(option['tested'] for option in options):
        return 'REBIND_THEN_TEST', 'mapping_or_measurement_not_tested'
    if (support.get('status') not in {'empirical_support', 'supported_in_scope'}
            or not support.get('evidence')):
        return 'UNKNOWN_MATCH', 'missing_executed_source_support'
    if support.get('context_ref') != target.get('context_ref'):
        return 'REBIND_THEN_TEST', 'support_from_different_context'
    if _premises(source, target):
        return 'UNKNOWN_MATCH', 'missing_current_premise_support'
    measured = source.get('measured_coverage_ref')
    if measured is None:
        return 'UNKNOWN_MATCH', 'missing_measured_coverage'
    requested = target.get('declared_scope_ref', target.get('measured_coverage_ref'))
    if requested != measured:
        return 'SCOPED_REUSE', 'reuse_limited_to_measured_coverage'
    if source.get('binding_ref') != target.get('binding_ref'):
        return 'REBIND_THEN_TEST', 'different_actual_binding'
    if any(option['needs_binding'] for option in options):
        return 'SCOPED_REUSE', 'tested_correspondence_in_measured_scope'
    return 'REUSE_INSTANCE', 'supported_instance_and_binding_in_requested_scope'


def _behavior(source, target, budget):
    left, right = source.get('behavior_signature'), target.get('behavior_signature')
    if not left or not right:
        return None
    if (source.get('output_unit') is None or source['output_unit'] != target.get('output_unit')
            or source.get('context_ref') is None or source['context_ref'] != target.get('context_ref')):
        return None
    if any(left.get(key) != right.get(key) or left.get(key) is None
           for key in ('panel_ref', 'readout_ref', 'scales')):
        return None
    lvalues, rvalues, scales = left.get('values', []), right.get('values', []), left['scales']
    if not lvalues or len(lvalues) != len(rvalues) or len(scales) != len(lvalues):
        return None
    differences = []
    for lvalue, rvalue, scale in zip(lvalues, rvalues, scales):
        budget.take(behavior=True)
        if not all(isinstance(x, (int, float)) and isfinite(x) for x in (lvalue, rvalue, scale)) or scale <= 0:
            return None
        differences.append(abs(lvalue - rvalue) / scale)
    tolerance = target.get('behavior_tolerance', 0.)
    return {'panel_ref': left['panel_ref'], 'readout_ref': left['readout_ref'],
            'scales': deepcopy(scales), 'normalized_differences': differences,
            'tolerance': tolerance, 'within_tolerance': max(differences) <= tolerance}


def typed_program_match(source, target, *, compute_budget=10000, allowed_mappings=None,
                        max_expansion_depth=4):
    """Compare one pair under declared roles, parameters, support and scope."""
    budget = _Budget(compute_budget)
    result = {'source_id': source.get('id'), 'target_id': target.get('id'),
              'family_id': source.get('family_id'), 'confirmation': 'not_established',
              'mapping': {}, 'mappings': [], 'parameter_changes': [], 'unverified_premises': [],
              'coverage': {'declared': deepcopy(source.get('declared_scope_ref')),
                           'measured': deepcopy(source.get('measured_coverage_ref')),
                           'requested': deepcopy(target.get('declared_scope_ref'))},
              'evidence': {'source_support': deepcopy(source.get('support', {}).get('evidence', [])),
                           'mapping_fit': [], 'mapping_test': []}}

    def finish(status, reason, **extra):
        return {**result, **extra, 'status': status, 'reason': reason,
                'node_comparisons': budget.nodes, 'behavior_comparisons': budget.behavior,
                'compute_cost': budget.nodes + budget.behavior, 'remaining_budget': budget.remaining}

    try:
        if not budget.remaining:
            raise _BudgetExhausted
        if source.get('source_status') == 'unavailable_retained_version':
            return finish('UNKNOWN_MATCH', 'retained_version_unavailable')
        if source.get('compatible') is False:
            return finish('PROPOSE_REVISION', 'recorded_candidate_incompatible_in_current_scope')
        left, right = _program(source), _program(target)
        if left is None or right is None:
            fields = ('python_source', 'fit_state', 'binding')
            if (not all(key in source and key in target for key in fields)
                    or not isinstance(source['python_source'], str) or not source['python_source']):
                return finish('UNKNOWN_MATCH', 'opaque_python_requires_saved_source_fit_and_binding', relation='opaque_python')
            for key in fields:
                budget.take()
                if source[key] != target[key]:
                    return finish('UNKNOWN_MATCH', 'opaque_python_has_no_exact_instance_match', relation='opaque_python')
            if (source.get('role_signature') != target.get('role_signature')
                    or source.get('output_unit') != target.get('output_unit')):
                return finish('REBIND_THEN_TEST', 'opaque_python_measurement_contract_changed', relation='opaque_python')
            if source.get('context_ref') != target.get('context_ref'):
                return finish('REBIND_THEN_TEST', 'opaque_python_context_changed', relation='opaque_python')
            status, reason = _applicability(source, target, [{'needs_binding': False, 'tested': False}])
            return finish(status, reason, relation='exact_saved_python_instance', level='D')
        try:
            features, parameters = _compare_trees(left, right, budget, _parameter_paths(source),
                _parameter_paths(target), max_expansion_depth=max_expansion_depth)
            result['parameter_changes'] = parameters
            options = _mapping_options(source, target, features, allowed_mappings or [])
        except _Different as mismatch:
            behavior = _behavior(source, target, budget)
            if behavior is not None:
                if not behavior['within_tolerance']:
                    return finish('COMPARE_ALTERNATIVES', 'different_predictions_on_fixed_panel',
                                  relation='finite_behavior_difference', level='C', behavior=behavior)
                status, reason = _applicability(source, target, [{'needs_binding': False, 'tested': False}])
                if status == 'REUSE_INSTANCE':
                    status, reason = 'SCOPED_REUSE', 'behavior_match_only_on_fixed_panel'
                return finish(status, reason, relation='scoped_behavior_match', level='C', behavior=behavior)
            if (source.get('support', {}).get('status') == 'contradicted_in_scope'
                    and source['support'].get('evidence')
                    and source['support'].get('context_ref') == target.get('context_ref')):
                return finish('PROPOSE_REVISION', 'executed_mismatch_in_requested_context', relation='structure_mismatch')
            return finish('NO_COMPATIBLE_CANDIDATE_FOUND', str(mismatch), relation='structure_mismatch')
        result['mappings'] = [deepcopy(option['mapping']) for option in options]
        result['mapping'] = deepcopy(options[0]['mapping']) if len(options) == 1 else {}
        result['mapping_cost'] = sum(left != right for left, right in features.items())
        result['strictly_preserved'] = ['ordered_operators', 'branch_positions', 'lag_and_window']
        result['unverified_premises'] = _premises(source, target)
        for option in options:
            mapping = option['mapping']
            result['evidence']['mapping_fit'].extend(deepcopy(mapping.get('fit_evidence', [])))
            if option['tested']:
                result['evidence']['mapping_test'].extend(deepcopy(mapping.get('test_evidence', mapping.get('evidence', []))))
        behavior = _behavior(source, target, budget)
        if behavior is not None:
            result['behavior'] = behavior
        if len(options) > 1 and len([item for item in options if item['mapping'].get('unit_conversions')
                                    or item['mapping'].get('output_conversion')]) > 1:
            return finish('COMPARE_ALTERNATIVES', 'multiple_explicit_measurement_mappings', relation='typed_rebinding', level='B')
        if ((any(option['needs_binding'] for option in options) and not any(option['tested'] for option in options))
                or source.get('binding_ref') != target.get('binding_ref')):
            return finish('REBIND_THEN_TEST', 'mapping_or_measurement_not_tested', relation='typed_rebinding', level='B')
        if parameters:
            return finish('REFIT_FAMILY', 'only_declared_parameter_constants_differ', relation='same_family', level='B')
        if behavior is not None and not behavior['within_tolerance']:
            return finish('COMPARE_ALTERNATIVES', 'different_predictions_on_fixed_panel',
                          relation='finite_behavior_difference', level='C')
        status, reason = _applicability(source, target, options)
        return finish(status, reason, relation='typed_rebinding' if any(option['needs_binding'] for option in options)
                      else 'exact_compute', level='D')
    except _BudgetExhausted:
        return finish('UNKNOWN_MATCH', 'incomplete_match_search')
    except _Unresolved as unresolved:
        return finish('UNKNOWN_MATCH', str(unresolved))


def match_theories(question, theories, *, target=None, compute_budget=10000, allowed_mappings=None,
                   max_expansion_depth=4):
    """Search the supplied finite library, preserving every theory and mapping."""
    remaining, matches = max(0, int(compute_budget)), []
    target = deepcopy(target if target is not None else question.get('theory_target'))
    if target is None:
        return {'status': 'UNKNOWN_MATCH', 'reason': 'missing_theory_target', 'matches': [],
                'question_id': question.get('id'), 'compute_cost': 0, 'remaining_budget': remaining,
                'confirmation': 'not_established'}
    for key in ('context_ref', 'declared_scope_ref', 'measured_coverage_ref', 'required_premises',
                'premise_statuses', 'behavior_tolerance'):
        if key in question:
            target[key] = deepcopy(question[key])
    for theory in theories:
        result = typed_program_match(theory, target, compute_budget=remaining,
            allowed_mappings=allowed_mappings if allowed_mappings is not None else question.get('allowed_mappings'),
            max_expansion_depth=max_expansion_depth)
        matches.append(result)
        remaining -= result['compute_cost']
        if result['reason'] == 'incomplete_match_search':
            break
    statuses = {item['status'] for item in matches}
    status = next((value for value in ('REUSE_INSTANCE', 'COMPARE_ALTERNATIVES', 'REFIT_FAMILY',
        'REBIND_THEN_TEST', 'SCOPED_REUSE', 'PROPOSE_REVISION', 'UNKNOWN_MATCH') if value in statuses),
        'NO_COMPATIBLE_CANDIDATE_FOUND')
    incomplete = any(item['reason'] == 'incomplete_match_search' for item in matches)
    if incomplete and status != 'REUSE_INSTANCE':
        status = 'UNKNOWN_MATCH'
    return {'status': status, 'reason': ('incomplete_match_search' if status == 'UNKNOWN_MATCH'
                else 'reusable_instance_found_in_incomplete_search') if incomplete else 'finite_library_search',
            'question_id': question.get('id'), 'matches': matches,
            'compute_cost': max(0, int(compute_budget)) - remaining, 'remaining_budget': remaining,
            'search_complete': not incomplete, 'confirmation': 'not_established'}


def prepare_theories_for_question(question, theories, matches):
    """Bind runnable alternatives; do not turn lack of support into rejection.

    Structural alternatives remain alternatives. Actual role/time/unit and
    premise failures block execution until the declared work has been done.
    This function never fits a model or changes the retained theory index.
    """
    rows = matches.get('matches', []) if isinstance(matches, dict) else matches
    indexed = {row['source_id']: row for row in rows}
    target = deepcopy(question.get('theory_target') or {})
    for key in ('context_ref', 'required_premises', 'premise_statuses'):
        if key in question:
            target[key] = deepcopy(question[key])
    roles = question.get('public_role_signature', question.get('role_signature', target.get('role_signature', {})))
    output_unit = question.get('output_unit', target.get('output_unit'))
    eligible, blocked, prepared = [], [], []

    def reject(theory, reason, object_type='measurement_mapping', action=None, **contract):
        required = (deepcopy(action) if action is not None else {
            'type': 'AuthorWorkRequest', 'object_type': object_type, 'required_contract': contract})
        blocked.append({'id': theory['id'], 'reason': reason, 'required_work': required})

    def public_features(query):
        if not isinstance(query, dict):
            return set()
        if query.get('op') == 'Readout':
            if 'finite_readouts' in query:
                return set().union(*(public_features(case) for case in query['finite_readouts']))
            binding = query.get('binding', {})
            names = set(binding.get('features', {}))
            for point in binding.get('history', []):
                names.update(point)
            return names
        if query.get('op') == 'Difference':
            return public_features(query.get('left')) | public_features(query.get('right'))
        if query.get('op') == 'AllOnFiniteSet':
            return set().union(*(public_features(case) for case in query.get('finite_cases', [])))
        return public_features(query.get('query'))

    available = public_features(question.get('query_semantics', {}).get('query_expression'))
    for theory in theories:
        match = indexed.get(theory['id'], {})
        if theory.get('source_status') == 'unavailable_retained_version':
            reject(theory, 'retained_version_unavailable', 'retained_candidate_reference', source_ref=theory.get('source_ref'))
            continue
        if theory.get('compatible') is False or theory.get('support', {}).get('status') in {
                'refuted', 'revoked', 'withdrawn', 'contradicted_in_scope'}:
            reject(theory, 'candidate_incompatible_in_current_scope', 'candidate_revision')
            continue
        missing = _premises(theory, target)
        if missing:
            reject(theory, 'missing_current_premise_support', 'premise_evidence', premise_refs=missing)
            continue
        if theory.get('forecast_ref') is not None:
            if theory.get('source_status') != 'retained_forecast_ready' or not theory.get('_retained_forecast'):
                reject(theory, 'retained_forecast_unavailable', 'retained_forecast_reference',
                       forecast_ref=theory['forecast_ref'])
                continue
            source_roles = theory.get('role_signature')
            names = {name for row in theory['_retained_forecast']['prediction_inputs']
                     for point in [row['features'], *row['history']]
                     for name in point if name != '__target__'}
            if (not isinstance(source_roles, dict) or not isinstance(roles, dict)
                    or set(source_roles) != names or source_roles != roles
                    or any(not isinstance(contract, dict) or not all(
                        key in contract for key in ('role', 'unit', 'time'))
                        for contract in source_roles.values())):
                reject(theory, 'retained_forecast_requires_identical_typed_roles',
                       role_signature=source_roles, target_role_signature=roles)
                continue
            if (output_unit is None or theory.get('output_unit') != output_unit
                    or target.get('context_ref') is None
                    or theory.get('context_ref') != target['context_ref']):
                reject(theory, 'retained_forecast_requires_identical_context_and_unit')
                continue
            def changes_coordinates(mapping):
                if any(left != right for field in ('feature_map', 'role_map')
                       for left, right in mapping.get(field, {}).items()):
                    return True
                for name, conversion in mapping.get('unit_conversions', {}).items():
                    unit = source_roles.get(name, {}).get('unit')
                    if (unit is None or not _conversion(conversion, unit, unit)
                            or conversion['scale'] != 1 or conversion['offset'] != 0):
                        return True
                conversion = mapping.get('output_conversion')
                return conversion is not None and (not _conversion(conversion, output_unit, output_unit)
                    or conversion['scale'] != 1 or conversion['offset'] != 0)

            if any(changes_coordinates(mapping) for mapping in question.get('allowed_mappings', [])):
                reject(theory, 'retained_forecast_mapping_requires_new_executed_predictions')
                continue
            rebound = deepcopy(theory)
            rebound['execution_kind'] = 'retained_forecast_column'
            eligible.append(rebound)
            prepared.append({'id': theory['id'], 'execution_kind': 'retained_forecast_column',
                'forecast_ref': deepcopy(theory['forecast_ref']), 'confirmation': 'not_established',
                'structural_match': 'not_established'})
            continue
        if match.get('status') == 'REFIT_FAMILY':
            reject(theory, 'declared_family_requires_refit', 'fitted_instance',
                   action=theory.get('refit_action'), parameter_changes=match.get('parameter_changes', []))
            continue
        program = _program(theory)
        if program is None:
            reject(theory, 'opaque_python_needs_executable_predictor_binding', 'candidate_program')
            continue
        names = set(program.required_features()) | {
            name for name, _ in program.observed_variables() if name != '__target__'}
        source_roles = theory.get('role_signature', {})
        if any(not all(key in source_roles.get(name, {}) for key in ('role', 'unit', 'time')) for name in names):
            reject(theory, 'missing_role_unit_or_time_contract', role_signature=source_roles)
            continue
        options = [{}]
        for declaration in [*match.get('mappings', []), *question.get('allowed_mappings', [])]:
            if declaration not in options:
                options.append(declaration)
        accepted, failures = [], []
        for declaration in options:
            mapping = {name: declaration.get('feature_map', {}).get(name, name) for name in names}
            changed = any(left != right for left, right in mapping.items())
            reason = None
            for left, right in mapping.items():
                before, after = source_roles[left], roles.get(right)
                if available and right not in available:
                    reason = 'feature_absent_from_public_binding'
                    break
                if roles and after is None:
                    reason = 'feature_absent_from_public_role_contract'
                    break
                if after is None:
                    if left in declaration.get('unit_conversions', {}):
                        reason = 'missing_target_unit_contract'
                        break
                    continue
                if before['time'] != after.get('time'):
                    reason = 'time_role_mismatch'
                    break
                if before['role'] != after.get('role'):
                    if declaration.get('role_map', {}).get(before['role']) != after.get('role'):
                        reason = 'role_binding_mismatch'
                        break
                    changed = True
                conversion = declaration.get('unit_conversions', {}).get(left)
                if before['unit'] != after.get('unit') or conversion is not None:
                    if not _conversion(conversion, before['unit'], after.get('unit')):
                        reason = 'unit_conversion_requires_executable_binding'
                        break
                    changed = True
            output = declaration.get('output_conversion')
            if reason is None and (output is not None or (
                    output_unit is not None and theory.get('output_unit') != output_unit)):
                if output_unit is None or not _conversion(output, theory.get('output_unit'), output_unit):
                    reason = 'output_unit_conversion_requires_executable_binding'
                else:
                    changed = True
            if reason is None and theory.get('context_ref') != target.get('context_ref'):
                changed = True
            tested = (declaration.get('status') == 'tested'
                      and bool(declaration.get('test_evidence') or declaration.get('evidence'))
                      and declaration.get('context_ref') == target.get('context_ref'))
            if reason is None and changed and not tested:
                reason = 'mapping_or_measurement_not_tested'
            if reason:
                failures.append(reason)
                continue
            option = {'feature_map': mapping, 'declaration': declaration, 'changed': changed}
            if not any(item['feature_map'] == mapping and item['changed'] == changed
                       and all(item['declaration'].get(key) == declaration.get(key)
                               for key in ('unit_conversions', 'output_conversion')) for item in accepted):
                accepted.append(option)
        if not accepted:
            reject(theory, failures[-1] if failures else 'no_runnable_binding',
                   role_signature=source_roles, target_role_signature=roles,
                   mappings=question.get('allowed_mappings', []))
            continue
        if len(accepted) > 1:
            reject(theory, 'multiple_runnable_bindings_need_selection', mappings=accepted)
            continue
        option = accepted[0]
        rebound = deepcopy(theory)
        if option['changed']:
            # Library calls are expanded through the existing pure Program
            # semantics so free observed fields inside a saved body are bound too.
            rebound['program'] = _compile_measurement_binding(
                program, option['feature_map'], option['declaration']).to_dict()
            rebound['role_signature'] = {name: deepcopy(roles[name]) for name in option['feature_map'].values()} if roles else {
                right: deepcopy(source_roles[left]) for left, right in option['feature_map'].items()}
            rebound['context_ref'] = deepcopy(target.get('context_ref'))
            if output_unit is not None:
                rebound['output_unit'] = output_unit
            rebound['prepared_binding'] = {'source_id': theory['id'],
                'feature_map': deepcopy(option['feature_map']),
                'unit_conversions': deepcopy(option['declaration'].get('unit_conversions', {})),
                'output_conversion': deepcopy(option['declaration'].get('output_conversion')),
                'source_support': deepcopy(theory.get('support', {})),
                'test_evidence': deepcopy(option['declaration'].get('test_evidence', option['declaration'].get('evidence', []))),
                'confirmation': 'not_established'}
            conversions = list(option['declaration'].get('unit_conversions', {}).values())
            if option['declaration'].get('output_conversion'):
                conversions.append(option['declaration']['output_conversion'])
            if any(item['scale'] != 1 or item['offset'] != 0 for item in conversions):
                # This is a concrete application of the retained instance.
                # Its new unit constants are not source-family fit slots.
                rebound['prepared_binding']['source_parameter_paths'] = [
                    list(path) for path in sorted(_parameter_paths(theory))]
                rebound['parameter_paths'] = []
            rebound['support'] = {'status': 'candidate', 'evidence': [],
                                  'context_ref': deepcopy(target.get('context_ref'))}
            prepared.append({'id': theory['id'], **deepcopy(rebound['prepared_binding'])})
        eligible.append(rebound)
    return {'theories': eligible, 'blocked': blocked, 'prepared': prepared,
            'support_updated': False, 'confirmation': 'not_established'}
