"""Bounded, typed consolidation of retained executable research episodes.

The existing AbstractTemplate performs every instantiation. This first pass
generalizes observed names and constant slots in an ordered Program; it never
turns an arbitrary Python predictor into an expression or an empirical family
into a universal law. Encoding costs are fixed JSON bytes, not formal MDL.
"""

from copy import deepcopy
import json

from .abstractions import AbstractTemplate, extract_templates
from .language import EvaluationContext, Program
from .models import Observation, SearchBudget


class _Incomplete(Exception):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _spend(budget, units=1):
    if not budget.spend(units):
        raise _Incomplete


def _unique(values):
    result, seen = [], set()
    for value in values:
        key = _json(value)
        if key not in seen:
            seen.add(key)
            result.append(deepcopy(value))
    return result


def _evidence_refs(episodes):
    """Notes referring to the same paid row are one support origin."""
    result, seen = [], set()
    identity_fields = ('source_id', 'paid_trace_ref', 'trace_ref', 'file', 'json_path',
                       'archive', 'unit', 'member', 'field', 'record_index', 'row_index')
    for episode in episodes:
        for reference in episode.get('evidence_refs', []):
            identity = ({key: reference[key] for key in identity_fields if key in reference}
                        if isinstance(reference, dict) else reference)
            if not identity:
                identity = reference
            elif isinstance(identity, dict):
                identity = {'source_id': _source_id(episode), **identity}
            key = _json(identity)
            if key not in seen:
                seen.add(key)
                qualified = ({'source_id': _source_id(episode), **reference}
                             if isinstance(reference, dict) else reference)
                result.append(deepcopy(qualified))
    return result


def _source_id(episode):
    reference = episode.get('source_ref', {})
    return episode.get('source_id', reference.get('source_id', episode['episode_id'])
                       if isinstance(reference, dict) else episode['episode_id'])


def _typed_record(episode, program, budget, *, path=(), fragment=False):
    signatures = episode.get('role_signature', {})
    names, roles, parameters, paths, temporal_nodes = {}, {}, [], [], 0

    def visit(node, location=()):
        nonlocal temporal_nodes
        _spend(budget)
        if node.op in {'accumulate', 'window_sum', 'event_delta'}:
            temporal_nodes += 1
        name = node.name
        if node.op in {'feature', 'lag'}:
            contract = signatures.get(name)
            if not isinstance(contract, dict) or any(key not in contract for key in ('role', 'unit', 'time')):
                raise ValueError('missing_role_unit_or_time_contract')
            if name not in names:
                canonical = f'role_{len(names)}'
                names[name] = canonical
                roles[canonical] = deepcopy(contract)
            name = names[name]
        if node.op == 'const':
            paths.append(location)
            parameters.append(float(node.value))
        children = tuple(visit(child, (*location, index)) for index, child in enumerate(node.children))
        return Program(node.op, node.value, name, node.lag, children, node.definition)

    pattern = visit(program)

    def shape(node):
        # The first traversal paid for these bounded in-memory node encodings.
        return (node.op, None if node.op == 'const' else node.value, node.name,
                node.lag, tuple(shape(child) for child in node.children))

    output_unit = episode.get('output_unit')
    if fragment:
        contract = episode.get('fragment_contracts', {}).get('.'.join(map(str, path)), {})
        output_unit = contract.get('output_unit')
    if output_unit is None:
        raise ValueError('missing_fragment_output_contract' if fragment else 'missing_output_unit_contract')
    if 'context_ref' not in episode:
        raise ValueError('missing_context_contract')
    context = episode.get('context_family_ref', episode['context_ref'])
    key = _json([shape(pattern), roles, output_unit, context, fragment])
    return {'key': key, 'episode': episode, 'program': program, 'pattern': pattern,
            'mapping': {canonical: name for name, canonical in names.items()},
            'role_signature': roles, 'parameters': parameters, 'parameter_paths': paths,
            'path': list(path), 'fragment': fragment, 'output_unit': output_unit,
            'nodes': pattern.size, 'temporal_nodes': temporal_nodes}


def instantiate_induction(candidate, instance):
    """Reconstruct a retained source with the ordinary abstraction interface."""
    template = AbstractTemplate.from_dict(candidate['template'])
    return template.instantiate(instance['mapping'], instance['parameters'])


def _counterexamples(candidate, records, budget):
    checks, blocked = [], []
    for record, instance in zip(records, candidate['instances']):
        for example in record['episode'].get('counterexamples', []):
            _spend(budget)
            source_context = record['episode']['context_ref']
            context = example.get('context_ref', source_context)
            domain = 'source' if context == source_context else 'transfer'
            check = {'episode_id': instance['episode_id'], 'source_ref': deepcopy(example.get('source_ref')),
                     'domain': domain, 'context_ref': deepcopy(context)}
            if record['fragment']:
                checks.append({**check, 'status': 'UNKNOWN', 'reason': 'whole_output_does_not_check_fragment'})
                continue
            binding = {**instance, 'mapping': example.get('mapping', instance['mapping']),
                       'parameters': example.get('parameters', instance['parameters'])}
            program = instantiate_induction(candidate, binding)
            payload = example.get('observation')
            if payload is None:
                checks.append({**check, 'status': 'UNKNOWN', 'reason': 'missing_retained_observation'})
                continue
            row = payload if isinstance(payload, Observation) else Observation(
                deepcopy(payload['features']), float(payload['target']),
                tuple(deepcopy(payload.get('history', []))))
            # Ordinary Program evaluation counts its real primitive steps,
            # including historical folds. A conservative bound is reserved
            # before executing, then only the actually entered nodes are paid.
            bound = record['nodes'] * (len(row.history) + 2) ** record['temporal_nodes']
            if not budget.can_spend(bound):
                raise _Incomplete
            evaluation = EvaluationContext(row)
            try:
                value = program.predict(row, evaluation)
            except (ValueError, KeyError, IndexError, ArithmeticError) as error:
                _spend(budget, evaluation.primitive_steps)
                checks.append({**check, 'status': 'UNKNOWN', 'reason': str(error)})
                continue
            _spend(budget, evaluation.primitive_steps)
            tolerance = float(example.get('tolerance', 1e-8))
            if tolerance < 0:
                raise ValueError('counterexample tolerance must be nonnegative')
            passed = abs(value - row.target) <= tolerance
            result = {**check, 'status': 'passed' if passed else 'failed',
                      'prediction': value, 'actual': row.target, 'tolerance': tolerance}
            if example.get('claim') is not None:
                result['claim'] = deepcopy(example['claim'])
            checks.append(result)
            if not passed and domain == 'transfer':
                blocked.append({'context_ref': deepcopy(context), 'mapping': deepcopy(binding['mapping']),
                                'parameters': list(binding['parameters']),
                                'source_ref': deepcopy(example.get('source_ref'))})
    return checks, _unique(blocked)


def _encoding_cost(candidate, records):
    # Count the complete original body, even when it is a named lookup/call.
    originals = _unique(record.get('encoded_source_program', record['program'].to_dict()) for record in records)
    baseline = {'programs': originals, 'instances': [
        {'source_id': _source_id(record['episode']),
         'program_index': originals.index(record.get('encoded_source_program', record['program'].to_dict()))}
        for record in records]}
    exceptions = _unique(value for record in records for value in [
        *record['episode'].get('exceptions', []),
        *([{'lookup_table': record['episode']['lookup_table']}]
          if 'lookup_table' in record['episode'] else [])])
    # Retaining a failed row by reference saves storage, but does not make the
    # exception free in this encoding comparison. Read its already available
    # value for the cost only; the resulting candidate retains just its check.
    failed_refs = {(check['episode_id'], _json(check['source_ref']))
                   for check in candidate['counterexample_checks'] if check['status'] == 'failed'}
    for record in records:
        for example in record['episode'].get('counterexamples', []):
            key = record['episode']['episode_id'], _json(example.get('source_ref'))
            if key not in failed_refs:
                continue
            payload = example.get('observation')
            if isinstance(payload, Observation):
                payload = {'features': payload.features, 'target': payload.target, 'history': list(payload.history)}
            exceptions.append({'source_id': _source_id(record['episode']),
                               'source_ref': deepcopy(example.get('source_ref')), 'observation': payload})
    exceptions = _unique(exceptions)
    compressed = {'definition': candidate['template'], 'selector': candidate['scope'],
                  'instances': [{key: instance[key] for key in ('source_id', 'mapping', 'parameters', 'source_path')}
                                for instance in candidate['instances']], 'exceptions': exceptions}
    original_bytes = len(_json({**baseline, 'exceptions': exceptions}).encode('utf-8'))
    parts = {key: len(_json(value).encode('utf-8')) for key, value in compressed.items()}
    total = len(_json(compressed).encode('utf-8'))
    return {'contract': 'canonical_json_utf8_bytes_v1', 'parameter_precision': 'original_json_float',
            'original_bytes': original_bytes, 'candidate_bytes': total,
            'definition_bytes': parts['definition'], 'selector_bytes': parts['selector'],
            'mapping_and_parameter_bytes': parts['instances'], 'exception_bytes': parts['exceptions'],
            'gain_bytes': original_bytes - total, 'formal_mdl': False}


def _candidate(records, identifier, budget):
    first = records[0]
    episodes = [record['episode'] for record in records]
    roles = tuple(first['mapping'])
    template = AbstractTemplate(identifier, episodes[0]['episode_id'], first['pattern'], roles,
                                tuple(first['mapping'][role] for role in roles),
                                tuple(first['parameter_paths']),
                                kind='subexpression' if first['fragment'] else 'theory',
                                parents=[episode['episode_id'] for episode in episodes])
    instances = [{'episode_id': record['episode']['episode_id'], 'source_id': _source_id(record['episode']),
                  'source_ref': deepcopy(record['episode'].get('source_ref')),
                  'source_path': record['path'], 'mapping': deepcopy(record['mapping']),
                  'inverse_mapping': {value: key for key, value in record['mapping'].items()},
                  'parameters': list(record['parameters'])} for record in records]
    candidate = {'id': identifier, 'kind': 'shared_subexpression' if first['fragment'] else 'typed_program_family',
                 'status': 'EXECUTABLE_CANDIDATE', 'frozen': True, 'template': template.to_dict(),
                 'instances': instances, 'role_signature': deepcopy(first['role_signature']),
                 'output_unit': first['output_unit'], 'parameter_paths': [list(path) for path in first['parameter_paths']],
                 'parameter_hypothesis': 'observed_Program_const_slots',
                 'source_contexts': _unique(episode['context_ref'] for episode in episodes),
                 'declared_scope_ref': _unique(episode.get('declared_scope_ref', episode['context_ref']) for episode in episodes),
                 'source_evidence_refs': _evidence_refs(episodes),
                 'source_ids': _unique(_source_id(episode) for episode in episodes),
                 'source_outcomes': _unique({'episode_id': episode['episode_id'],
                     'source_id': _source_id(episode), 'source_ref': deepcopy(episode.get('source_ref')),
                     **{key: deepcopy(episode[key]) for key in (
                         'status', 'failure_refs', 'adoption', 'independent_results') if key in episode}}
                     for episode in episodes),
                 'scope': {'context_ref': deepcopy(episodes[0].get('context_family_ref', episodes[0]['context_ref'])),
                           'role_signature': deepcopy(first['role_signature']),
                           'output_unit': first['output_unit']},
                 'transfer_status': 'not_established', 'global_support': False}
    reconstruction = []
    for record, instance in zip(records, instances):
        _spend(budget, record['nodes'])
        rebuilt = instantiate_induction(candidate, instance)
        if rebuilt != record['program']:
            raise ValueError('source reconstruction failed')
        reconstruction.append({'episode_id': instance['episode_id'], 'source_ref': instance['source_ref'],
                               'source_path': instance['source_path'], 'exact': True,
                               'comparison': 'ordered_expanded_Program',
                               'target': 'source_fragment' if first['fragment'] else 'source_program'})
    candidate['source_reconstruction'] = reconstruction
    candidate['independent_source_count'] = len(candidate['source_ids'])
    candidate['counterexample_checks'], candidate['blocked_transfer_bindings'] = _counterexamples(candidate, records, budget)
    source_checks = [check for check in candidate['counterexample_checks'] if check['domain'] == 'source']
    candidate['source_counterexample_status'] = ('not_tested' if not source_checks else
        'failed' if any(check['status'] == 'failed' for check in source_checks) else
        'UNKNOWN' if any(check['status'] == 'UNKNOWN' for check in source_checks) else 'passed_retained_checks')
    candidate['encoding_cost'] = _encoding_cost(candidate, records)
    candidate['active_library'] = bool(candidate['encoding_cost']['gain_bytes'] > 0
        and candidate['independent_source_count'] > 1 and candidate['source_counterexample_status'] != 'failed')
    candidate['adoption'] = 'optional_candidate_generation_only'
    return candidate


def consolidate_theories(episodes, *, allocation=None, candidate_id='induction'):
    """Compile finite same-topic families; preserve unsupported work explicitly.

    ``allocation`` has compute and max_groups. Compute charges each structural
    visit/reconstruction node and actually evaluated primitive prediction node.
    Missing contracts and incompatible structures are not guessed from prose.
    """
    allocation = allocation or {}
    compute = allocation.get('compute', 10000)
    max_groups = allocation.get('max_groups', 32)
    if type(compute) is not int or compute < 0 or type(max_groups) is not int or max_groups < 0:
        raise ValueError('induction allocation must contain nonnegative integers')
    budget = SearchBudget(compute)
    groups, excluded, author_work, candidates = {}, [], [], []
    incomplete = False
    grouping_completed = False
    try:
        for episode in episodes:
            _spend(budget)
            if 'program' not in episode or episode['program'] is None:
                author_work.append({'episode_id': episode['episode_id'], 'source_ref': deepcopy(episode.get('source_ref')),
                                    'type': 'AuthorWorkRequest', 'reason': 'arbitrary_python_requires_authored_Program'})
                continue
            program = episode['program']
            program = program if isinstance(program, Program) else Program.from_dict(program)
            encoded_source_program = program.to_dict()
            program = program.expanded()
            try:
                record = _typed_record(episode, program, budget)
                record['encoded_source_program'] = encoded_source_program
            except ValueError as error:
                excluded.append({'episode_id': episode['episode_id'], 'source_ref': deepcopy(episode.get('source_ref')),
                                 'reason': str(error)})
                continue
            groups.setdefault(record['key'], []).append(record)
            # Existing extractor supplies only useful observed ordered fragments.
            # A fragment needs its own output type, not the whole task's unit.
            if episode.get('fragment_contracts'):
                _spend(budget, record['nodes'])
                fragments = extract_templates(program, episode['episode_id'])[1:]
                def locate(node, wanted, path=()):
                    if node == wanted:
                        return path
                    for index, child in enumerate(node.children):
                        found = locate(child, wanted, (*path, index))
                        if found is not None:
                            return found
                    return None
                for template in fragments:
                    fragment = template.instantiate(dict(zip(template.roles, template.source_features)))
                    path = locate(program, fragment)
                    if path is None or '.'.join(map(str, path)) not in episode['fragment_contracts']:
                        continue
                    fragment_record = _typed_record(episode, fragment, budget, path=path, fragment=True)
                    groups.setdefault(fragment_record['key'], []).append(fragment_record)
        grouping_completed = True
        eligible = [records for records in groups.values() if len({r['episode']['episode_id'] for r in records}) > 1]
        for index, records in enumerate(eligible):
            if index >= max_groups:
                incomplete = True
                break
            candidates.append(_candidate(records, f'{candidate_id}:{index}', budget))
    except _Incomplete:
        incomplete = True
    grouped_ids = {instance['episode_id'] for candidate in candidates for instance in candidate['instances']}
    for records in groups.values():
        budget_unknown = incomplete and (not grouping_completed or
            len({record['episode']['episode_id'] for record in records}) > 1)
        for record in records:
            if record['episode']['episode_id'] not in grouped_ids:
                author_work.append({'episode_id': record['episode']['episode_id'],
                                    'source_ref': deepcopy(record['episode'].get('source_ref')),
                                    'type': 'AuthorWorkRequest',
                                    **({'status': 'UNKNOWN', 'reason': 'induction_budget_exhausted'}
                                       if budget_unknown else {'reason': 'no_compatible_ordered_typed_peer'})})
    return {'status': 'UNKNOWN' if incomplete else 'completed', 'candidates': candidates,
            'groups': [{'episode_ids': _unique(record['episode']['episode_id'] for record in records),
                        'kind': 'shared_subexpression' if records[0]['fragment'] else 'typed_program_family'}
                       for records in groups.values()],
            'excluded': excluded, 'author_work': _unique(author_work),
            'actual_compute_cost': budget.spent, 'budget_exhausted': incomplete,
            'support_updated': False, 'new_observations': 0, 'llm_calls': 0}
