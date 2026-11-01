"""Reidentify one explicitly bound relation group and its complete readout.

The source graph supplies a testable family, not the target's parameter range.
Candidates come from actual revisions; their usefulness still needs paired and
independent evidence through the ordinary research-program consumer.
"""

from copy import deepcopy
from math import isfinite

from .language import Program
from .models import Observation, SearchBudget
from .relation_kernel_priors import parameter_relations
from .relation_language import relation_objects
from .relation_revisions import _distances
from .relation_uncertainty import RelationIntervalResearch, _bind
from .structural_research_program import research_program_fit_contract


def _saved_binding(payload):
    return next((step for step in reversed(payload.get('selection_steps', []))
                 if step['operation'] == 'resolve_relation_intervals'), None)


def _occurrences(program, symbol, path=()):
    found = []
    if program.op in {'neighbor_sum', 'relation_sum'} and program.children[0] == Program('arg', name=symbol):
        found.append(list(path))
    for index, child in enumerate(program.children):
        found.extend(_occurrences(child, symbol, (*path, index)))
    return found


def transfer_partition_view(payload):
    """Read saved symbols, never infer sharing from equal fitted numbers."""
    fit = research_program_fit_contract(payload)
    saved = _saved_binding(payload)
    result = {'status': 'binding_unavailable', 'coordinate_basis': fit['coordinate_basis'],
              'term_nodes': list(payload['term_nodes']), 'groups': []}
    if not saved or not saved.get('parameter_terms'):
        return result
    terms = [Program.from_dict(term) for term in saved['parameter_terms']]
    parameters = saved['parameters']
    values = {item['symbol']: Program('const', value=item['radius']) for item in parameters}
    if len(terms) != len(fit['fit_terms']) or any(
            _bind(term, values).expanded().simplified() != actual.expanded().simplified()
            for term, actual in zip(terms, fit['fit_terms'])):
        return {**result, 'status': 'binding_does_not_reconstruct_columns'}
    for index, item in enumerate(parameters):
        symbol = item['symbol']
        occurrences = [{'term_node': key, 'path': path}
                       for key, term in zip(payload['term_nodes'], terms)
                       for path in _occurrences(term, symbol)]
        if occurrences:
            result['groups'].append({'symbol': symbol, 'parameter_index': index,
                'source_value': item['radius'],
                'source_intervals': deepcopy(item.get('radius_intervals', [])),
                'binding': 'saved_parameter_symbol', 'occurrences': occurrences})
    result.update(status='ready' if result['groups'] else 'binding_unavailable',
                  parameter_terms=deepcopy(saved['parameter_terms']))
    return result


def _order_key(program):
    """Ordering for correspondence only, never rewrite floating-point execution."""
    children = program.children
    if program.op in {'neighbor_sum', 'relation_sum'}:
        offset = 2 if program.op == 'relation_sum' else 1
        triples = tuple(sorted(tuple(_order_key(part) for part in children[index:index + 3])
                               for index in range(offset, len(children), 3)))
        return (program.op, str(program.name), '', program.lag,
                ((_order_key(children[1]),) if offset == 2 else ()) + triples)
    keys = tuple(_order_key(child) for child in children)
    if program.op in {'object_sum', 'add', 'mul'}:
        keys = tuple(sorted(keys))
    return (program.op, str(program.name), str(program.value), program.lag, keys)


def _match_parameters(template, actual, symbols, values):
    if template.op == 'arg' and template.name in symbols:
        if actual.op != 'const' or not isfinite(actual.value):
            return False
        if template.name in values and values[template.name] != actual.value:
            return False
        values[template.name] = actual.value
        return True
    if ((template.op, template.name, template.value, template.lag) !=
            (actual.op, actual.name, actual.value, actual.lag)
            or len(template.children) != len(actual.children)):
        return False
    left, right = list(template.children), list(actual.children)
    if template.op in {'neighbor_sum', 'relation_sum'}:
        offset = 2 if template.op == 'relation_sum' else 1
        def ordered(children):
            triples = [children[index:index + 3] for index in range(offset, len(children), 3)]
            triples.sort(key=lambda triple: tuple(_order_key(part) for part in triple))
            return children[:offset] + [part for triple in triples for part in triple]
        left, right = ordered(left), ordered(right)
    elif template.op in {'object_sum', 'add', 'mul'}:
        left.sort(key=_order_key)
        right.sort(key=_order_key)
    return all(_match_parameters(a, b, symbols, values) for a, b in zip(left, right))


def _observed_feature_rebinding(old, new):
    """Separate a feature assignment change from changed structure/parameters."""
    changed = False

    def match(left, right):
        nonlocal changed
        if left == right:
            return True
        if left.op == right.op == 'feature' and left.lag == right.lag:
            changed = changed or left.name != right.name
            return True
        if (left.op, left.name, left.value, left.lag) != (right.op, right.name, right.value, right.lag):
            return False
        return len(left.children) == len(right.children) and all(
            match(a, b) for a, b in zip(left.children, right.children))

    return (len(old) == len(new) and all(match(a.expanded(), b.expanded()) for a, b in zip(old, new))
            and changed)


def _discrete_partition_proposals(before, after, provenance):
    """Compile observed binding/temporal revisions using existing public contracts.

    A saved after-model locates a changed computation; its fitted coefficients
    and parameter values are not search answers. Each proposed method refits
    the complete columns on its caller's data and emits at most one draft.
    """
    from .abstractions import AbstractTemplate
    from .mechanism_revision import _replace_shared
    from .research_drafts import _rebind_step
    from .research_programs import _temporal

    old = research_program_fit_contract(before)['fit_terms']
    new = research_program_fit_contract(after)['fit_terms']
    if len(old) != len(new):
        return []
    saved = _saved_binding(before)
    candidates = []

    def append(kind, alternatives, edit):
        payload = deepcopy(before)
        payload['transfer_partition'] = {'kind': kind, 'released_groups': [],
            'max_attempts': len(alternatives), 'max_candidates': 4,
            'alternatives': alternatives, 'source_revision': deepcopy(provenance),
            'search_domain': 'observed_local_revision_existing_vocabulary', 'status': 'proposed'}
        candidates.append({'research_program': payload,
            'status': 'requires_paired_evaluation', 'edit': edit})

    # Restrict permutations to typed object-field assignments. The generic
    # AbstractTemplate fallback over arbitrary feature names is not Pi2 evidence.
    template_value = before.get('input_template')
    schema = after.get('source_schema') or before.get('source_schema')
    same_schema = not (before.get('source_schema') and after.get('source_schema')) or (
        before['source_schema'] == after['source_schema'])
    if template_value and schema and same_schema and _observed_feature_rebinding(old, new):
        template = AbstractTemplate.from_dict(template_value)
        fields = schema['fields']
        names = [name for obj in schema['objects'] for name in fields[obj].values()]
        roles_ready = (set(template.object_roles) == set(template.roles)
            and all(role in fields[obj] for obj in schema['objects']
                    for _, role in template.object_roles.values()))
        legal, observed = [], None
        if roles_ready:
            from itertools import islice
            for mapping in islice(template.mappings(names, object_schema=schema), 48):
                # mappings() also emits historical and identity assignments;
                # check their field roles against the public contract as well.
                assigned = {}
                for role, (index, field) in template.object_roles.items():
                    objects = [obj for obj in schema['objects']
                               if fields[obj][field] == mapping[role]]
                    if len(objects) != 1 or (index in assigned and assigned[index] != objects[0]):
                        break
                    assigned[index] = objects[0]
                else:
                    if len(set(assigned.values())) != len(assigned):
                        continue
                    rename = dict(zip(template.source_features,
                                      (mapping[role] for role in template.roles)))
                    rebound = _rebind_step(before, rename)
                    terms = research_program_fit_contract(rebound)['fit_terms']
                    if terms == old:
                        continue
                    alternative = {'fit_terms': [term.to_dict() for term in terms],
                        'source_binding': rename}
                    binding = _saved_binding(rebound)
                    if binding and binding.get('parameter_terms'):
                        alternative['parameter_terms'] = deepcopy(binding['parameter_terms'])
                    if alternative not in legal:
                        legal.append(alternative)
                    if terms == new:
                        observed = alternative
        if observed is not None:
            alternatives = [observed, *(item for item in legal if item != observed)][:4]
            append('Pi2', alternatives, {'operation': 'rebind_public_object_roles',
                'source_role_contract': deepcopy(template.object_roles),
                'source_revision_kind': 'observed_lawful_binding_revision'})
            # The ordinary caller must bind every coordinate that this method
            # may actually test, including newly addressed objects. Retaining
            # only the old answer's template leaves the new branch unbound on
            # a renamed successor.
            used = sorted({name for terms in [old, *[
                [Program.from_dict(term) for term in item['fit_terms']] for item in alternatives]]
                for term in terms for name in term.required_features()})
            object_fields = {name: (index, role) for index, obj in enumerate(schema['objects'])
                             for role, name in fields[obj].items()}
            if all(name in object_fields for name in used):
                roles = tuple(f'input_{index}' for index in range(len(used)))
                rebound_template = AbstractTemplate('research_inputs', 'research_procedure',
                    Program('object_sum', children=tuple(Program('feature', name=role) for role in roles)),
                    roles, tuple(used), (), object_roles={role: object_fields[name]
                        for role, name in zip(roles, used)})
                candidates[-1]['research_program'].update(input_template=rebound_template.to_dict(),
                    source_schema=deepcopy(schema), input_features=used)
            else:
                candidates.pop()  # no complete public call contract for this edit

    # Locate one repeated subtree change. Reuse the existing shared replacement
    # semantics so that every occurrence/fit column keeps the same dependency.
    changes = []
    temporal = {'window_sum', 'event_delta', 'accumulate'}

    def compare(left, right):
        if left == right:
            return True
        if right.op in temporal and right.children[0] == left:
            changes.append((left, right.op))
            return True
        if (left.op in temporal and right.op in temporal
                and left.children[0] == right.children[0]):
            changes.append((left, right.op))
            return True
        if (left.op, left.name, left.value, left.lag, left.definition) != (
                right.op, right.name, right.value, right.lag, right.definition):
            return False
        return len(left.children) == len(right.children) and all(
            compare(a, b) for a, b in zip(left.children, right.children))

    if all(compare(a, b) for a, b in zip(old, new)) and changes and len(set(changes)) == 1:
        context, family = changes[0]
        operand = context.children[0] if context.op in temporal else context
        alternatives = []
        for replacement in _temporal(operand):
            if replacement.op != family or replacement == context:
                continue
            terms = [_replace_shared(term, context, replacement)[0] for term in old]
            alternative = {'fit_terms': [term.to_dict() for term in terms],
                'local_context': context.to_dict(), 'replacement': replacement.to_dict()}
            if saved and saved.get('parameter_terms'):
                # Replace the corresponding symbolic subtree, preserving every
                # saved parameter symbol rather than reopening by numeric value.
                bindings = {item['symbol']: Program('const', value=item['radius'])
                            for item in saved['parameters']}
                symbolic = []
                def replace_symbolic(node):
                    if _bind(node, bindings).simplified() == context:
                        inner = node.children[0] if node.op in temporal else node
                        from dataclasses import replace
                        return replace(replacement, children=(inner, *replacement.children[1:]))
                    from dataclasses import replace
                    return replace(node, children=tuple(replace_symbolic(child) for child in node.children))
                symbolic = [replace_symbolic(Program.from_dict(term)) for term in saved['parameter_terms']]
                if any(_bind(term, bindings).simplified() != actual
                       for term, actual in zip(symbolic, terms)):
                    continue
                alternative['parameter_terms'] = [term.to_dict() for term in symbolic]
            alternatives.append(alternative)
        if alternatives:
            append('Pi3', alternatives[:4], {'operation': 'revise_one_temporal_subtree',
                'source_revision_kind': 'observed_local_temporal_revision',
                'context': context.to_dict(), 'operator_family': family})

    # The existing relation researcher also revises orientation. Learn that
    # local operation from a real before/after edit; the source direction does
    # not decide the target direction. Use exact shared-subtree replacement so
    # equal numeric radii in unrelated parameter groups are not merged.
    directions = []
    def compare_direction(left, right):
        if left == right:
            return True
        if (left.op == right.op and left.op in {'neighbor_sum', 'relation_sum'}
                and left.children == right.children and left.name != right.name):
            directions.append(left)
            return True
        if (left.op, left.name, left.value, left.lag, left.definition) != (
                right.op, right.name, right.value, right.lag, right.definition):
            return False
        return len(left.children) == len(right.children) and all(
            compare_direction(a, b) for a, b in zip(left.children, right.children))

    if all(compare_direction(a, b) for a, b in zip(old, new)) and len(set(directions)) == 1:
        from dataclasses import replace
        context = directions[0]
        alternatives = []
        for direction in (None, 'right', 'left'):
            if direction == context.name:
                continue
            replacement = replace(context, name=direction)
            terms = [_replace_shared(term, context, replacement)[0] for term in old]
            alternative = {'fit_terms': [term.to_dict() for term in terms],
                'local_context': context.to_dict(), 'replacement': replacement.to_dict()}
            if saved and saved.get('parameter_terms'):
                bindings = {item['symbol']: Program('const', value=item['radius'])
                            for item in saved['parameters']}
                def replace_direction(node):
                    if _bind(node, bindings).simplified() == context:
                        return replace(node, name=direction)
                    return replace(node, children=tuple(replace_direction(child) for child in node.children))
                symbolic = [replace_direction(Program.from_dict(term)) for term in saved['parameter_terms']]
                if any(_bind(term, bindings).simplified() != actual for term, actual in zip(symbolic, terms)):
                    continue
                alternative['parameter_terms'] = [term.to_dict() for term in symbolic]
            alternatives.append(alternative)
        if alternatives:
            append('Pi3', alternatives, {'operation': 'revise_one_relation_direction',
                'source_revision_kind': 'observed_local_relation_revision',
                'context': context.to_dict(), 'operator_family': context.op})
    return candidates


def propose_transfer_partitions(before, after, *, source_ref=None, before_ref=None, after_ref=None):
    """Propose a target-identification edit from a recorded before/after revision.

    The after graph detects what changed. It never supplies a target search
    value, an initialization or a purported isolated effect of a joint change.
    """
    view = transfer_partition_view(before)
    baseline = deepcopy(before)
    baseline['transfer_partition'] = {'kind': 'Pi0', 'released_groups': [],
        'max_attempts': 1, 'max_candidates': 1, 'status': 'proposed'}
    result = {'status': view['status'], 'candidates': [], 'baseline_research_program': baseline}
    provenance = {'source_ref': deepcopy(source_ref), 'before_ref': deepcopy(before_ref),
                  'after_ref': deepcopy(after_ref), 'isolated_group_effect': False,
                  'source_operation': 'not_inferred_from_output_graph'}
    discrete = _discrete_partition_proposals(before, after, provenance)
    if view['status'] != 'ready':
        if discrete:
            result.update(status='proposed', candidates=discrete)
        return result
    after_fit = research_program_fit_contract(after)
    symbols = {item['symbol'] for item in view['groups']}
    values = {}
    templates = [Program.from_dict(term).expanded().simplified() for term in view['parameter_terms']]
    aligned = len(templates) == len(after_fit['fit_terms']) and all(
        _match_parameters(template, actual.expanded().simplified(), symbols, values)
        for template, actual in zip(templates, after_fit['fit_terms']))
    changed = [group for group in view['groups']
               if aligned and group['symbol'] in values and values[group['symbol']] != group['source_value']]
    revision_kind = 'relation_and_readout_revision'
    if not aligned:
        # A complete after fit may have changed composition too. That is a
        # source for a hypothesis to test, never isolated Pi1 utility evidence.
        saved_after = _saved_binding(after)
        actual = {item['symbol']: item['radius'] for item in saved_after.get('parameters', [])} if saved_after else {}
        changed = [group for group in view['groups'] if group['symbol'] in actual
                   and actual[group['symbol']] != group['source_value']]
        if not changed and len(view['groups']) == 1:
            radius = after.get('structural_replay', {}).get('radius')
            if radius is not None and radius != view['groups'][0]['source_value']:
                changed = view['groups']
        revision_kind = 'joint_variant'
    provenance = {'source_ref': deepcopy(source_ref), 'before_ref': deepcopy(before_ref),
                  'after_ref': deepcopy(after_ref), 'revision_kind': revision_kind,
                  'isolated_group_effect': False,
                  'source_operation': 'not_inferred_from_output_graph',
                  'proposed_operation': 'release_group'}
    for group in changed:
        proposal = deepcopy(before)
        proposal['transfer_partition'] = {'kind': 'Pi1',
            'released_groups': [group['symbol']], 'max_attempts': 48, 'max_candidates': 12,
            'search_domain': 'current_paid_geometry', 'source_interval_role': 'initialization_only',
            'source_revision': deepcopy(provenance), 'status': 'proposed'}
        result['candidates'].append({'research_program': proposal, 'status': 'requires_paired_evaluation',
            'edit': {'operation': 'release_group', 'group': group['symbol'],
                     'source_revision_kind': revision_kind}})
    result['status'] = 'proposed' if result['candidates'] else 'no_observed_group_revision'
    result['candidates'].extend(discrete)
    if discrete:
        result['status'] = 'proposed'
    return result


def _geometry_frames(rows, terms, relations):
    depth = max((term.history_depth for term in terms), default=0)
    relation_depth = max((term.history_depth for term in relations), default=0)
    fields = sorted({name for relation in relations for name, _ in relation.observed_variables()})
    frames = {}
    for row in rows:
        for position in range(max(0, len(row.history) - depth), len(row.history) + 1):
            point = row.features if position == len(row.history) else row.history[position]
            past = row.history[:position]
            key = (tuple(point.get(name) for name in fields),
                   tuple(tuple(frame.get(name) for name in fields)
                         for frame in past[-relation_depth:]) if relation_depth else ())
            frames.setdefault(key, Observation(point, 0., past))
    return list(frames.values())


def _quote_parent_bound_continuation(baseline, continuation, rows, feedback, max_attempts):
    """Cover every legal first-stage structure before its coefficients exist."""
    saved = _saved_binding(baseline)
    contract = continuation.get('transfer_partition', {})
    if not saved or contract.get('kind') != 'Pi1':
        return {'status': 'parent_parameter_dependency_unavailable', 'required_compute': None}
    values = {item['symbol']: Program('const', value=item['radius']) for item in saved['parameters']}
    local = baseline['transfer_partition']
    alternatives = local['alternatives'][:min(max_attempts, local['max_attempts'], 4)]
    quotes = []
    for index, alternative in enumerate(alternatives):
        declared = [Program.from_dict(term) for term in alternative['fit_terms']]
        symbolic = [Program.from_dict(term) for term in alternative.get('parameter_terms', ())]
        if len(symbolic) != len(declared) or any(
                _bind(term, values).expanded().simplified() != actual.expanded().simplified()
                for term, actual in zip(symbolic, declared)):
            return {'status': 'parent_column_binding_unavailable', 'required_compute': None,
                    'alternative_index': index}
        groups = [item['symbol'] for item in saved['parameters']
                  if any(_occurrences(term, item['symbol']) for term in symbolic)]
        if len(groups) != 1:
            return {'status': 'one_explicit_parent_group_required', 'required_compute': None,
                    'alternative_index': index}
        relations = list(dict.fromkeys(node for term in symbolic
                                      for node in parameter_relations(term, groups[0])))
        frames = _geometry_frames(rows, symbolic, relations)
        geometry = len(frames) * sum(3 * len(relation_objects(node)) for node in relations)
        # Either actual fitted sum has at most one intercept, one multiply and
        # one addition per column. This covers coefficients not yet fitted;
        # source coefficients or a single source output are not an upper bound.
        initialization = 2 * (1 + sum(term.expanded().size + 3 for term in declared))
        attempts = min(max_attempts, contract['max_attempts'], 48)
        second = contract.get('second_stage')
        followup = int(second.get('compute_limit', 15000)) if second and feedback else 0
        quotes.append({'alternative_index': index, 'fit_columns': len(declared),
            'required_compute': initialization + geometry + attempts * (len(declared) + 2) * len(rows)
                + followup + (len(feedback) if followup else 0),
            'initialization_compute': initialization, 'geometry_compute': geometry,
            'followup_compute_upper': followup})
    if not quotes:
        return {'status': 'parent_alternatives_unavailable', 'required_compute': None}
    return {'status': 'quoted', 'required_compute': max(item['required_compute'] for item in quotes),
            'parent_alternatives': quotes,
            'quote_scope': 'largest_bound_continuation_over_legal_first_stage_alternatives'}


def quote_transfer_partition(payload, training, *, max_attempts=48, feedback_rows=()):
    """Upper bound without predictions; column caches can reduce actual costs."""
    if payload.get('staged_research'):
        staged = payload['staged_research']
        rows, feedback = list(training), list(feedback_rows)
        quotes = {'baseline': quote_transfer_partition(staged['baseline'], rows, max_attempts=max_attempts)}
        quotes['continuation'] = (_quote_parent_bound_continuation(
            staged['baseline'], staged['continuation'], rows, feedback, max_attempts)
            if staged['operation'] == 'revise_then_reidentify' else
            quote_transfer_partition(staged['continuation'], rows,
                max_attempts=max_attempts, feedback_rows=feedback))
        if any(item.get('required_compute') is None for item in quotes.values()):
            return {'status': 'unquoted_staged_dependency', 'required_compute': None, 'stages': quotes}
        guard_compute = len(staged.get('conditions', [staged['condition']]))
        return {'status': 'quoted', 'required_compute': sum(item['required_compute'] for item in quotes.values())
                    + 2 * len(feedback) + guard_compute,
                'stages': quotes, 'guard_compute': guard_compute, 'fit_rows': len(rows),
                'assessment_upper': 2 * len(feedback), 'maximum_new_candidates': 1,
                'quote_scope': 'both_complete_bounded_stages_before_paid_reuse_or_early_stop'}
    fit = research_program_fit_contract(payload)
    rows = list(training)
    contract = payload['transfer_partition']
    attempts = min(max_attempts, contract['max_attempts'], 48)
    initialization = geometry = 0
    fit_compute = None
    if contract['kind'] == 'Pi1':
        view = transfer_partition_view(payload)
        if view['status'] != 'ready':
            return {'status': view['status'], 'required_compute': None}
        symbol = contract['released_groups'][0]
        terms = [Program.from_dict(term) for term in view['parameter_terms']]
        relations = list(dict.fromkeys(node for term in terms for node in parameter_relations(term, symbol)))
        frames = _geometry_frames(rows, terms, relations)
        saved = _saved_binding(payload)
        values = {item['symbol']: Program('const', value=item['radius']) for item in saved['parameters']}
        initialization = _bind(Program.from_dict(saved['parameter_template']), values).simplified().expanded().size + fit['program'].expanded().size
        geometry = len(frames) * sum(3 * len(relation_objects(relation)) for relation in relations)
    elif contract['kind'] in {'Pi2', 'Pi3'}:
        attempts = min(attempts, len(contract['alternatives']), 4)
        fit_compute = sum((len(item['fit_terms']) + 2) * len(rows)
                          for item in contract['alternatives'][:attempts])
    else:
        attempts = 1
    followup = payload['transfer_partition'].get('second_stage')
    feedback = list(feedback_rows)
    followup_upper = int(followup.get('compute_limit', 15000)) if followup and feedback else 0
    if fit_compute is None:
        fit_compute = attempts * (len(fit['fit_terms']) + 2) * len(rows)
    return {'status': 'quoted', 'required_compute': initialization + geometry + fit_compute
                + followup_upper + (len(feedback) if followup_upper else 0),
            'initialization_compute': initialization, 'geometry_compute': geometry,
            'attempt_limit': attempts, 'fit_rows': len(rows), 'validation_compute': 0,
            'followup_compute_upper': followup_upper,
            'feedback_readout_upper': len(feedback) if followup_upper else 0,
            'followup_quote_scope': 'bounded_interval_call_not_complete_parameter_domain'}


def replay_staged_partition(payload, training, budget, *, risk_quantile=.5,
                            feature_mapping=None, feedback_rows=(), tolerance=1e-7,
                            paid_stages=None):
    """Run a complete first research before deciding whether to reidentify a group.

    Cached stage results must describe this exact executable and paid view.
    The learned guard sees only the completed first stage, never its successor.
    """
    from .evaluation import initial_prediction, loss
    from .method_conditions import _conflict_predictions, method_condition_features
    from .research_drafts import _rebind_step
    from .research_programs import research_program_replay_inputs
    from .staged_method_research import baseline_return_profile

    if feature_mapping:
        payload = _rebind_step(payload, feature_mapping)
    contract = payload['staged_research']
    parent_bound = contract['operation'] == 'revise_then_reidentify'
    baseline_kind = contract['baseline'].get('transfer_partition', {}).get('kind')
    if ((parent_bound and baseline_kind not in {'Pi2', 'Pi3'})
            or (not parent_bound and (contract['operation'] != 'refit_then_reidentify' or baseline_kind != 'Pi0'))
            or contract['continuation'].get('transfer_partition', {}).get('kind') != 'Pi1'
            or any(contract[key].get('staged_research') for key in ('baseline', 'continuation'))):
        raise ValueError('Staged replay requires a Pi0 refit or Pi2/Pi3 revision followed by Pi1')
    rows, feedback = list(training), list(feedback_rows)
    started = budget.spent
    result = {'status': 'baseline_pending', 'execution_complete': False,
        'budget_incomplete': False, 'compute_cost': 0, 'new_observations': 0,
        'independent_support': False, 'data_role': 'exposed_adaptive_development',
        'selected_stage': 'baseline', 'guard': {'status': 'not_evaluated'},
        'training_rows': len(rows), 'feedback_rows': len(feedback)}
    if parent_bound:
        result.update(operation=contract['operation'], risk_quantile=risk_quantile, tolerance=tolerance,
                      input_scope='same_training_and_feedback_for_both_stages')
    fallback = initial_prediction(rows, risk_quantile)

    def stage(name, body=None, paid_initial=None):
        body = contract[name] if body is None else body
        paid = (paid_stages or {}).get(name, {})
        reused = bool(paid.get('execution_complete') and paid.get('entry') is not None
            and paid.get('replay_inputs') == research_program_replay_inputs(body)
            and paid.get('training') == rows and paid.get('feedback') == feedback
            and paid.get('risk_quantile') == risk_quantile and paid.get('tolerance') == tolerance)
        before, denied = budget.spent, budget.denied_requests
        entry = (deepcopy(paid['entry']) if reused else replay_transfer_partition(body, rows, budget,
            risk_quantile=risk_quantile, feedback_rows=feedback if name == 'continuation' else (),
            tolerance=tolerance, **({'paid_initial': paid_initial} if paid_initial is not None else {})))
        detail = {'reused': reused, 'execution_complete': False, 'compute_cost': 0,
                  'budget_incomplete': False}
        if parent_bound:
            detail['replay_inputs'] = research_program_replay_inputs(body)
            if name == 'continuation':
                detail.update(parent_stage='baseline',
                    parent_parameter_revision=deepcopy(body['parent_parameter_revision']))
        if reused:
            detail.update(original_cost_ref=deepcopy(paid.get('cost_ref')),
                          feedback_loss_refs=deepcopy(paid.get('feedback_loss_refs', [])))
        result[name] = detail
        if entry is None:
            detail.update(status='no_delivered_fit', compute_cost=budget.spent - before,
                          budget_incomplete=budget.denied_requests != denied)
            return None
        search = entry.get('transfer_partition_search', {})
        followup = entry.get('transfer_partition_followup', {})
        complete = search.get('completed', False) and (
            not body.get('transfer_partition', {}).get('second_stage')
            or followup.get('execution_complete', False))
        measured = paid.get('feedback_loss') if reused else followup.get('feedback_loss')
        if measured is None and feedback and budget.spend(len(feedback)):
            measured = loss(entry['candidate'].program, feedback, fallback, risk_quantile)
        trained = entry.get('training_loss')
        finite = (type(trained) in (int, float) and isfinite(trained)
                  and type(measured) in (int, float) and isfinite(measured))
        combined = ((len(rows) * trained + len(feedback) * measured) / (len(rows) + len(feedback))
                    if finite else None)
        detail.update(program=entry['candidate'].program.to_dict(), training_loss=trained,
            feedback_loss=measured, all_paid_loss=combined,
            execution_complete=bool(complete and finite and feedback),
            status='complete_paid_assessment' if complete and finite and feedback else 'incomplete_stage',
            compute_cost=budget.spent - before,
            budget_incomplete=budget.denied_requests != denied or followup.get('budget_incomplete', False))
        return entry

    def finish(entry, status, complete=False):
        if entry is None:
            return None
        result.update(status=status, execution_complete=complete, compute_cost=budget.spent - started,
            budget_incomplete=any(result.get(name, {}).get('budget_incomplete', False)
                                  for name in ('baseline', 'continuation')))
        selected = deepcopy(entry)
        detail = result[result['selected_stage']]
        result['feedback_loss'] = detail.get('feedback_loss')
        result['all_paid_loss'] = detail.get('all_paid_loss')
        selected.update(staged_research_result=deepcopy(result))
        selected['research_program']['staged_research'] = deepcopy(contract)
        selected['research_program']['staged_research_result'] = deepcopy(result)
        # Existing callers already consume search completion. A stopped or
        # unfunded continuation must not masquerade as the complete Pi0 alone.
        selected['transfer_partition_search'] = {
            **deepcopy(selected.get('transfer_partition_search', {})), 'completed': complete,
            'status': status, 'compute_cost': budget.spent - started}
        selected['research_program']['transfer_partition_search'] = deepcopy(selected['transfer_partition_search'])
        return selected

    baseline = stage('baseline')
    if baseline is None or not result['baseline']['execution_complete']:
        return finish(baseline, 'baseline_incomplete')
    first = result['baseline']
    profile = baseline_return_profile({'fit_loss': first['all_paid_loss'], 'execution_complete': True}, tolerance)
    result['guard']['profile'] = profile
    features = method_condition_features(profile, contract['coordinate_paths'])
    conditions = contract.get('conditions', [contract['condition']])
    if not conditions:
        return finish(baseline, 'condition_candidates_unresolved')
    if any(features.get(name) is None for item in conditions
           for name in Program.from_dict(item).required_features()):
        return finish(baseline, 'condition_inputs_unavailable')
    if not budget.spend(len(conditions)):
        result['baseline']['budget_incomplete'] = True
        return finish(baseline, 'condition_compute_unresolved')
    predictions = _conflict_predictions({'coordinate_paths': contract['coordinate_paths'],
        'candidates': [{'condition': item} for item in conditions]}, profile)
    result['guard'].update(status='evaluated', features=features,
        candidate_predictions=predictions, compute_cost=len(conditions))
    if not predictions or len({item['helpful'] for item in predictions}) != 1:
        result['guard'].update(status='competing_conditions_unresolved', decision=None)
        if first['training_loss'] <= tolerance and first['feedback_loss'] <= tolerance:
            return finish(baseline, 'baseline_supported', True)
        return finish(baseline, 'competing_conditions_unresolved')
    proceed = predictions[0]['helpful']
    result['guard']['decision'] = proceed
    if first['training_loss'] <= tolerance and first['feedback_loss'] <= tolerance:
        result['guard']['ignored_because_goal_satisfied'] = proceed
        return finish(baseline, 'baseline_supported', True)
    if not proceed:
        return finish(baseline, 'continuation_declined', True)
    if parent_bound:
        prepared = prepare_parent_parameter_revision(baseline, contract['continuation'])
        result['parent_binding'] = {key: deepcopy(value) for key, value in prepared.items()
                                    if key != 'research_program'}
        if prepared['status'] != 'ready':
            return finish(baseline, 'continuation_dependency_unresolved')
        paid_initial = {'entry': baseline, 'training': rows, 'risk_quantile': risk_quantile,
            'cost_ref': deepcopy(first.get('original_cost_ref')) or {
                'operation': contract['operation'], 'stage': 'baseline',
                'compute_cost': first['compute_cost'], 'training_rows': len(rows)}}
        continuation = stage('continuation', prepared['research_program'], paid_initial)
    else:
        continuation = stage('continuation')
    if continuation is None or not result['continuation']['execution_complete']:
        return finish(baseline, 'continuation_incomplete')
    later = result['continuation']
    if (later['all_paid_loss'] < first['all_paid_loss'] - tolerance
            or first['all_paid_loss'] > tolerance >= later['all_paid_loss']):
        result['selected_stage'] = 'continuation'
        return finish(continuation, 'continuation_selected', True)
    return finish(baseline, 'baseline_retained_after_comparison', True)


def finalize_interval_candidates(entry, candidates, research, training, feedback_rows, *,
                                 feedback_experiments=None, fit_source_ref=None):
    """Attach the original fit/return roles to already evaluated interval drafts.

    Uses the row objects evaluated by ``research.alternatives`` and its paid
    squared errors; it performs no fitting or prediction. A draft with the same
    executable Program is retained because its supported interval may change.
    """
    from statistics import mean

    payload = entry.get('research_program') or {}
    training, feedback = list(training), list(feedback_rows)
    paid = [*training, *feedback]
    experiments = [feedback] if feedback_experiments is None else feedback_experiments
    source = payload.get('readout_fit_source', {})
    source_rows = source.get('fit_rows', payload.get('fit_rows', len(training)))
    for candidate in candidates:
        errors = research.last_candidate_errors[candidate['candidate'].program]
        fit_loss = mean(errors[id(row)] for row in training)
        returned_loss = mean(errors[id(row)] for row in feedback)
        all_loss = candidate['validation_loss']
        candidate.update(training_loss=fit_loss, validation_loss=returned_loss,
            all_paid_loss=all_loss, first_training_loss=fit_loss, feedback_loss=returned_loss,
            evaluation_rows=len(paid), paid_row_squared_errors=[errors[id(row)] for row in paid],
            feedback_trajectory_losses=[mean(errors[id(row)] for row in part) for part in experiments],
            selection_basis='all_acquired_evidence')
        body = candidate['research_program']
        body.update(fit_rows=source_rows, evaluation_rows=len(paid), evaluation_loss=all_loss,
            readout_fit_source={'source_ref': deepcopy(fit_source_ref if fit_source_ref is not None else
                source.get('source_ref', payload.get('source_fit_ref'))),
                'fit_rows': source_rows, 'fixed_readout': True, 'new_fits': 0})
        # Evaluation of additional rows never changes the original readout's
        # training scope, columns, or fitted coefficients.
        body['graph']['nodes'][body['output_node']]['loss'] = fit_loss
        body['transfer_partition'] = deepcopy(payload.get('transfer_partition', {}))
        body['transfer_partition']['released_groups'] = [research.parameters[candidate['parameter_index']]['symbol']]
        body['transfer_partition_search'] = deepcopy(payload.get('transfer_partition_search', {}))
        body['transfer_partition_followup'] = {
            'status': 'paid_interval_supported', 'fit_rows': len(training), 'feedback_rows': len(feedback),
            'data_role': 'exposed_adaptive_development', 'first_training_loss': fit_loss,
            'feedback_loss': returned_loss, 'all_paid_loss': all_loss,
            'fixed_readout_only': True, 'new_fits': 0, 'independent_support': False}
    return candidates


def continue_transfer_partition(entry, training, feedback_rows, budget, *, tolerance,
                                feedback_loss=None, feedback_experiments=None,
                                feedback_squared_errors=None, fit_source_ref=None):
    """Test an unresolved relation cell using a fitted answer's paid return.

    The first stage has already identified readout weights. Its returned rows
    become adaptive development evidence for the existing interval researcher;
    there is no second readout fit and no automatic acquisition. A failure here
    concerns this fixed complete response, not every possible readout/structure.
    """
    from statistics import mean
    from .parallel_prediction import predict_rows

    start = budget.spent
    payload = entry.get('research_program') or {}
    contract = payload.get('transfer_partition', {})
    second = contract.get('second_stage')
    training, feedback = list(training), list(feedback_rows)
    result = {'status': 'second_stage_not_declared', 'selected': entry,
        'candidates': [], 'execution_complete': False, 'budget_incomplete': False,
        'compute_cost': 0, 'new_fits': 0, 'new_observations': 0,
        'data_role': 'exposed_adaptive_development', 'independent_support': False,
        'fit_rows': len(training), 'feedback_rows': len(feedback),
        'fixed_readout_only': True, 'selected_new': False,
        'returned_evidence_consumed': False}

    def finish(status, local=None):
        if local is not None:
            result.update(interval_compute_cost=local.spent,
                budget_incomplete=local.last_denied_request is not None,
                denied_request=deepcopy(local.last_denied_request))
        result.update(status=status, compute_cost=budget.spent - start)
        return result

    if not second:
        return result
    if second['operation'] != 'resolve_relation_intervals':
        raise ValueError('Unsupported transfer followup operation')
    if not training or not feedback:
        return finish('paid_return_required')
    experiments = [feedback]
    if feedback_experiments is not None:
        experiments, offset = [], 0
        for supplied in feedback_experiments:
            supplied = list(supplied)
            actual = feedback[offset:offset + len(supplied)]
            if not supplied or actual != supplied:
                raise ValueError('Feedback experiments must partition the paid feedback in order')
            # Bind separately restored but equal records back to the paid row
            # objects. Their own reset histories remain unchanged.
            experiments.append(actual)
            offset += len(supplied)
        if offset != len(feedback):
            raise ValueError('Feedback experiments must cover all paid feedback')
    result['feedback_trajectory_rows'] = [len(part) for part in experiments]
    trained = entry.get('training_loss')
    if trained is None or not isfinite(trained):
        return finish('fitted_training_result_required')
    if trained > tolerance:
        return finish('first_stage_training_unexplained')
    errors = None
    if feedback_squared_errors is not None:
        errors = list(feedback_squared_errors)
        if (len(errors) != len(feedback)
                or any(not isfinite(value) or value < 0. for value in errors)):
            raise ValueError('Paid feedback errors must cover each feedback row')
        feedback_loss = mean(errors)
    if feedback_loss is None or (feedback_experiments is not None and
            feedback_loss <= tolerance and errors is None):
        if not budget.spend(len(feedback)):
            result['budget_incomplete'] = True
            return finish('compute_unresolved')
        values = predict_rows(entry['candidate'].program, feedback)
        errors = [(value - row.target) ** 2 for value, row in zip(values, feedback)]
        feedback_loss = mean(errors)
    if not isfinite(feedback_loss):
        return finish('finite_return_loss_required')
    result['prior_feedback_loss'] = feedback_loss
    result.update(first_training_loss=trained, feedback_loss=feedback_loss,
        all_paid_loss=(len(training) * trained + len(feedback) * feedback_loss) / (len(training) + len(feedback)),
        returned_evidence_consumed=True)
    trajectory_losses = None
    if errors is not None:
        trajectory_losses, offset = [], 0
        for part in experiments:
            trajectory_losses.append(mean(errors[offset:offset + len(part)]))
            offset += len(part)
        result.update(prior_feedback_squared_errors=errors,
            prior_feedback_trajectory_losses=trajectory_losses)
    if feedback_loss <= tolerance and (trajectory_losses is None or
            all(value <= tolerance for value in trajectory_losses)):
        result.update(execution_complete=True, full_comparison=True)
        return finish('paid_return_already_supported')
    view = transfer_partition_view(payload)
    groups = contract.get('released_groups', [])
    if view['status'] != 'ready' or len(groups) != 1:
        return finish('one_explicit_relation_group_required')
    saved = _saved_binding(payload)
    index = next((i for i, item in enumerate(saved['parameters']) if item['symbol'] == groups[0]), None)
    if index is None:
        return finish('released_group_binding_required')
    local = SearchBudget(min(budget.remaining, max(0, int(second.get('compute_limit', 15000)))))
    try:
        # Explicit saved groups must not be merged by the legacy same-source
        # inference. The scientific prior itself remains in the original payload.
        current_payload = deepcopy(payload)
        for item in _saved_binding(current_payload)['parameters']:
            item['source_concept_id'] = None
        research = RelationIntervalResearch(entry['candidate'].program, [], local,
            research_program=current_payload)
        if local.last_denied_request is not None:
            return finish('compute_unresolved', local)
        if len(research.parameters) != len(saved['parameters']):
            return finish('parameter_binding_unresolved', local)
        for actual, original in zip(research.parameters, saved['parameters']):
            actual['source_concept_id'] = original.get('source_concept_id')
        paid = [*training, *feedback]
        # Complete return trajectories cannot be diluted by the old fit rows.
        # Precision is determined by the target's residual contract, not the
        # source's hard-coded 1e-6 geometric stopping width.
        candidates = research.alternatives(paid, local, tolerance,
            experiments=experiments, interval_precision=0., parameter_index=index)
        result.update(research_status=research.status, evidence_updates=deepcopy(research.evidence_updates),
            intervals=deepcopy(research.intervals))
        if local.last_denied_request is not None:
            result['candidates'] = candidates
            result['partial_evaluations'] = [{'program': program.to_dict(),
                'first_training_loss': mean(errors[id(row)] for row in training),
                'feedback_loss': mean(errors[id(row)] for row in feedback),
                'all_paid_loss': mean(errors[id(row)] for row in paid),
                'evaluation_rows': len(paid),
                'paid_row_squared_errors': [errors[id(row)] for row in paid]}
                for program, errors in research.last_candidate_errors.items()]
            return finish('compute_unresolved', local)
        if not candidates:
            result['execution_complete'] = research.status == 'refuted'
            result['full_comparison'] = research.status == 'refuted'
            return finish('fixed_readout_interval_unexplained' if research.status == 'refuted'
                          else 'interval_unresolved', local)
        finalize_interval_candidates(entry, candidates, research, training, feedback,
            feedback_experiments=experiments, fit_source_ref=fit_source_ref)
        chosen = min(candidates, key=lambda item: (
            0. if item['all_paid_loss'] <= tolerance else item['all_paid_loss'], item['candidate'].program.size))
        result.update(selected=chosen, candidates=candidates, execution_complete=True,
            full_comparison=True, first_training_loss=chosen['training_loss'],
            feedback_loss=chosen['validation_loss'], all_paid_loss=chosen['all_paid_loss'],
            selected_new=chosen['candidate'].program != entry['candidate'].program)
        return finish('paid_interval_supported', local)
    finally:
        # The activity deadline may interrupt construction or evaluation.
        # Charge the actual nested work once on success, return, or exception.
        budget.spend(local.spent)
        result['compute_cost'] = budget.spent - start


def _complete_fit(fit, declared_terms):
    # _Fitter removes constant and duplicate columns. Restore the declared
    # basis with zero coefficients; give a duplicate's coefficient once only.
    weights = dict(zip(fit['fit_terms'], fit['coefficients'][1:]))
    coefficients = [fit['coefficients'][0]]
    for term in declared_terms:
        coefficients.append(weights.pop(term, 0.))
    return {**fit, 'fit_terms': tuple(declared_terms), 'coefficients': coefficients}


def _replay_discrete_partition(payload, rows, budget, *, risk_quantile, max_attempts,
                               max_candidates, paid_fits=()):
    """Run an evidence-selected, bounded binding or local-structure family."""
    from .continuous_state import _sum_terms
    from .research_programs import _Fitter, _entry, _rank

    start, denied = budget.spent, budget.denied_requests
    contract = payload['transfer_partition']
    alternatives = contract['alternatives'][:min(max_attempts, contract['max_attempts'], 4)]
    if not alternatives or max_candidates <= 0:
        return None
    fitter = _Fitter(rows, budget, risk_quantile)
    best, outcomes = None, []
    for index, alternative in enumerate(alternatives):
        terms = [Program.from_dict(term) for term in alternative['fit_terms']]
        fit, reuse = None, {'status': 'not_supplied', 'reused': False}
        for record in paid_fits:
            fit, reuse = _paid_fit(record, rows, risk_quantile, tuple(terms))
            if fit is not None:
                reuse.update(status='reused_paid_alternative_fit', reused=True,
                    scope='one_alternative_fit_not_completed_partition_search')
                break
        if fit is None:
            fit = fitter.fit(terms)
        if budget.denied_requests != denied:
            break
        outcomes.append({'index': index, 'training_loss': fit['loss'] if fit else None,
                         'status': 'fitted' if fit else 'fit_unavailable',
                         'paid_fit': reuse})
        if fit is not None:
            if not reuse['reused']:
                fit = _complete_fit(fit, terms)
            if best is None or _rank(fit) < _rank(best[0]):
                best = fit, index, alternative
    if best is None:
        return None
    fit, index, alternative = best
    steps = []
    saved = _saved_binding(payload)
    if saved and alternative.get('parameter_terms'):
        symbolic = [Program.from_dict(term) for term in alternative['parameter_terms']]
        template = _sum_terms(symbolic, fit['coefficients'][1:], fit['coefficients'][0])
        parameters = deepcopy(saved['parameters'])
        source_terms = [Program.from_dict(term) for term in saved['parameter_terms']]
        for parameter in parameters:
            symbol = parameter['symbol']
            before = [node for term in source_terms for node in parameter_relations(term, symbol)]
            after = [node for term in symbolic for node in parameter_relations(term, symbol)]
            changed = [new for old, new in zip(before, after)
                if old.op == new.op and old.children == new.children and old.name != new.name]
            if not changed:
                continue
            # A radius supported for another orientation is initialization,
            # not a current interval exclusion or a reusable supported kernel.
            prior = parameter.setdefault('prior_radius_intervals', [])
            for interval in parameter.get('radius_intervals', []):
                if interval not in prior:
                    prior.append(interval)
            parameter.update(relation=changed[0].name or 'symmetric', radius_intervals=[])
        steps.append({'operation': 'resolve_relation_intervals',
            'question': 'Preserve relation bindings through the local computation revision',
            'program': fit['program'], 'loss': fit['loss'],
            'parameter_template': template.to_dict(),
            'parameter_terms': deepcopy(alternative['parameter_terms']),
            'parameters': parameters, 'evidence_contexts': []})
    steps.append({'operation': 'refit_research_program',
        'question': 'Compare the evidence-selected local revisions on current paid rows',
        'program': fit['program'], 'loss': fit['loss'], 'execution_scope': 'transfer_partition'})
    summary = {'status': ('compute_unresolved' if budget.denied_requests != denied
                          else 'completed_bounded_search'),
        'completed': len(outcomes) == len(alternatives), 'attempts': len(outcomes),
        'planned_attempts': len(alternatives),
        'remaining': len(contract['alternatives']) - len(outcomes),
        'finite_domain_exhausted': len(outcomes) == len(contract['alternatives']),
        'selected_alternative': index, 'alternative_outcomes': outcomes,
        'failed_fits': sum(item['status'] != 'fitted' for item in outcomes),
        'reused_fits': sum(item['paid_fit']['reused'] for item in outcomes),
        'new_fits': sum(not item['paid_fit']['reused'] for item in outcomes),
        'compute_cost': budget.spent - start, 'independent_support': False,
        'search_domain': contract['search_domain'],
        'search_policy': 'evidence_selected_existing_local_vocabulary'}
    entry = _entry(fit, steps, len(rows), budget.spent - start)
    entry['research_program'].update(transfer_partition=deepcopy(contract),
        transfer_partition_search=summary, reused_procedure=True)
    for name in ('input_template', 'source_schema', 'input_features', 'input_source_ids'):
        if name in payload:
            entry['research_program'][name] = deepcopy(payload[name])
    entry['transfer_partition_search'] = summary
    return entry


def prepare_parent_parameter_revision(parent, source_method, *, source_method_ref=None,
                                      parent_ref=None):
    """Instantiate an existing Pi1 operation on the current complete parent.

    The old frozen method remains unchanged. This is a new target-specific
    research body, not evidence that the source method or a switching condition
    has transferred successfully.
    """
    source = source_method.get('transfer_partition', {})
    if source.get('kind') != 'Pi1' or source_method.get('staged_research'):
        return {'status': 'explicit_pi1_operator_required'}
    body = parent.get('research_program')
    if not body:
        return {'status': 'saved_parent_research_required'}
    fit = research_program_fit_contract(body)
    if fit['program'] != parent['candidate'].program:
        return {'status': 'parent_output_mismatch'}
    view = transfer_partition_view(body)
    if view['status'] != 'ready' or len(view['groups']) != 1:
        return {'status': 'one_explicit_parent_group_required'}
    current = deepcopy(body)
    for name in ('transfer_partition_search', 'transfer_partition_followup', 'staged_research'):
        current.pop(name, None)
    contract = {key: deepcopy(source[key]) for key in ('kind', 'max_attempts', 'max_candidates')}
    if source.get('second_stage'):
        contract['second_stage'] = deepcopy(source['second_stage'])
    contract.update(released_groups=[view['groups'][0]['symbol']],
        search_domain='current_paid_geometry', source_interval_role='initialization_only',
        status='target_instance_requires_evaluation')
    current['transfer_partition'] = contract
    binding = {'operation_reused': 'Pi1', 'new_parent_binding': True,
        'source_method_ref': deepcopy(source_method_ref), 'parent_ref': deepcopy(parent_ref),
        'source_support_inherited': False, 'independent_support': False}
    current['parent_parameter_revision'] = binding
    return {'status': 'ready', 'research_program': current, **binding}


def _paid_fit(record, rows, risk_quantile, terms):
    """Read an exact paid solve of these columns, not a cached research return."""
    status = {'status': 'not_supplied', 'reused': False}
    if record is None:
        return None, status
    status.update(status='input_mismatch', cost_ref=deepcopy(record.get('cost_ref')))
    if (not record.get('cost_ref') or record.get('training') != rows
            or record.get('risk_quantile') != risk_quantile):
        return None, status
    entry = record.get('entry') or {}
    body = entry.get('research_program') or {}
    if not body or body.get('fit_rows') != len(rows):
        return None, {**status, 'status': 'paid_fit_contract_mismatch'}
    paid = research_program_fit_contract(body)
    if paid['program'] != entry['candidate'].program or paid['fit_terms'] != terms:
        return None, {**status, 'status': 'paid_fit_contract_mismatch'}
    measured = entry.get('training_loss')
    if measured is None or not isfinite(measured) or measured < 0. or measured != paid['loss']:
        return None, {**status, 'status': 'paid_fit_loss_mismatch'}
    return {**paid, 'terms': paid['fit_terms']}, {**status, 'status': 'eligible_paid_fit',
        'fit_rows': len(rows), 'training_loss': measured}


def _paid_initial_fit(record, payload, rows, risk_quantile):
    """Reuse one exact paid fit; never interpret it as a cached Pi1 search."""
    current = research_program_fit_contract(payload)
    paid, status = _paid_fit(record, rows, risk_quantile, current['fit_terms'])
    if paid is None:
        return None, status
    paid_binding, binding = _saved_binding(record['entry']['research_program']), _saved_binding(payload)
    if (paid['program'] != current['program'] or not paid_binding or not binding
            or [(item['symbol'], item['radius']) for item in paid_binding['parameters']] !=
               [(item['symbol'], item['radius']) for item in binding['parameters']]):
        return None, {**status, 'status': 'paid_fit_contract_mismatch'}
    return paid, {**status, 'status': 'eligible_initializer_fit',
        'scope': 'one_initializer_only_not_completed_partition_search'}


def propose_parent_parameter_revisions(spec, training, budget, *, parent, method,
                                       feedback_rows=(), tolerance=1e-7,
                                       max_candidates=4, paid_initial=None):
    """Run the existing joint parameter/readout search on a new parent binding."""
    started, denied, fits = budget.spent, budget.denied_requests, budget.completed_fits
    report = {'status': 'parent_binding_required', 'execution_complete': False,
        'new_observations': 0, 'independent_support': False, 'new_parent_binding': True}
    result = {'entries': [], 'research': report}
    if (method.get('transfer_partition', {}).get('kind') != 'Pi1'
            or research_program_fit_contract(method)['program'] != parent['candidate'].program):
        report.update(compute_cost=0, new_fits=0, budget_incomplete=False)
        return result
    entry = replay_transfer_partition(method, training, budget,
        risk_quantile=spec.risk_quantile, max_candidates=max_candidates,
        feedback_rows=feedback_rows, tolerance=tolerance, paid_initial=paid_initial)
    search = (entry or {}).get('transfer_partition_search', {})
    followup = (entry or {}).get('transfer_partition_followup', {})
    complete = bool(entry and search.get('completed') and not search.get('failed_fits')
        and budget.denied_requests == denied)
    # A fixed-readout continuation has its own training-support prerequisite.
    # Its inapplicability does not erase the completed joint first search.
    followup_declared = bool(method['transfer_partition'].get('second_stage'))
    followup_complete = not followup_declared or bool(followup.get('execution_complete'))
    report.update(compute_cost=budget.spent - started, new_fits=budget.completed_fits - fits,
        budget_incomplete=budget.denied_requests != denied or bool(followup.get('budget_incomplete')),
        bounded_first_search_complete=complete, search=deepcopy(search), followup=deepcopy(followup),
        execution_complete=complete and followup_complete,
        source_method_ref=deepcopy(method.get('parent_parameter_revision', {}).get('source_method_ref')),
        parent_ref=deepcopy(method.get('parent_parameter_revision', {}).get('parent_ref')))
    report['status'] = ('completed_parameter_research' if report['execution_complete'] else
        'compute_unresolved' if report['budget_incomplete'] else
        'parameter_followup_not_applicable' if complete and followup.get('status') ==
            'first_stage_training_unexplained' else 'parameter_research_unresolved')
    if entry is not None:
        entry['research_program']['parent_parameter_revision'] = deepcopy(
            method.get('parent_parameter_revision', {}))
        result['entries'].append(entry)
    return result


def replay_transfer_partition(payload, training, budget, *, risk_quantile=.5,
                              feature_mapping=None, max_attempts=48, max_candidates=12,
                              feedback_rows=(), feedback_loss=None, tolerance=1e-7,
                              paid_initial=None, paid_fits=()):
    """Execute one released group with joint readout fits; return the best draft."""
    from .research_programs import _Fitter, _entry, _rank
    if feature_mapping:
        from .research_drafts import _rebind_step
        payload = _rebind_step(payload, feature_mapping)
    rows = list(training)
    if len(rows) < 6:
        return None
    if payload['transfer_partition']['kind'] in {'Pi2', 'Pi3'}:
        return _replay_discrete_partition(payload, rows, budget,
            risk_quantile=risk_quantile, max_attempts=max_attempts, max_candidates=max_candidates,
            paid_fits=paid_fits)
    start = budget.spent
    original = research_program_fit_contract(payload)
    contract = payload['transfer_partition']
    attempts_limit = min(max_attempts, contract['max_attempts'], 48)
    keep = min(max_candidates, contract['max_candidates'], 12)
    if attempts_limit <= 0 or keep <= 0:
        return None
    templates, values, parameter, research = list(original['fit_terms']), {}, None, None
    all_values, domain = [None], None
    if contract['kind'] == 'Pi1':
        view = transfer_partition_view(payload)
        if view['status'] != 'ready' or len(contract['released_groups']) != 1:
            return None
        symbol = contract['released_groups'][0]
        if symbol not in {group['symbol'] for group in view['groups']}:
            return None
        denied = budget.denied_requests
        # These explicit symbols define sharing for the target hypothesis.
        # Do not invoke the legacy prior-by-value group-merging heuristic.
        research_payload = deepcopy(payload)
        saved_parameters = _saved_binding(payload)['parameters']
        for item in _saved_binding(research_payload)['parameters']:
            item['source_concept_id'] = None
        research = RelationIntervalResearch(original['program'], [], budget, research_program=research_payload)
        if budget.denied_requests != denied or len(research.parameters) != len(saved_parameters):
            return None
        if research._parameter_terms is None:
            return None
        rename = {item['symbol']: Program('arg', name=old['symbol'])
                  for item, old in zip(research.parameters, saved_parameters)}
        templates = [_bind(term, rename) for term in research._parameter_terms]
        research.parameters = [{**item, 'symbol': old['symbol'],
                                'source_concept_id': old.get('source_concept_id')}
                               for item, old in zip(research.parameters, saved_parameters)]
        values = {item['symbol']: item['radius'] for item in research.parameters}
        parameter = next(item for item in research.parameters if item['symbol'] == symbol)
        relations = list(dict.fromkeys(node for term in templates for node in parameter_relations(term, symbol)))
        frames = _geometry_frames(rows, templates, relations)
        distances = _distances(relations, frames, budget)
        if budget.denied_requests != denied:
            return None
        upper = max(distances, default=0.) + max(1., max(distances, default=0.)) * 1e-6
        domain = [0., upper]
        cuts = sorted(set([0., *distances, upper]))
        cells = [(left + right) / 2 for left, right in zip(cuts, cuts[1:])]
        initials = [parameter['radius']]
        all_values = list(dict.fromkeys([*initials, *cells, *distances]))
        coarse_count = min(16, max(1, attempts_limit // 2), len(cells))
        coarse = sorted({cells[round(index * (len(cells) - 1) / max(1, coarse_count - 1))]
                         for index in range(coarse_count)})
        queue = list(initials)
        # The existing structural researcher covers the range before refining
        # neighborhoods of the lowest measured profile loss. Apply that same
        # finite strategy to this single unchanged response family.
        while coarse:
            value = max(coarse, key=lambda item: min(abs(item - old) for old in queue))
            coarse.remove(value)
            if value not in queue:
                queue.append(value)
    elif contract['kind'] == 'Pi0':
        queue = [None]
    else:
        raise ValueError('Unsupported transfer partition')
    initial_fit, initial_reuse = _paid_initial_fit(paid_initial, payload, rows, risk_quantile)
    fitter = _Fitter(rows, budget, risk_quantile)
    retained, attempts, failures, evaluated = [], 0, 0, []
    planned = min(attempts_limit, len(all_values))
    denied = budget.denied_requests
    for _ in range(planned):
        tested = [item[0] for item in evaluated]
        if queue:
            value = queue.pop(0)
        else:
            remaining = [item for item in all_values if item not in tested]
            value = None
            for center, _loss in sorted(evaluated, key=lambda item: item[1]):
                lower = max((item for item in tested if item < center), default=domain[0])
                upper = min((item for item in tested if item > center), default=domain[1])
                neighborhoods = [[item for item in remaining if lower < item < center],
                                 [item for item in remaining if center < item < upper]]
                neighborhood = max(neighborhoods, key=len)
                if neighborhood:
                    neighborhood.sort()
                    value = neighborhood[(len(neighborhood) - 1) // 2]
                    break
            if value is None:
                value = max(remaining, key=lambda item: min(abs(item - old) for old in tested))
        current = dict(values)
        if parameter is not None:
            current[parameter['symbol']] = value
        bindings = {key: Program('const', value=number) for key, number in current.items()}
        terms = [_bind(term, bindings).simplified() for term in templates]
        if (attempts == 0 and initial_fit is not None
                and tuple(terms) == initial_fit['fit_terms']):
            fit = initial_fit
            initial_reuse.update(status='same_initializer_fit', reused=True)
        else:
            if attempts == 0 and initial_fit is not None:
                initial_reuse['status'] = 'initializer_columns_changed'
            fit = fitter.fit(terms)
        if budget.denied_requests != denied:
            break
        attempts += 1
        evaluated.append((value, fit['loss'] if fit is not None else float('inf')))
        if fit is None:
            failures += 1
            continue
        fit = _complete_fit(fit, terms)
        retained.append((fit, current, terms))
        retained.sort(key=lambda item: _rank(item[0]))
        retained = retained[:keep]
    if not retained:
        return None
    fit, chosen, terms = retained[0]
    summary = {'status': 'compute_unresolved' if budget.denied_requests != denied else 'completed_bounded_search',
        'completed': attempts == planned, 'attempts': attempts,
        'planned_attempts': planned, 'remaining': len(all_values) - attempts,
        'finite_domain_exhausted': attempts == len(all_values), 'failed_fits': failures,
        'search_domain': domain, 'search_policy': 'coarse_then_profile_refinement',
        'source_interval_role': 'initialization_only',
        'compute_cost': budget.spent - start, 'independent_support': False,
        'paid_initial': initial_reuse, 'reused_initial_fits': int(initial_reuse['reused'])}
    steps = []
    saved = _saved_binding(payload)
    if parameter is not None or (saved and saved.get('parameter_terms')):
        from .continuous_state import _sum_terms
        if parameter is not None:
            params = [{**deepcopy(item), 'radius': chosen[item['symbol']]} for item in research.parameters]
            selected_parameter = next(item for item in params if item['symbol'] == parameter['symbol'])
            radius = chosen[parameter['symbol']]
            cell = [max((edge for edge in cuts if edge < radius), default=domain[0]),
                    min((edge for edge in cuts if edge >= radius), default=domain[1])]
            if cell[0] == cell[1]:
                cell = list(domain)
            selected_parameter.update(source_radius_intervals=deepcopy(parameter['radius_intervals']),
                prior_radius_intervals=[list(domain)], radius_intervals=[cell],
                interval_meaning='current_geometry_cell_not_confidence_interval')
        else:
            params = deepcopy(saved['parameters'])
            templates = [Program.from_dict(term) for term in saved['parameter_terms']]
        template = _sum_terms(templates, fit['coefficients'][1:], fit['coefficients'][0])
        steps.append({'operation': 'resolve_relation_intervals',
            'question': 'Reidentify one bound relation group and refit the complete readout on current evidence',
            'program': fit['program'], 'loss': fit['loss'], 'parameter_template': template.to_dict(),
            'parameter_terms': [term.to_dict() for term in templates], 'parameters': params,
            'parameter_index': next((index for index, item in enumerate(params)
                                     if parameter is not None and item['symbol'] == parameter['symbol']), 0),
            'evidence_contexts': [], 'source_intervals_are_initialization': True})
    steps.append({'operation': 'refit_research_program', 'question': 'Refit all saved coordinates on current evidence',
                  'program': fit['program'], 'loss': fit['loss'], 'execution_scope': 'transfer_partition'})
    entry = _entry(fit, steps, len(rows), budget.spent - start)
    entry['research_program'].update(transfer_partition=deepcopy(contract),
        transfer_partition_search=summary, reused_procedure=True)
    entry['transfer_partition_search'] = summary
    if contract.get('second_stage'):
        followup = continue_transfer_partition(entry, rows, feedback_rows, budget,
            tolerance=tolerance, feedback_loss=feedback_loss)
        entry = followup['selected']
        entry['transfer_partition_search'] = deepcopy(summary)
        returned = {key: deepcopy(value) for key, value in followup.items()
                    if key not in {'selected', 'candidates'}}
        entry['transfer_partition_followup'] = returned
        entry['research_program']['transfer_partition_followup'] = deepcopy(returned)
    return entry
