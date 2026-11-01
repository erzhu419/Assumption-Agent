"""Revise a relation inside retained compositions without replacing their calls."""

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

from .language import Definition, EvaluationContext, MissingHistory, Program
from .relation_conditions import TRUE, atomic_conditions, compound_conditions
from .relation_language import EDGE_ARGUMENTS, instantiate_predicate, relation_objects


ZERO = Program("const", value=0.)


def _parent_relation_plan(parent, max_candidates):
    """Locate actual saved parameter groups; equal numbers do not imply sharing."""
    from .structural_research_program import research_program_fit_contract
    from .transfer_partitions import _saved_binding, transfer_partition_view

    payload = parent['research_program']
    fit = research_program_fit_contract(payload)
    if fit['program'] != parent['candidate'].program:
        raise ValueError('Parent research output does not match its retained candidate')
    view = transfer_partition_view(payload)
    if view['status'] != 'ready':
        return {'status': view['status'], 'variants': []}
    saved = _saved_binding(payload)
    symbolic = tuple(Program.from_dict(term) for term in saved['parameter_terms'])
    symbols = {item['symbol'] for item in saved['parameters']}
    groups = {}

    def visit(node):
        if (node.op in {'neighbor_sum', 'relation_sum'}
                and node.children[0].op == 'arg' and node.children[0].name in symbols):
            key = (node.children[0].name, _geometry(node))
            group = groups.setdefault(key, [])
            if node not in group:
                group.append(node)
        for child in node.children:
            visit(child)

    for term in symbolic:
        visit(term)
    variants = []
    for (symbol, geometry), contexts in groups.items():
        for direction in ('symmetric', 'right', 'left'):
            if direction != geometry[2]:
                variants.append({'symbol': symbol, 'before_direction': geometry[2],
                    'direction': direction, 'contexts': contexts,
                    'name': f'{symbol}:{geometry[2]}->{direction}:{len(variants)}'})
    total = len(variants)
    variants = variants[:max(0, max_candidates)]
    # Charged graph work: read parent/columns, check their parameter binding,
    # and traverse the complete symbolic basis for each shared replacement.
    preparation = (fit['program'].size + 3 * sum(term.size for term in fit['fit_terms'])
        + (2 + 3 * len(variants)) * sum(term.size for term in symbolic)
        + len(saved['parameters']))
    return {'status': 'ready' if variants else 'no_local_alternatives',
        'fit': fit, 'saved': saved, 'symbolic': symbolic, 'variants': variants,
        'available_variants': total, 'preparation_compute': preparation}


def quote_parent_relation_revisions(parent, training_rows, *, max_candidates=4,
                                     comparison_rows=0):
    """Quote a fixed-radius local form question, not a complete family search."""
    plan = _parent_relation_plan(parent, max_candidates)
    count = len(plan['variants'])
    columns = len(plan['fit']['fit_terms']) if 'fit' in plan else 0
    fitting = count * (columns + 2) * training_rows
    preparation = plan.get('preparation_compute', 0)
    return {'status': plan['status'], 'planned_attempts': count,
        'variant_names': [item['name'] for item in plan['variants']],
        'fit_columns': columns, 'preparation_compute': preparation,
        'fit_compute_upper': fitting, 'comparison_compute': count * comparison_rows,
        'required_compute': preparation + fitting,
        'search_domain': 'saved_relation_direction_at_current_radius'}


def propose_parent_relation_revisions(spec, training, budget, parent, *, max_candidates=4):
    """Refit every saved column for the other existing relation directions.

    This finite local question keeps the current radii fixed. It neither searches
    a complete relation family nor chooses a direction from task configuration.
    The paid parent is retained by the caller and is never fitted here.
    """
    from .continuous_state import _sum_terms
    from .mechanism_revision import _replace_shared
    from .relation_uncertainty import _bind
    from .research_programs import _Fitter, _entry
    from .transfer_partitions import _complete_fit

    start, denied, fits = budget.spent, budget.denied_requests, budget.completed_fits
    plan = _parent_relation_plan(parent, max_candidates)
    result = {'entries': [], 'research': {'status': plan['status'], 'attempts': [],
        'planned_attempts': len(plan['variants']), 'execution_complete': False,
        'search_domain': 'saved_relation_direction_at_current_radius',
        'independent_support': False, 'parent_refitted': False, 'new_observations': 0}}
    report = result['research']

    def finish(status):
        report.update(status=status, compute_cost=budget.spent - start,
            new_fits=budget.completed_fits - fits,
            budget_incomplete=budget.denied_requests != denied)
        return result

    if plan['status'] != 'ready':
        return finish(plan['status'])
    if len(training) < 6:
        return finish('insufficient_training_rows')
    if not budget.spend(plan['preparation_compute']):
        return finish('compute_unresolved')
    report.update(preparation_compute=plan['preparation_compute'],
        available_variants=plan['available_variants'], fit_columns=len(plan['symbolic']),
        variant_names=[item['name'] for item in plan['variants']])
    values = {item['symbol']: Program('const', value=item['radius'])
              for item in plan['saved']['parameters']}
    fitter = _Fitter(training, budget, spec.risk_quantile)
    for index, variant in enumerate(plan['variants']):
        symbolic = list(plan['symbolic'])
        for context in variant['contexts']:
            replacement = replace(context,
                name=None if variant['direction'] == 'symmetric' else variant['direction'])
            symbolic = [_replace_shared(term, context, replacement)[0] for term in symbolic]
        terms = [_bind(term, values) for term in symbolic]
        attempt = {key: variant[key] for key in ('name', 'symbol', 'before_direction', 'direction')}
        before = budget.spent
        try:
            fit = fitter.fit(terms)
        except (ArithmeticError, ValueError) as error:
            fit = None
            attempt['failure'] = f'{type(error).__name__}: {error}'
        attempt.update(status=('compute_unresolved' if budget.denied_requests != denied
            else 'fitted' if fit is not None else 'fit_unavailable'),
            compute_cost=budget.spent - before, fit_complete=fit is not None,
            training_loss=fit['loss'] if fit is not None else None)
        report['attempts'].append(attempt)
        if budget.denied_requests != denied:
            break
        if fit is None:
            continue
        fit = _complete_fit(fit, terms)
        parameters = deepcopy(plan['saved']['parameters'])
        for parameter in parameters:
            if parameter['symbol'] == variant['symbol']:
                prior = parameter.setdefault('prior_radius_intervals', [])
                for interval in parameter.get('radius_intervals', []):
                    if interval not in prior:
                        prior.append(interval)
                parameter['radius_intervals'] = []
                parameter['relation'] = variant['direction']
        binding = {'operation': 'resolve_relation_intervals',
            'question': 'Retain parameter bindings after changing the relation direction',
            'program': fit['program'], 'loss': fit['loss'],
            'parameter_template': _sum_terms(symbolic, fit['coefficients'][1:],
                fit['coefficients'][0]).to_dict(),
            'parameter_terms': [term.to_dict() for term in symbolic],
            'parameters': parameters, 'evidence_contexts': [],
            'prior_binding': {'parent_candidate_id': parent['candidate'].candidate_id,
                'selection_step_index': parent['research_program']['selection_steps'].index(plan['saved'])},
            'binding_scope': 'new_direction_same_symbol_current_radius_initialization'}
        step = {'operation': 'refit_parent_relation_form',
            'question': 'Fit all saved response columns for this competing relation direction',
            'program': fit['program'], 'loss': fit['loss'],
            **{key: variant[key] for key in ('symbol', 'before_direction', 'direction')}}
        entry = _entry(fit, [binding, step], len(training), budget.spent - start, index)
        for name in ('input_template', 'source_schema', 'input_features', 'input_source_ids'):
            if name in parent['research_program']:
                entry['research_program'][name] = deepcopy(parent['research_program'][name])
        entry['parent_relation_revision'] = deepcopy(attempt)
        result['entries'].append(entry)
    report['attempts_complete'] = (len(report['attempts']) == len(plan['variants'])
                                   and budget.denied_requests == denied)
    report['execution_complete'] = (report['attempts_complete']
                                    and all(item['fit_complete'] for item in report['attempts']))
    report['finite_domain_exhausted'] = (report['execution_complete']
                                        and len(plan['variants']) == plan['available_variants'])
    return finish('completed_local_question' if report['execution_complete'] else
        'compute_unresolved' if budget.denied_requests != denied else 'fit_unresolved')


def _has_relation(program, budget, definitions):
    if not budget.spend(1):
        return False
    if program.op in {"neighbor_sum", "relation_sum"}:
        return True
    if program.op == "call":
        key = program.definition
        if key not in definitions:
            definitions[key] = _has_relation(program.definition.body, budget, definitions)
        if definitions[key]:
            return True
    return any(_has_relation(child, budget, definitions) for child in program.children)


def _substitute(program, bindings):
    if program.op == "arg":
        return bindings.get(program.name, program)
    return replace(program, children=tuple(_substitute(child,
        {key: value for key, value in bindings.items() if key not in EDGE_ARGUMENTS}
        if program.op == "relation_sum" and index == 1 else bindings)
        for index, child in enumerate(program.children)))


def _expose(program, budget):
    """Open relation-containing definitions; independent calls remain calls."""
    if not budget.spend(1):
        return None
    if program.op == "call":
        body = _substitute(program.definition.body, dict(zip(program.definition.arguments, program.children)))
        opened = _expose(body, budget)
        if opened is None:
            return None
        return opened if _relations(opened) else program
    children = tuple(_expose(child, budget) for child in program.children)
    return None if any(child is None for child in children) else replace(program, children=children)


def _relations(program):
    if program.op == "call":
        return []
    result = [program] if program.op in {"neighbor_sum", "relation_sum"} and program.children[0].op == "const" else []
    return list(dict.fromkeys([*result, *(node for child in program.children for node in _relations(child))]))


def _geometry(relation):
    return (relation.children[0], tuple(item[1] for item in relation_objects(relation)),
            relation.name or "symmetric", relation.children[1] if relation.op == "relation_sum" else TRUE)


def _replace_relation(program, geometry, radius, kind, predicate=TRUE):
    if program.op == "call":
        return program
    if program.op in {"neighbor_sum", "relation_sum"} and _geometry(program) == geometry:
        objects = tuple(part for item in relation_objects(program) for part in item)
        return Program("neighbor_sum" if predicate == TRUE else "relation_sum",
            name=None if kind == "symmetric" else kind,
            children=(Program("const", value=radius), *((predicate,) if predicate != TRUE else ()), *objects))
    return replace(program, children=tuple(_replace_relation(child, geometry, radius, kind, predicate)
                                           for child in program.children))


def _replacement_call(original, body, serial):
    """Keep calibrated input slots while saving a distinct revised definition."""
    def calls(node):
        if node.op == "call":
            yield node
        else:
            for child in node.children:
                yield from calls(child)

    anchor = next(calls(original), None)
    leaves = list(anchor.children) if anchor else []
    arguments = list(anchor.definition.arguments) if anchor else []

    def parameterize(node):
        # A measured input may be an entire calibrated expression. Match it
        # before descending, so its existing semantic input contract survives.
        if node in leaves:
            return Program("arg", name=arguments[leaves.index(node)])
        if node.op in {"feature", "lag", "available"}:
            name = f"replacement_input_{len(leaves)}"
            while name in arguments:
                name += "_new"
            leaves.append(node)
            arguments.append(name)
            return Program("arg", name=name)
        return replace(node, children=tuple(parameterize(child) for child in node.children))

    expression = parameterize(body)
    name = f"{anchor.name if anchor else 'relation'}:relation_replacement:{serial}"
    source = (anchor.definition.source_concept_id or anchor.name) if anchor else None
    definition = Definition(name, tuple(arguments), expression, source)
    return Program("call", name=name, definition=definition, children=tuple(leaves))


def _difference(program, geometry, radius, relation_type=None):
    """Carry a changed relation through its actual enclosing computation."""
    if program.op == "call":
        return ZERO
    if program.op in {"neighbor_sum", "relation_sum"} and _geometry(program) == geometry:
        kind = relation_type or geometry[2]
        revised = replace(program, name=None if kind == "symmetric" else kind,
                          children=(Program("const", value=radius), *program.children[1:]))
        return Program("sub", children=(revised, program))
    changes = tuple(_difference(child, geometry, radius, relation_type) for child in program.children)
    if all(change == ZERO for change in changes):
        return ZERO
    if program.op in {"add", "sub", "object_sum", "window_sum", "event_delta", "accumulate"}:
        return replace(program, children=changes).simplified()
    if program.op == "mul":
        a, b = program.children
        da, db = changes
        return Program("add", children=(Program("add", children=(
            Program("mul", children=(da, b)), Program("mul", children=(a, db)))),
            Program("mul", children=(da, db)))).simplified()
    revised = replace(program, children=tuple(Program("add", children=(child, change)).simplified()
                                              for child, change in zip(program.children, changes)))
    return Program("sub", children=(revised, program)).simplified()


def _distances(relations, rows, budget):
    distances = set()
    for relation in relations:
        objects = relation_objects(relation)
        for row in rows:
            if not budget.spend(len(objects) * 3):
                return sorted(distances)
            context = EvaluationContext(row)
            try:
                values = [[part.predict(row, context=context) for part in item] for item in objects]
            except MissingHistory:
                continue
            for (selector, position, value), (other_selector, other_position, other_value) in combinations(values, 2):
                if (selector != other_selector and
                    ((value != 0. or other_value != 0.) if relation.name else value != other_value)):
                    distances.add(abs(position - other_position))
    return sorted(distances)


def propose_relation_revisions(spec, training, budget, calls, *, tolerance=1e-7):
    """Fit local relation corrections and retain distinct untested radius cells."""
    from .research_programs import _Fitter, _entry

    if len(training) < 6:
        return []
    start = budget.spent
    fitter = _Fitter(training, budget, spec.risk_quantile)
    candidates = []
    definitions = {}
    for call in calls:
        if not _has_relation(call, budget, definitions):
            continue
        parent = fitter.fit([call])
        if parent is not None:
            candidates.append((parent, call))
    candidates.sort(key=lambda item: item[0]["loss"])
    if not candidates:
        return []
    # Existing components may simply need a new outer combination.
    complete = fitter.fit([item[1] for item in candidates[:3]] + [call for call in calls if call not in [item[1] for item in candidates]])
    if complete is not None and complete["loss"] <= tolerance:
        return []
    results = []
    for parent, call in candidates[:3]:
        if parent["loss"] <= tolerance:
            continue
        # Pay to expose only the candidates selected by their actual response
        # fit. Repeated versions can contain a large nested correction body.
        opened = _expose(call, budget)
        relations = _relations(opened) if opened is not None else []
        groups = list(dict.fromkeys(_geometry(relation) for relation in relations))
        for geometry in groups[:2]:
            related = [relation for relation in relations if _geometry(relation) == geometry]
            distances = _distances(related, training, budget)
            if len(distances) < 2:
                continue
            # A residual may reject the relation family while its existing
            # distance scale remains useful. Test complete local replacements
            # before spending this task's budget on another radius grid.
            typed, tested = [], {}
            radius = geometry[0].value
            low = max((d for d in distances if d < radius), default=0.)
            high = min((d for d in distances if d >= radius), default=distances[-1] + 1e-6)

            def compare_condition(kind, predicate, value=radius):
                key = kind, predicate, value
                if key not in tested:
                    body = _replace_relation(opened, geometry, value, kind, predicate)
                    revised = _replacement_call(call, body, len(tested))
                    tested[key] = fitter.fit([revised])
                return tested[key]

            def retain_condition(kind, predicate):
                for value in dict.fromkeys((radius, low + .25 * (high - low), low + .75 * (high - low))):
                    fit = compare_condition(kind, predicate, value)
                    if fit is None or fit["loss"] > tolerance:
                        continue
                    operation = "revise_relation_condition" if predicate != TRUE or geometry[3] != TRUE else "revise_relation_type"
                    condition_questions = []
                    if predicate != TRUE:
                        for relation in related:
                            objects = relation_objects(relation)
                            for source in objects:
                                for target in objects:
                                    if source == target:
                                        continue
                                    position = Program("sub", children=(target[1], source[1]))
                                    if kind == "left":
                                        position = Program("mul", children=(Program("const", value=-1.), position))
                                    edge = Program("and", children=(Program("lt", children=(ZERO, position)),
                                        Program("lt", children=(position, Program("const", value=value)))))
                                    condition = instantiate_predicate(predicate, source, target)
                                    condition_questions.append(Program("and", children=(edge, condition)).to_dict())
                    steps = [{"operation": "retain_composed_call", "question": "Preserve the previous learned definition",
                              "program": call, "loss": parent["loss"]},
                             {"operation": operation, "question": "Test a composed edge condition inside the retained computation",
                              "program": fit["program"], "loss": fit["loss"],
                              "parent_loss": parent["loss"], "previous_relation": geometry[2], "relation": kind,
                              "previous_radius": radius, "radius": value, "radius_interval": [low, high],
                              "predicate": predicate.to_dict(), "previous_predicate": geometry[3].to_dict(),
                              "condition_questions": condition_questions, "compact_replacement": True,
                              "relation_expressions": [Program("sub", children=(right, left)).to_dict()
                                  for left, right in combinations(geometry[1], 2)]}]
                    typed.append(_entry(fit, steps, len(training), budget.spent - start, len(typed)))

            for kind in ("symmetric", "right", "left"):
                if kind == geometry[2] and geometry[3] == TRUE:
                    continue
                fit = compare_condition(kind, TRUE)
                if fit is not None and fit["loss"] <= tolerance:
                    retain_condition(kind, TRUE)
            if typed:
                return typed
            cells = [(left + right) / 2 for left, right in zip(distances, distances[1:])]
            coarse = sorted({0., distances[-1] + 1e-6, *(cells[round(index * (len(cells) - 1) / 15)]
                                                       for index in range(16))})
            fitted = []

            def compare(radius):
                delta = _difference(opened, geometry, radius)
                fit = fitter.fit([call, delta])
                if fit is not None and delta in fit["terms"]:
                    fitted.append((fit, radius, delta))
                return fit

            # Compare inexpensive radius and condition directions on the same
            # response rows before committing the remaining research budget.
            radius_trials = [coarse[round(fraction * (len(coarse) - 1))]
                             for fraction in (.2, .4, .6, .8)]
            for value in dict.fromkeys(radius_trials):
                compare(value)
            # Search syntax, not laboratory condition names. Endpoint roles
            # supply atoms; partial response fits nominate conjunctions/disjunctions.
            for role in ("quantity", "position"):
                atoms = atomic_conditions(role)
                partial = {kind: [] for kind in ("right", "left")}
                for kind in partial:
                    for predicate in atoms:
                        fit = compare_condition(kind, predicate)
                        if fit is None:
                            continue
                        partial[kind].append((fit["loss"], predicate))
                        if fit["loss"] <= tolerance:
                            retain_condition(kind, predicate)
                if typed:
                    return typed
                best_condition = min((error for fits in partial.values() for error, _ in fits), default=float("inf"))
                if fitted and min(item[0]["loss"] for item in fitted) <= best_condition:
                    break
                for kind, fits in partial.items():
                    leaders = [predicate for _, predicate in sorted(fits, key=lambda item: item[0])[:4]]
                    for predicate in compound_conditions(leaders):
                        fit = compare_condition(kind, predicate)
                        if fit is not None and fit["loss"] <= tolerance:
                            retain_condition(kind, predicate)
                if typed:
                    return typed
            for radius in coarse:
                if radius not in radius_trials:
                    compare(radius)
            if not fitted:
                continue
            leading = min(fitted, key=lambda item: item[0]["loss"])
            for radius in sorted(cells, key=lambda radius: abs(radius - leading[1]))[:16]:
                if leading[0]["loss"] <= tolerance:
                    break
                if radius in coarse:
                    continue
                fitted_candidate = compare(radius)
                if fitted_candidate is not None and fitted_candidate["loss"] < leading[0]["loss"]:
                    leading = min(fitted, key=lambda item: item[0]["loss"])
            fit, radius, delta = leading
            if fit["loss"] > tolerance:
                continue
            low = max((distance for distance in distances if distance < radius), default=0.)
            high = min((distance for distance in distances if distance >= radius), default=distances[-1] + 1e-6)
            # These unchanged-on-evidence alternatives expose the unresolved
            # relation boundary to the existing full-program experiment planner.
            for fraction in (.25, .75):
                compare(low + (high - low) * fraction)
            for fit, radius, delta in fitted:
                if fit["loss"] > tolerance or call not in fit["terms"]:
                    continue
                interval = [max((distance for distance in distances if distance < radius), default=0.),
                    min((distance for distance in distances if distance >= radius), default=distances[-1] + 1e-6)]
                steps = [{"operation": "retain_composed_call", "question": "Preserve the learned composition",
                          "program": call, "loss": parent["loss"]},
                         {"operation": "revise_nested_relation", "question": "Revise only a retained relation inside its existing context",
                          "program": fit["program"], "loss": fit["loss"], "parent_loss": parent["loss"],
                          "previous_radius": geometry[0].value, "radius": radius, "radius_interval": interval,
                          "relation": geometry[2],
                          "relation_expressions": [Program("abs", children=(Program("sub", children=(left, right)),)).to_dict()
                              for left, right in combinations(geometry[1], 2)],
                          "correction": delta.to_dict()}]
                results.append(_entry(fit, steps, len(training), budget.spent - start, len(results)))
            if results:
                return results
    return results
