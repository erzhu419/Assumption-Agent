"""Propose all-object instances of a saved, fully covered structural computation.

Object-count invariance is a hypothesis here. The source remains unchanged;
current rows fit the proposal and the ordinary candidate pool decides its use.
"""

from copy import deepcopy
from dataclasses import replace
import json

from .language import Program
from .research_programs import _entry, _Fitter
from .structural_discovery import _neighbors, _sum
from .structure_lab import ObjectSchema


def _schema(value):
    if value is None or isinstance(value, ObjectSchema):
        return value
    return ObjectSchema.from_dict(json.loads(value) if isinstance(value, str) else value)


def _whole_view(schema):
    members = [member for group in schema.objects for member in schema.members.get(group, ())]
    return (bool(schema.objects) and len(set(schema.objects)) == len(schema.objects)
        and len(members) == len(set(members)) and set(members) == set(schema.controls))


def object_scope_revision_terms(prior, current_schema):
    """Construct one untested change of object scope without predicting or fitting.

    Only uniform one-object summands and complete f/x/q neighborhood lists are
    expanded. Bare individual fields and partial object lists cannot acquire a
    whole-apparatus meaning merely because the target contains more objects.
    """
    source, target = _schema(prior.get('object_schema')), _schema(current_schema)
    plan = prior.get('research_program')
    if (prior.get('status') != 'supported' or not plan or source is None or target is None
            or len(source.objects) == len(target.objects)
            or not _whole_view(source) or not _whole_view(target)):
        return None
    owners = {name: (group, role) for group in source.objects
              for role, name in source.fields[group].items() if role in {'q', 'u', 'x', 'f'}}
    if len(owners) != sum(len([role for role in source.fields[group]
                              if role in {'q', 'u', 'x', 'f'}]) for group in source.objects):
        return None
    expanded = []

    def full_sum(node, path):
        groups, patterns = [], []
        for child in node.children:
            fields = child.required_features()
            if not fields or not fields <= owners.keys():
                return None
            local = {owners[name][0] for name in fields}
            if len(local) != 1:
                return None
            group = next(iter(local))
            groups.append(group)
            patterns.append(child.remap_features({name: 'object_role:' + role
                for role, name in source.fields[group].items()}))
        if (len(groups) != len(source.objects) or set(groups) != set(source.objects)
                or any(pattern != patterns[0] for pattern in patterns[1:])):
            return None
        pattern = patterns[0]
        roles = {name.removeprefix('object_role:') for name in pattern.required_features()}
        if any(not roles <= target.fields[group].keys() for group in target.objects):
            return None
        result = _sum(pattern.remap_features({'object_role:' + role: target.fields[group][role]
            for role in roles}) for group in target.objects)
        expanded.append({'path': list(path), 'operation': 'object_sum', 'roles': sorted(roles)})
        return result

    def full_neighbors(node, path):
        radius = node.children[0]
        if radius.op != 'const':
            return None
        triples = [node.children[i:i + 3] for i in range(1, len(node.children), 3)]
        expected = [tuple(Program('feature', name=source.fields[group][role])
                    for role in ('f', 'x', 'q')) for group in source.objects
                    if {'f', 'x', 'q'} <= source.fields[group].keys()]
        if (len(expected) != len(source.objects) or len(triples) != len(expected)
                or set(triples) != set(expected)
                or any(not {'f', 'x', 'q'} <= target.fields[group].keys() for group in target.objects)):
            return None
        expanded.append({'path': list(path), 'operation': 'neighbor_sum', 'roles': ['f', 'x', 'q']})
        return replace(_neighbors(target, radius.value), name=node.name)

    def rewrite(node, path):
        if node.op == 'object_sum':
            changed = full_sum(node, path)
            if changed is not None:
                return changed
            # A sum of already aggregated computations is ordinary composition.
            # A partial sum of bare object fields fails in the recursive step.
        if node.op == 'neighbor_sum':
            return full_neighbors(node, path)
        if node.op in {'feature', 'lag', 'arg', 'call', 'relation_sum'}:
            return None
        children = tuple(rewrite(child, (*path, index)) for index, child in enumerate(node.children))
        if any(child is None for child in children):
            return None
        return replace(node, children=children)

    terms = []
    for key in plan['term_nodes']:
        answer = plan['graph']['nodes'][key].get('answer')
        if answer is None:
            return None
        term = rewrite(Program.from_dict(answer), (key,))
        if term is None:
            return None
        terms.append(term)
    if not terms or not expanded:
        return None
    scope = {'status': 'untested_object_scope_revision',
        'source_concept_id': prior['source_concept_id'],
        'source_objects': list(source.objects), 'target_objects': list(target.objects),
        'source_schema': source.to_dict(), 'target_schema': target.to_dict(),
        'source_term_nodes': list(plan['term_nodes']), 'expanded_reductions': expanded,
        'parameter_scope': 'refit_linear_weights; retain_source_temporal_and_relation_parameters',
        'claim': 'The source whole-object computations may extend over the current complete object view'}
    if plan.get('source_fit_ref') is not None:
        scope['source_fit_ref'] = deepcopy(plan['source_fit_ref'])
    return {'fit_terms': tuple(terms), 'source_concept_id': prior['source_concept_id'],
            'proposal_scope': scope}


def propose_object_scope_revisions(spec, training, budget, priors, *, max_candidates=4):
    """Fit scope proposals using the existing research-program contract."""
    rows = list(training)
    if len(rows) < 6:
        return []
    fitter, entries = _Fitter(rows, budget, spec.risk_quantile), []
    for prior in priors:
        if len(entries) >= max_candidates:
            break
        proposal = object_scope_revision_terms(prior, spec.context.get('object_schema'))
        if proposal is None or any(not term.required_features() <= set(spec.feature_names)
                                   for term in proposal['fit_terms']):
            continue
        before = budget.spent
        fit = fitter.fit(proposal['fit_terms'])
        if fit is None:
            continue
        scope, source = proposal['proposal_scope'], proposal['source_concept_id']
        steps = [{'operation': 'instantiate_full_object_scope',
            'question': 'Test whether the saved whole-object computation extends to the current object count',
            'program': fit['program'], 'loss': fit['loss'], 'source_concept_id': source}]
        entry = _entry(fit, steps, len(rows), budget.spent - before, len(entries))
        sources = list(dict.fromkeys([*entry['research_program']['source_ids'], source]))
        entry['candidate'] = replace(entry['candidate'], candidate_id=f'object_scope_revision:{len(entries)}',
            parents=tuple(sources))
        entry['method_source_ids'] = sources
        entry['object_scope_revision'] = deepcopy(scope)
        entry['research_program'].update(source_ids=sources, object_scope_revision=deepcopy(scope))
        entries.append(entry)
    return entries
