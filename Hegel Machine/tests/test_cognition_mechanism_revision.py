"""Paid additive responses nominate, but do not confirm, mechanism removal."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.mechanism_revision import propose_mechanism_revisions
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.relation_revisions import _relations
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_mechanism_revision_research import ChangingMechanismLab, source_knowledge


def source_parent(library, source_id):
    source = next(item for item in library.concepts if item.concept_id == source_id)
    spec = ChangingMechanismLab(-.7, 'source_rebinding').spec
    call = library.callable_atoms(spec, limit=1)[0]
    assert call.expanded() == source.program.expanded()
    return {'candidate': Candidate('retained_source', call, 'callable_transfer'),
            'kind': 'callable_transfer', 'family_id': 'T03'}


def contains_relation(program):
    return program.op in {'neighbor_sum', 'relation_sum'} or any(
        contains_relation(child) for child in program.children)


def test_removed_mechanism_is_proposed_alongside_full_refit_and_keeps_learned_source():
    library, source_id = source_knowledge()
    before = deepcopy(library.to_dict())
    parent = source_parent(library, source_id)
    original = parent['candidate'].program.to_dict()
    lab = ChangingMechanismLab(0., 'observed_change')
    training = [row for seed in (11, 12) for row in lab.sample(seed, 24).observations]
    budget = SearchBudget(6000)
    entries = propose_mechanism_revisions(lab.spec, training, budget, parent=parent)
    full = next(item for item in entries if item['kind'] == 'mechanism_full_refit')
    removed = next(item for item in entries if item['kind'] == 'mechanism_deletion'
        and item['mechanism_revision']['removed_term'] is not None
        and contains_relation(Program.from_dict(item['mechanism_revision']['removed_term'])))
    assert full['training_loss'] < 1e-16 and removed['training_loss'] < 1e-16
    assert loss(parent['candidate'].program, training, 0., .5) > .01
    assert not contains_relation(removed['candidate'].program)
    assert removed['candidate'].program.history_depth == 2
    for entry in entries:
        assert entry['candidate'].parents == (source_id,)
        assert entry['method_source_ids'] == [source_id]
        assert entry['mechanism_revision']['source_ids'] == [source_id]
        assert entry['mechanism_revision']['original_program'] == original
        assert entry['mechanism_revision']['status'] == 'proposed'
        assert 'validation_loss' not in entry and 'structure' not in entry
    assert library.to_dict() == before
    assert parent['candidate'].program.to_dict() == original
    assert 0 < budget.spent <= budget.limit


def test_still_necessary_relation_is_not_removed_by_refitting_the_remaining_response():
    library, source_id = source_knowledge()
    lab = ChangingMechanismLab(-.35, 'another_observed_change')
    training = [row for seed in (11, 12) for row in lab.sample(seed, 24).observations]
    entries = propose_mechanism_revisions(lab.spec, training, SearchBudget(6000),
                                        parent=source_parent(library, source_id))
    full = next(item for item in entries if item['kind'] == 'mechanism_full_refit')
    removed = next(item for item in entries if item['kind'] == 'mechanism_deletion'
        and item['mechanism_revision']['removed_term'] is not None
        and contains_relation(Program.from_dict(item['mechanism_revision']['removed_term'])))
    fresh = lab.sample(991, 47).observations
    assert loss(full['candidate'].program, fresh, 0., .5) < 1e-16
    assert loss(removed['candidate'].program, fresh, 0., .5) > .01
    assert removed['mechanism_revision']['status'] == 'proposed'


def test_one_temporal_revision_removes_shared_window_without_losing_inner_event():
    library, source_id = source_knowledge()
    parent = source_parent(library, source_id)
    lab = ChangingMechanismLab(-.7, 'observed_change')
    training = [row for seed in (11, 12) for row in lab.sample(seed, 24).observations]
    entries = propose_mechanism_revisions(lab.spec, training, SearchBudget(6000), parent=parent)
    operators = [item for item in entries if item['mechanism_revision']['removed_operator']]
    assert [item['mechanism_revision']['removed_operator'] for item in operators] == [
        {'op': 'window_sum', 'lag': 3}, {'op': 'event_delta', 'lag': 1}]

    def contains(program, op):
        return program.op == op or any(contains(child, op) for child in program.children)

    without_window = operators[0]['candidate'].program
    assert not contains(without_window, 'window_sum')
    assert contains(without_window, 'event_delta') and contains_relation(without_window)
    without_event = operators[1]['candidate'].program
    assert not contains(without_event, 'event_delta')
    assert contains(without_event, 'window_sum') and contains_relation(without_event)
    assert all(item['mechanism_revision']['removed_term'] is None for item in operators)
    assert all(item['mechanism_revision']['status'] == 'proposed' for item in operators)
    fresh = lab.sample(991, 47).observations
    assert all(loss(item['candidate'].program, fresh, 0., .5) > .001 for item in operators)


def test_shared_nonlinear_consumer_survives_local_mechanism_deletion():
    library, source_id = source_knowledge()
    parent = source_parent(library, source_id)
    lab = ChangingMechanismLab(0., 'observed_change')
    rows = [replace(row, target=row.target ** 2) for row in lab.sample(11, 48).observations]
    call = parent['candidate'].program
    nonlinear = {**parent, 'candidate': Candidate('squared_response',
        Program('mul', children=(call, call)), 'composed_response')}
    original = deepcopy(nonlinear['candidate'].program.to_dict())
    budget = SearchBudget(6000)
    entries = propose_mechanism_revisions(lab.spec, rows, budget, parent=nonlinear)
    full = next(item for item in entries if item['kind'] == 'mechanism_full_refit')
    removed = next(item for item in entries if item['mechanism_revision'].get('edit_context')
        and not contains_relation(item['candidate'].program))
    fresh = [replace(row, target=row.target ** 2) for row in lab.sample(91, 39).observations]
    assert loss(full['candidate'].program, fresh, 0., .5) > .001
    assert loss(removed['candidate'].program, fresh, 0., .5) < 1e-16
    program = removed['candidate'].program
    assert program.op == 'mul' and program.children[0] == program.children[1]
    assert program.history_depth == 2
    assert removed['mechanism_revision']['shared_occurrences'] == 2
    assert removed['candidate'].parents == (source_id,)
    assert nonlinear['candidate'].program.to_dict() == original
    assert 0 < budget.spent <= budget.limit


def test_local_subtraction_keeps_sign_and_coefficient_in_every_consumer():
    lab = ChangingMechanismLab(0., 'signed_local_change')
    left = Program('mul', children=(Program('const', value=2.), Program('feature', name='q0')))
    right = Program('mul', children=(Program('const', value=3.), Program('feature', name='q1')))
    context = Program('sub', children=(left, right))
    offset = Program('const', value=.4)
    original = Program('mul', children=(context, Program('add', children=(context, offset))))
    replacement = Program('mul', children=(Program('const', value=-1.), right))
    expected = Program('mul', children=(replacement, Program('add', children=(replacement, offset))))
    parent = {'candidate': Candidate('scalar_composition', original, 'observed_response')}
    rows = [replace(row, target=expected.predict(row)) for row in lab.sample(22, 48).observations]
    entries = propose_mechanism_revisions(lab.spec, rows, SearchBudget(6000), parent=parent)
    local = [entry for entry in entries if entry['mechanism_revision'].get('edit_context')
             == context.to_dict()]
    assert len(local) == 2
    removed = next(entry for entry in local
                   if entry['mechanism_revision']['removed_term'] == left.to_dict())
    assert removed['mechanism_revision']['replacement_context'] == replacement.to_dict()
    assert all(entry['mechanism_revision']['shared_occurrences'] == 2 for entry in local)
    fresh = [replace(row, target=expected.predict(row)) for row in lab.sample(93, 47).observations]
    assert loss(removed['candidate'].program, fresh, 0., .5) < 1e-16
    assert parent['candidate'].program == original


def test_exhausted_search_cannot_fit():
    library, source_id = source_knowledge()
    parent = source_parent(library, source_id)
    lab = ChangingMechanismLab(0., 'observed_change')
    rows = lab.sample(11, 24).observations
    exhausted = SearchBudget(1)
    assert propose_mechanism_revisions(lab.spec, rows, exhausted, parent=parent) == []
    assert exhausted.spent == exhausted.limit


@pytest.mark.parametrize('nonlinear', [False, True])
def test_refitting_preserves_consumable_range_evidence_only_for_retained_relations(nonlinear):
    library, source_id = source_knowledge()
    parent = source_parent(library, source_id)
    if nonlinear:
        call = parent['candidate'].program
        parent = {**parent, 'candidate': Candidate('squared_source',
            Program('mul', children=(call, call)), 'composed_response')}
    source = ChangingMechanismLab(-.7, 'range_source')
    paid = source.sample(31, 64).observations
    if nonlinear:
        paid = [replace(row, target=row.target ** 2) for row in paid]
    radius = _relations(parent['candidate'].program.expanded())[0].children[0].value
    upper = max(abs(row.features['x0'] - row.features['x1']) for row in paid) + 1e-6
    prior = {'source_concept_id': source_id,
        'structure': {'fits': [{'radius': radius}], 'radius_interval': [0., upper]}}
    work = SearchBudget(40000)
    inherited = RelationIntervalResearch(parent['candidate'].program, [prior], work)
    conditioned = inherited.alternatives(paid, work, experiments=[paid])
    assert conditioned and inherited.evidence_contexts
    # The squared response owns this new context; its component source still
    # describes the original unsquared callable.
    if nonlinear:
        parent['research_program'] = conditioned[0]['research_program']
    else:
        prior['research_program'] = conditioned[0]['research_program']
    before = deepcopy(prior)

    target = ChangingMechanismLab(-.7 if nonlinear else -.35, 'changed_response')
    rows = target.sample(71, 48).observations
    if nonlinear:
        rows = [replace(row, target=1.7 * row.target ** 2) for row in rows]
    proposals = propose_mechanism_revisions(target.spec, rows, SearchBudget(10000),
                                            parent=parent, priors=[prior])
    full = next(entry for entry in proposals if entry['kind'] == 'mechanism_full_refit')
    for entry in proposals:
        plan = json.loads(json.dumps(entry.get('research_program')))
        resumed = RelationIntervalResearch(entry['candidate'].program, [prior],
                                            SearchBudget(4000), research_program=plan)
        if contains_relation(entry['candidate'].program):
            assert len(resumed.parameters) == 1 and resumed.status == 'unresolved'
            assert resumed.parameters[0]['radius_intervals'] == inherited.parameters[0]['radius_intervals']
            assert resumed.evidence_contexts == json.loads(json.dumps(inherited.evidence_contexts))
            assert plan['status'] == 'requires_fresh_validation'
        else:
            assert plan is None and not resumed.parameters
    replay = RelationIntervalResearch(full['candidate'].program, [prior], SearchBudget(4000),
                                      research_program=full['research_program'])
    assert replay.alternatives(rows, SearchBudget(16000), experiments=[rows])
    assert prior == before

    # An already exposed parent needs its local parameter template as well;
    # retaining source IDs alone cannot recover those current coefficients.
    second = propose_mechanism_revisions(target.spec, rows, SearchBudget(10000),
                                         parent=full, priors=[prior])
    assert second
    assert second[0]['mechanism_revision']['original_research_program'] == full['research_program']
    assert second[0]['research_program']['status'] == 'requires_fresh_validation'
