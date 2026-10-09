"""Observed trajectories revise learned time scales beyond the initial grammar."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.mechanism_revision import propose_mechanism_revisions
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import replay_research_program


def window(width, feature='x'):
    return Program('window_sum', lag=width, children=(Program('feature', name=feature),))


def response(width):
    return Program('add', children=(Program('mul', children=(Program('const', value=1.3), window(width))),
        Program('mul', children=(Program('const', value=-.7), window(width, 'z')))))


def rows(program, seed, length=24):
    rng, history, observed = Random(seed), [], []
    for _ in range(length):
        features = {'x': rng.uniform(-1., 1.), 'z': rng.uniform(-1., 1.)}
        observation = Observation(features, 0., tuple(history))
        observation = replace(observation, target=.2 + program.predict(observation))
        observed.append(observation)
        history.append({**features, '__target__': observation.target})
    return observed


def parent(program):
    definition = Definition('learned_response', (), program, source_concept_id='learned_window')
    return {'candidate': Candidate('source', Program('call', name=definition.definition_id,
        definition=definition), 'callable_transfer'), 'kind': 'callable_transfer'}


SPEC = TaskSpec('measured_response', 'Predict the observed response', ('x', 'z'), (), ())


@pytest.mark.parametrize(('old_width', 'new_width'), [(3, 5), (3, 7), (7, 2)])
def test_paid_history_expands_or_contracts_shared_scale_and_preserves_replay(old_width, new_width):
    original, target = parent(response(old_width)), response(new_width)
    before = deepcopy(original)
    training = rows(target, 11) + rows(target, 12)
    budget = SearchBudget(16000)
    entries = propose_mechanism_revisions(SPEC, training, budget, parent=original)
    revised = next(entry for entry in entries if entry['kind'] == 'mechanism_temporal_refit')
    assert entries[0]['kind'] == 'mechanism_full_refit'
    assert any(entry['kind'] == 'mechanism_deletion' for entry in entries)
    assert loss(entries[0]['candidate'].program, rows(target, 13), 0., .5) > .01
    assert loss(revised['candidate'].program, rows(target, 13), 0., .5) < 1e-16
    metadata = revised['mechanism_revision']
    assert metadata['temporal_revisions'] == [{'operator': 'window_sum', 'temporal_path': [],
        'width_before': old_width, 'width_after': new_width, 'shared_occurrences': 2,
        'history_bound': 24}]
    assert all(1 <= trial['width'] <= 24 for trial in metadata['temporal_trials'])
    assert metadata['temporal_search_compute'] == sum(trial['compute_cost'] for trial in metadata['temporal_trials'])
    assert revised['candidate'].parents == ('learned_window',)
    assert revised['method_source_ids'] == ['learned_window']
    assert metadata['status'] == 'proposed' and 'validation_loss' not in revised
    plan = json.loads(json.dumps(revised['research_program']))
    assert plan['source_ids'] == ['learned_window']
    assert plan['selection_steps'][0]['operation'] == 'revise_temporal_scale'
    replay = replay_research_program(plan, training, SearchBudget(4000))
    assert loss(replay['candidate'].program, rows(target, 14), 0., .5) < 1e-16
    assert original == before and 0 < budget.spent <= budget.limit


def test_nested_equal_width_layers_are_distinct_and_repeated_subexpressions_move_together():
    def composed(width):
        nested = Program('window_sum', lag=width, children=(response(3),))
        return Program('mul', children=(nested, nested))

    original, target = parent(composed(3)), composed(5)
    entries = propose_mechanism_revisions(SPEC, rows(target, 21) + rows(target, 22),
                                         SearchBudget(18000), parent=original)
    revised = next(entry for entry in entries if entry['kind'] == 'mechanism_temporal_refit')
    assert loss(revised['candidate'].program, rows(target, 23), 0., .5) < 1e-12
    edit = revised['mechanism_revision']['temporal_revisions'][0]
    assert edit['temporal_path'] == [] and edit['shared_occurrences'] == 2
    assert edit['width_before'] == 3 and edit['width_after'] == 5
    assert edit['history_bound'] == 22
    assert revised['candidate'].program.history_depth == 6


def test_no_search_without_residual_or_identifiable_history_and_small_budget_keeps_control():
    original, target = parent(response(3)), response(7)
    already_explained = propose_mechanism_revisions(SPEC, rows(response(3), 31),
                                                   SearchBudget(8000), parent=original)
    assert not any(entry['kind'] == 'mechanism_temporal_refit' for entry in already_explained)
    short = propose_mechanism_revisions(SPEC, rows(target, 31, 2) + rows(target, 32, 2),
                                       SearchBudget(8000), parent=original)
    assert not any(entry['kind'] == 'mechanism_temporal_refit' for entry in short)
    budget = SearchBudget(600)
    limited = propose_mechanism_revisions(SPEC, rows(target, 31) + rows(target, 32), budget, parent=original)
    assert limited and limited[0]['kind'] == 'mechanism_full_refit'
    assert budget.spent <= budget.limit


def test_changed_window_keeps_relation_symbols_and_historical_constraints_after_json():
    from hegel_machine.cognition.relation_revisions import _relations
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from test_cognition_mechanism_revision import source_parent
    from test_cognition_mechanism_revision_research import ChangingMechanismLab, source_knowledge

    library, source_id = source_knowledge()
    original = source_parent(library, source_id)
    lab = ChangingMechanismLab(-.7, 'measured_relation')
    old_rows = lab.sample(31, 48).observations
    radius = _relations(original['candidate'].program.expanded())[0].children[0].value
    prior = {'source_concept_id': source_id,
        'structure': {'fits': [{'radius': radius}], 'radius_interval': [0., radius + .4]}}
    work = SearchBudget(30000)
    inherited = RelationIntervalResearch(original['candidate'].program, [prior], work)
    conditioned = inherited.alternatives(old_rows, work, experiments=[old_rows])
    assert conditioned and inherited.evidence_contexts
    prior['research_program'] = conditioned[0]['research_program']
    before = deepcopy(prior)

    def wider(node):
        return replace(node, lag=5 if node.op == 'window_sum' else node.lag,
                       children=tuple(wider(child) for child in node.children))

    target = wider(original['candidate'].program.expanded())
    training = [replace(row, target=target.predict(row)) for row in lab.sample(51, 48).observations]
    entries = propose_mechanism_revisions(lab.spec, training, SearchBudget(16000),
                                         parent=original, priors=[prior])
    revised = next(entry for entry in entries if entry['kind'] == 'mechanism_temporal_refit')
    fresh = [replace(row, target=target.predict(row)) for row in lab.sample(91, 35).observations]
    assert loss(revised['candidate'].program, fresh, 0., .5) < 1e-14
    plan = json.loads(json.dumps(revised['research_program']))
    restored = RelationIntervalResearch(revised['candidate'].program, [prior],
                                       SearchBudget(4000), research_program=plan)
    assert restored.parameters[0]['radius_intervals'] == inherited.parameters[0]['radius_intervals']
    assert restored.evidence_contexts == json.loads(json.dumps(inherited.evidence_contexts))
    assert [step['operation'] for step in plan['selection_steps']] == [
        'resolve_relation_intervals', 'revise_temporal_scale']
    assert plan['status'] == 'requires_fresh_validation' and prior == before
