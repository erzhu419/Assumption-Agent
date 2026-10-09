"""Deleting one acquired computation preserves independently acquired research."""

from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec


def populated_library():
    library = KnowledgeLibrary()
    spec = TaskSpec('observed_source', 'predict measurements', ('x',), (), ())
    feature = Program('feature', name='x')

    def learn(label, offset, parents=()):
        program = Program('add', children=(feature, Program('const', value=offset)))
        rows = [Observation({'x': value}, value + offset) for value in (-1., 0., 1.)]
        concept_id, added = library.learn(Candidate(label, program, 'observed_response',
                                                   parents=parents), spec, rows)
        assert added
        library.meta_priors.append({'prior_id': label, 'source_concept_id': concept_id,
            'parent_concept_ids': list(parents), 'status': 'supported', 'support_rows': len(rows)})
        return concept_id

    source = learn('source', 1.)
    child = learn('child', 2., (source,))
    descendant = learn('descendant', 3., (child,))
    independent = learn('independent', 4.)
    # A second observation can associate a known computation with a different
    # research lineage without replacing its independently supported record.
    library.meta_priors.append({'prior_id': 'dependent_reuse', 'source_concept_id': independent,
                               'parent_concept_ids': [source], 'status': 'supported'})
    interface = {'controls': {}, 'features': ['x'], 'risk_quantile': .5,
                 'decision_costs': None, 'instruments': {}}

    def measurement(ids, conflicts=()):
        return {'interface': deepcopy(interface), 'source_task': spec.task_id,
                'successful': True, 'source_ids': list(ids), 'channels': [],
                'pilot_length': 8, 'coverage_rows': 24, 'conflicts': list(conflicts),
                'validation_loss': 0.}

    library.measurements.records = [measurement([source]), measurement([source, independent]),
        measurement([independent]), measurement([child], [{
            'observation_end': 24, 'reason': 'new_trajectory_residual',
            'channels': [], 'validation_loss': .4}])]
    offer = {'kind': 'response_revision', 'action': 'refit', 'profile': {},
             'observation_cost': 24, 'compute_cost': 1000}
    library.research_opportunities.observe(offer,
        {'execution_complete': True, 'reference_rows': 24, 'normalized_gain': .5, 'gain': .2},
        observation_cost=24, compute_cost=1200)
    library.research_opportunities.evaluator.events.append({'action': 'retained_observed_return'})
    return library, source, child, descendant, independent


def test_empty_deletion_preserves_complete_library_and_independent_copy():
    library, *_ = populated_library()
    original = json.loads(json.dumps(library.to_dict()))
    unchanged = library.without(set())
    assert json.loads(json.dumps(unchanged.to_dict())) == original
    unchanged.meta_priors[0]['status'] = 'counterexample'
    unchanged.measurements.records[0]['coverage_rows'] = 48
    unchanged.research_opportunities.outcomes[0]['gain'] = -.5
    assert json.loads(json.dumps(library.to_dict())) == original


def test_deleting_lineage_preserves_other_priors_measurements_and_research_returns():
    library, source, child, descendant, independent = populated_library()
    original = deepcopy(library.to_dict())
    removed = library.without({source})
    assert [concept.concept_id for concept in removed.concepts] == [independent]
    assert {source, child, descendant} <= set(removed.deleted_ids)
    assert [prior['prior_id'] for prior in removed.meta_priors] == ['independent']
    records = removed.measurements.to_dict()
    assert len(records) == 3
    assert records[0] == {**original['measurements'][1], 'source_ids': [independent]}
    assert records[1] == original['measurements'][2]
    assert records[2] == {**original['measurements'][3], 'source_ids': [], 'successful': False}
    assert removed.research_opportunities.to_dict() == original['research_opportunities']
    assert library.to_dict() == original
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(removed.to_dict())))
    assert restored.to_dict() == removed.to_dict()
