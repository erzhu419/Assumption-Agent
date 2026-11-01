"""Paid interval updates retain the fitted readout and evidence row roles."""

from copy import deepcopy
from dataclasses import replace
from statistics import mean

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch, _bind
from hegel_machine.cognition.research_programs import _entry
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import finalize_interval_candidates


def evaluated_interval_candidates():
    relation = Program('neighbor_sum', children=(Program('arg', name='r'),
        *(Program('feature', name=name) for name in ('f0', 'x0', 'q0', 'f1', 'x1', 'q1'))))
    templates = [Program('feature', name='drive'), relation,
        *(Program('feature', name=f'extra{i}') for i in range(6))]
    coefficients = [.5, 2., 3., 0., .1, -.2, .4, .7, -.8]
    terms = [_bind(term, {'r': Program('const', value=.75)}) for term in templates]
    program = _sum_terms(terms, coefficients[1:], coefficients[0])
    fit = {'program': program, 'fit_terms': terms, 'terms': terms,
        'coefficients': coefficients, 'loss': 0.}
    step = {'operation': 'resolve_relation_intervals', 'question': 'Resolve the saved relation',
        'program': program, 'loss': 0., 'parameters': [{'symbol': 'r', 'radius': .75,
            'radius_intervals': [[.6, 1.]], 'prior_radius_intervals': [[0., 1.2]],
            'source_concept_id': None, 'relation': 'symmetric'}],
        'parameter_template': _sum_terms(templates, coefficients[1:], coefficients[0]).to_dict(),
        'parameter_terms': [term.to_dict() for term in templates]}
    entry = _entry(fit, [step], 96, 1)
    body = entry['research_program']
    body['readout_fit_source'] = {'source_ref': {'path': 'paid_fit', 'entry_index': 0},
        'fit_rows': 96, 'fixed_readout': True, 'new_fits': 0}
    body['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'second_stage': {'operation': 'resolve_relation_intervals', 'compute_limit': 15000}}
    body['transfer_partition_search'] = {'status': 'completed', 'fit_rows': 96}
    rows = []
    for i, distance in enumerate([.2, .4, .6, 1.1] * 24 + [.7, .8] * 39):
        row = Observation({'drive': (i % 5) * .3, 'f0': 1., 'f1': 0.,
            'x0': 0., 'x1': distance, 'q0': .2 * (i % 7), 'q1': 1. + .13 * (i % 4),
            **{f'extra{j}': (i % (j + 2)) * .1 for j in range(6)}}, 0.)
        rows.append(replace(row, target=program.predict(row)))
    budget = SearchBudget(15000)
    research = RelationIntervalResearch(program, [], budget, research_program=body)
    experiments = [rows[96:144], rows[144:171], rows[171:]]
    candidates = research.alternatives(rows, budget, 1e-7,
        experiments=experiments, interval_precision=0., parameter_index=0)
    assert candidates and budget.last_denied_request is None
    return entry, research, rows, experiments, candidates, budget


@pytest.mark.parametrize('source_ref', [None, {'path': 'original_paid_fit', 'entry_index': 3}])
def test_same_program_interval_update_preserves_eight_columns_without_recomputation(monkeypatch, source_ref):
    import hegel_machine.cognition.research_programs as module

    entry, research, rows, experiments, candidates, budget = evaluated_interval_candidates()
    before, spent = deepcopy(entry), budget.spent
    original_fit = research_program_fit_contract(entry['research_program'])
    def unexpected(*args, **kwargs):
        pytest.fail('finalizing paid candidates repeated prediction or fitting')
    monkeypatch.setattr(Program, 'predict', unexpected)
    monkeypatch.setattr(module._Fitter, 'fit', unexpected)

    assert finalize_interval_candidates(entry, candidates, research, rows[:96], rows[96:],
        feedback_experiments=experiments, fit_source_ref=source_ref) is candidates
    retained = next(item for item in candidates if item['candidate'].program == entry['candidate'].program)
    body = retained['research_program']
    fit = research_program_fit_contract(body)
    assert len(fit['fit_terms']) == 8 and fit['fit_terms'] == original_fit['fit_terms']
    assert fit['coefficients'] == original_fit['coefficients'] and fit['coefficients'][3] == 0.
    assert body['fit_rows'] == 96 and body['evaluation_rows'] == retained['evaluation_rows'] == 174
    assert body['selection_steps'][-1]['parameters'][0]['radius_intervals'] == [[.7, .8]]
    assert body['readout_fit_source'] == {**before['research_program']['readout_fit_source'],
        'source_ref': source_ref or before['research_program']['readout_fit_source']['source_ref']}
    assert body['transfer_partition_search'] == before['research_program']['transfer_partition_search']
    assert body['transfer_partition']['released_groups'] == [research.parameters[0]['symbol']]
    assert retained['feedback_trajectory_losses'] == [0., 0., 0.]
    assert len(retained['paid_row_squared_errors']) == 174
    assert entry == before and budget.spent == spent


def test_paid_error_cache_keeps_original_training_and_each_feedback_scope():
    entry, research, rows, experiments, candidates, _ = evaluated_interval_candidates()
    candidate = candidates[0]
    # Test aggregation independently with deliberately different paid losses.
    errors = [1.] * 96 + [2.] * 48 + [3.] * 27 + [4.] * 3
    research.last_candidate_errors[candidate['candidate'].program] = dict(zip(map(id, rows), errors))
    candidate['validation_loss'] = mean(errors)
    finalize_interval_candidates(entry, [candidate], research, rows[:96], rows[96:],
        feedback_experiments=experiments)
    assert candidate['training_loss'] == 1.
    assert candidate['validation_loss'] == mean(errors[96:])
    assert candidate['all_paid_loss'] == mean(errors)
    assert candidate['paid_row_squared_errors'] == errors
    assert candidate['feedback_trajectory_losses'] == [2., 3., 4.]
    body = candidate['research_program']
    assert body['evaluation_loss'] == mean(errors)
    assert body['graph']['nodes'][body['output_node']]['loss'] == 1.
    assert body['transfer_partition_followup']['new_fits'] == 0
    assert not body['transfer_partition_followup']['independent_support']
