"""Saved relation programs reach existing joint research without a structure copy."""

from copy import deepcopy
import json
import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_programs import _Fitter, _entry, replay_research_program
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_structure import _entries, _research_response_terms
from test_cognition_structural_neighbor_cells import paid_response
from test_relation_fitted_columns import fitted_entry, interval_answer


def test_actual_replayed_relation_reaches_joint_fit_and_complete_paid_comparison():
    rows, schema, response = paid_response()
    spec = TaskSpec('paid_replayed_relation', '', schema.feature_names, (), (),
        context={'object_schema': json.dumps(schema.to_dict())})
    setup = SearchBudget(10000)
    # Retain the historical weak cell; the paid fixture already refutes it.
    found = discover_structure(rows, schema, budget=setup, families=('joint_event',),
        response_terms=response, radius_candidates=[.9872102935595741])
    source = _entries(found, 'retained', spec, rows, setup, 1e-7)[0]
    research = RelationIntervalResearch(source['candidate'].program, [], setup,
        structure=source['structure'], research_program=source['research_program'])
    saved = research._entry(research.values, source['structure']['radius_interval'],
                           0, source['training_loss'], len(rows), 0)
    parent = replay_research_program(saved['research_program'], rows, setup)
    assert parent is not None and 'structure' not in parent
    assert loss(parent['candidate'].program, rows, 0., .5) > 1e-10
    original = deepcopy((source, saved, parent))
    known = {item['candidate'].program: loss(item['candidate'].program, rows, 0., .5)
             for item in (source, parent)}
    budget, actions = SearchBudget(100000), []
    result = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-10))._revise_paid_structure(
        spec, parent, rows, budget, actions, finalists=[source, parent], paid_scores=known)

    assert result['status'] == 'improved' and result['full_comparison']
    assert result['new_claims'] and result['selected']['validation_loss'] < 1e-16
    selected = result['selected']
    assert selected['structure']['fits'][0]['radius'] != found.fits[0]['radius']
    assert selected['structure']['fits'][0]['coefficients'] != found.fits[0]['coefficients']
    assert set(known) <= {item['candidate'].program for item in result['compared_claims']}
    assert result['action']['old_scoring_compute'] == 0
    assert result['action']['new_observations'] == 0
    assert 0 < result['compute_cost'] == budget.spent <= budget.limit
    assert (source, saved, parent) == original


def test_joint_response_basis_keeps_saved_zero_weight_column():
    retained = interval_answer(fitted_entry(), .7)
    program = retained['candidate'].program
    assert 'z6' not in program.required_features()
    basis = _research_response_terms(retained, SearchBudget(1000))
    assert Program('feature', name='z6') in basis
    assert not any('neighbor_sum' in term.describe() for term in basis)


def test_unrelated_program_is_not_implicitly_admitted_to_structure_research(monkeypatch):
    from hegel_machine.cognition import unified_structure
    rows, schema, _ = paid_response()
    spec = TaskSpec('plain_program', '', schema.feature_names, (), (),
        context={'object_schema': json.dumps(schema.to_dict())})
    setup = SearchBudget(1000)
    fit = _Fitter(rows, setup, .5).fit([Program('feature', name=schema.feature_names[0])])
    parent = _entry(fit, [], len(rows), setup.spent)
    monkeypatch.setattr(unified_structure, 'propose_structures',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unrelated fit started structure research')))
    budget = SearchBudget(100000)
    result = UnifiedResearchAgent()._revise_paid_structure(spec, parent, rows, budget, [])
    assert result['status'] == 'unavailable'
    assert not result['new_claims'] and budget.spent == 0


@pytest.mark.parametrize('action', ['relation_form_revision', 'relation_parameter_revision'])
def test_relation_form_segment_keeps_paid_pool_and_separates_fit_from_comparison(monkeypatch, action):
    from hegel_machine.cognition import relation_revisions, transfer_partitions
    from hegel_machine.cognition.continuous_state import _sum_terms
    from hegel_machine.cognition.models import Candidate
    from test_relation_fitted_columns import coordinates, measured_rows

    parent = interval_answer(fitted_entry(), .7)
    target = _sum_terms(coordinates(.7), [-.3, .1, .8, -.5, .9, -.2, .4, 1.7], -.6)
    rows = measured_rows(target)
    training = rows[:72]
    spec = TaskSpec('paid_form_revision', '', tuple(rows[0].features), (), ())
    prior_loss = loss(parent['candidate'].program, rows, 0., .5)
    proposal = {**parent, 'candidate': Candidate('revised', target, 'representation')}
    original = deepcopy(parent)
    calls = []

    initial = {'entry': parent, 'training': training, 'risk_quantile': .5, 'cost_ref': 'paid_fit'}
    method = {'actual_current_parent_body': True}

    def proposed(spec, observations, budget, *, parent, max_candidates, **options):
        assert options == ({'method': method, 'feedback_rows': rows[72:],
            'paid_initial': initial, 'tolerance': 1e-10} if action == 'relation_parameter_revision' else {})
        calls.append((list(observations), parent, max_candidates, budget.limit))
        budget.spend(7)
        return {'entries': [proposal], 'research': {'execution_complete': True}}

    monkeypatch.setattr(transfer_partitions if action == 'relation_parameter_revision' else relation_revisions,
        'propose_parent_parameter_revisions' if action == 'relation_parameter_revision'
        else 'propose_parent_relation_revisions', proposed)
    budget = SearchBudget(10000)
    result = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-10))._revise_paid_structure(
        spec, parent, rows, budget, [], finalists=[parent],
        paid_scores={parent['candidate'].program: prior_loss},
        research_action=action, fit_rows=training, research_method=method,
        feedback_rows=rows[72:], paid_initial=initial)
    assert calls == [(training, parent, 4, 10000 - 4 * len(rows))]
    assert result['status'] == 'improved' and result['execution_complete']
    assert result['selected']['candidate'].program == target
    assert result['selected']['validation_loss'] == 0.
    assert len(result['compared_claims']) == 2
    assert result['action']['old_scoring_compute'] == 0
    assert budget.spent == 7 + len(rows)
    assert parent == original
