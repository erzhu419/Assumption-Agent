"""Actual structure fits become reusable programs without replaying discovery."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structural_research_program import (
    attach_structural_research_program, structural_fit_contract,
)
from hegel_machine.cognition.unified_structure import _entries
from test_cognition_structural_neighbor_cells import paid_response


def test_actual_structure_fit_retains_complete_columns_and_replays_new_weights():
    rows, schema, response = paid_response()
    budget = SearchBudget(9000)
    found = discover_structure(rows, schema, budget=budget, families=('joint_event',),
                               response_terms=response)
    spec = TaskSpec('paid', '', schema.feature_names, (), (),
                    context={'object_schema': json.dumps(schema.to_dict())})
    before = budget.spent
    entries = _entries(found, 'full', spec, rows, budget, 1e-7)
    assert budget.spent == before  # Compiling the paid fit does not run another solve.
    entry = entries[0]
    fit = entry['structure']['fits'][0]
    assert len(fit['response_terms']) == 3
    assert len(fit['fit_terms']) == 5
    assert len(fit['coefficients']) == 6
    plan = entry['research_program']
    assert plan['structural_replay']['coordinate_basis'] == 'complete_fitted_columns'
    assert plan['structural_replay']['structure_revision_entry'] == 'propose_structures'
    assert plan['status'] == 'requires_fresh_validation'
    graph = ResearchGraph.from_dict(plan['graph'])
    assert graph.nodes[plan['output_node']].answer == entry['candidate'].program
    terms = [graph.nodes[key].answer for key in plan['term_nodes']]
    assert terms == [Program.from_dict(term) for term in fit['fit_terms']]
    assert any(term.op == 'neighbor_sum' for term in terms)
    assert any(term.op == 'event_delta' for term in terms)
    json.dumps(found.to_dict())
    json.dumps(plan)

    changed = _sum_terms(terms, [.2, -.6, .4, 1.8, -.9], .3)
    current = [Observation(dict(row.features), changed.predict(row), row.history) for row in rows]
    replay = replay_research_program(plan, current, SearchBudget(1500))
    assert replay is not None
    assert replay['research_program']['reused_procedure']
    assert replay['candidate'].program != entry['candidate'].program
    assert loss(replay['candidate'].program, current, 0., .5) < 1e-12


def saved_entry():
    x = Program('feature', name='x')
    # Keep the reset-dependent constant inside the window as part of its term.
    window = Program('window_sum', lag=3, children=(
        Program('add', children=(x, Program('const', value=2.))),))
    event = Program('event_delta', lag=1, children=(x,))
    program = _sum_terms([window, event], [1.8, -.6], .25)
    return {'candidate': Candidate('old_structure', program, 'representation'),
        'kind': 'structural:joint_window_event', 'family_id': 'T12', 'training_loss': 1e-21,
        'validation_loss': 2e-21, 'structure': {'compute_cost': 471,
            'fits': [{'family': 'joint_window_event', 'window': 3, 'radius': .7,
                      'radius_interval': [.65, .75], 'response_terms': [x.to_dict()]}]}}


def test_old_snapshot_recovers_only_actual_ast_terms_without_prediction_or_refitting(monkeypatch):
    entry = saved_entry()
    original = deepcopy(entry)
    def no_science(*args, **kwargs):
        raise AssertionError('A paid fit must not be recomputed to save its method')
    monkeypatch.setattr(Program, 'predict', no_science)
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', no_science)
    result = attach_structural_research_program(entry, fit_rows=236,
        source_ref={'file': 'paid/selected_entry.json'})
    assert entry == original
    assert result['candidate'] is entry['candidate']
    assert result['structure'] == entry['structure']
    plan = result['research_program']
    assert plan['structural_replay']['coordinate_basis'] == 'saved_executable_terms'
    assert plan['source_fit_ref'] == {'file': 'paid/selected_entry.json'}
    assert plan['fit_rows'] == 236 and plan['search_compute'] == 471
    graph = ResearchGraph.from_dict(plan['graph'])
    terms = [graph.nodes[key].answer for key in plan['term_nodes']]
    assert [term.op for term in terms] == ['window_sum', 'event_delta']
    assert terms[0].children[0].children[1] == Program('const', value=2.)
    assert graph.nodes[plan['output_node']].answer == entry['candidate'].program
    assert plan['status'] == 'requires_fresh_validation'


def test_mismatched_saved_columns_cannot_become_a_replay_contract():
    entry = saved_entry()
    entry['structure']['fits'][0].update(fit_terms=[Program('feature', name='x').to_dict()],
                                       coefficients=[0., 1.])
    with pytest.raises(ValueError, match='reconstruct the saved executable'):
        structural_fit_contract(entry)


def test_existing_program_and_its_confirmation_are_retained():
    entry = saved_entry()
    entry['research_program'] = {'kind': 'existing', 'status': 'supported_on_fresh_trajectory'}
    assert attach_structural_research_program(entry, fit_rows=236) == entry
