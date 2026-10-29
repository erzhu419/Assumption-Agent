"""A witness repair reaches the same row objective as its complete-pool consumer."""

import pytest

from hegel_machine.cognition import representation_inquiry
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.representation_diagnostics import representation_ambiguity
from hegel_machine.cognition.research_programs import _Fitter
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_representation_counterexample_coordinates import setup


def test_export_cap_uses_row_weighted_development_not_best_old_region(monkeypatch):
    # No fitting is needed to isolate the selection failure: both nominees
    # already exist, one preserves the old region and the other wins overall.
    training = [Observation({'u': 0.}, 1., ({'u': float(index)},)) for index in range(5)]
    validation = [Observation({'u': 0.}, 0., ({'u': 9.},))]
    proposals = [{'candidate': Candidate(str(value), Program('const', value=value), 'saved'),
                  'training_loss': (1. - value) ** 2} for value in (0., 1.)]
    monkeypatch.setattr(_Fitter, 'fit', lambda *args: None)
    monkeypatch.setattr(representation_inquiry, '_witness_relation_claims',
                        lambda *args, **kwargs: proposals)
    budget, actions = SearchBudget(1000), []
    kept = representation_inquiry._witness_coordinate_claims(
        TaskSpec('selection', '', ('u',), (), ()), training,
        {'old': Program('const', value=2.)}, [[0, 1]], budget, actions,
        max_candidates=1, validation=validation, fallback=0.)
    assert len(kept) == 1 and kept[0]['candidate'].program.value == 1.
    assert kept[0]['validation_loss'] == 1.  # Old-region protection would choose 0.
    assert actions[0]['selection'] == 'full_exposed_development_mse_before_export_cap'
    assert actions[0]['selection_rows'] == {'training': 5, 'validation': 1}
    chosen = next(item for item in actions[0]['candidate_comparisons'] if item['exported'])
    assert chosen['development_mse'] == pytest.approx(1. / 6.)
    assert not actions[0]['independent_confirmation']
    assert budget.completed_fits == 0


def test_native_winner_and_export_selection_share_full_development_objective(tmp_path, monkeypatch):
    agent, _, arguments, lab = setup(tmp_path, monkeypatch)
    parent = Program('const', value=0.)
    for index, record in enumerate(agent.observations):
        for row in record['rows']:
            row['target'] = 1. if index < 4 else 0.
    agent.rows[0]['target'] = 2.  # An actual conflict in the constant parent view.
    agent._save('old.py', export_temporal_program(parent))
    arguments['theories'] = [{'id': 'old', 'program': parent.to_dict()}]
    arguments['research_question']['representation_diagnostic'] = representation_ambiguity(
        [0.] * len(agent.rows), agent.rows)

    def saved_candidates(spec, training, parents, groups, budget, actions, **kwargs):
        assert groups
        program = Program('const', value=1.)
        return [{'candidate': Candidate('paid_one', program, 'saved'),
                 'training_loss': loss(program, training, 0., .5)}]

    monkeypatch.setattr(representation_inquiry, '_witness_coordinate_claims', saved_candidates)
    # Isolate selection of these saved competitors; state learning has its own
    # integration controls and can now fit this fixture's binary training rows.
    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims',
                        lambda *args, **kwargs: [])
    result = agent._execute('run', arguments, lab)
    native = representation_inquiry.load_native_representation_result(
        result['native_representation_ref'], agent.workspace)
    selected = native['selected']
    assert not selected['retained'] and selected['validation_mse'] == 1.
    assert selected['best_retained_validation_mse'] == 0.
    assert selected['development_mse'] == pytest.approx(13. / 60.)
    assert selected['best_retained_development_mse'] == pytest.approx(51. / 60.)
    assert selected['selection']['selection_records'] == [0, 1, 2, 3, 4]
    assert selected['selection']['diagnostic_validation_records'] == [4]
    assert selected['selection']['fitting_records'] == [0, 1, 2, 3]
    assert selected['selection']['data_role'] == 'exposed_development'
    assert not selected['selection']['independent_confirmation']
    assert native['selection_objective']['weighting'] == 'uniform_rows'
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert (agent.workspace / 'old.py').read_text() == export_temporal_program(parent)

