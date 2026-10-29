"""A restored native decision shares its paid prelude, not its mutable search."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.research_programs import _Fitter, _entry, _plan
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_structural_neighbor_cells import paid_response


def structural_scene():
    rows, schema, response = paid_response()
    spec = TaskSpec('native_scene', 'Compare measured explanations', schema.feature_names, (), (),
        context={'object_schema': json.dumps(schema.to_dict())})
    fit_budget = SearchBudget(5000)
    fit = _Fitter(rows[:34], fit_budget, .5).fit(response)
    parent = _entry(fit, [], 34, fit_budget.spent)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_candidates=4))
    budget, actions = SearchBudget(50000), []
    frame = agent.prepare_research_decision(spec, rows[:34], rows[34:58], budget, actions,
        parent=parent, observation_remaining=0)
    return agent, frame, budget, actions


def test_actual_structure_segment_uses_shared_pool_and_never_reexecutes_prelude(monkeypatch):
    agent, frame, preparation, _ = structural_scene()
    assert not frame['terminal'] and frame['comparison_complete']
    assert frame['prepare_compute_cost'] == preparation.spent > 0
    assert 'structural_research' in frame['directions']
    old = deepcopy(frame['assessed'])
    original = deepcopy(frame)

    def no_prepare(*args, **kwargs):
        raise AssertionError('A paid native decision cannot repeat its prelude')

    monkeypatch.setattr(agent, 'prepare_research_decision', no_prepare)
    work, actions = SearchBudget(50000), []
    result = agent.execute_research_decision(frame, 'structural_research', work, actions)
    assert result['execution_complete'] and result['full_comparison']
    assert result['completion_scope'] == 'bounded_native_segment_and_full_pool_comparison'
    assert type(result['search_budget_denied']) is bool
    assert (result['last_denied_request'] is not None) == result['search_budget_denied']
    assert result['structure_search_complete'] == [
        {'candidate_id': item['candidate'].candidate_id,
         'search_complete': item['structure']['search_complete']}
        for item in result['new_claims'] if 'search_complete' in (item.get('structure') or {})]
    assert result['new_claims'] and result['search_compute_cost'] > 0
    assert result['compute_cost'] == work.spent <= work.limit
    assert result['selected']['validation_loss'] < 1e-7
    assert result['claims'][:len(old)] == old
    assert {k: v for k, v in frame.items() if k != 'policy'} == {
        k: v for k, v in original.items() if k != 'policy'}
    assert frame['policy'].__dict__ == original['policy'].__dict__
    assert result['quote']['old_comparison_compute'] == 0
    assert result['new_observations'] == 0 and not result['independent_confirmation']
    assert not any(a['action'] == 'choose_research_action' for a in actions)


def test_real_program_replay_refits_only_the_declared_saved_procedure(monkeypatch):
    term = Program('mul', children=(Program('feature', name='u'),) * 2)
    payload = _plan({'fit_terms': (term,), 'coefficients': [0., 1.],
        'program': term, 'loss': 0.}, [], 10, 20)
    template = SimpleNamespace(roles=('input',), source_features=('u',))
    plans = [('retained_source', payload, template, [{'input': 'u'}])]
    rows = [Observation({'u': x}, .25 + 1.8 * x * x) for x in (-3., -2., -1., 0., 1., 2., 3.)]
    validation = [Observation({'u': x}, .25 + 1.8 * x * x) for x in (-2.5, .5, 2.5)]
    spec = TaskSpec('replay', 'Refit a retained procedure', ('u',), (), ())
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    monkeypatch.setattr(agent, '_replay_program_queue', lambda *a, **k: plans)
    frame = agent.prepare_research_decision(spec, rows, validation, SearchBudget(10000), [])
    assert 'program_replay' in frame['directions']
    left_agent, left = deepcopy((agent, frame))
    right_agent, right = deepcopy((agent, frame))
    left['policy'].experience['test_branch'] = {'trials': 1}
    assert 'test_branch' not in right['policy'].experience
    assert left['policy']._store is left_agent.library.action_experience
    work = SearchBudget(2000)
    result = left_agent.execute_research_decision(left, 'program_replay', work, [])
    assert result['full_comparison'] and result['execution_complete']
    assert result['selected']['validation_loss'] < 1e-7
    assert result['new_claims'][0]['method_source_ids'] == ['retained_source']
    assert len(result['new_claims']) <= 4
    assert result['quote']['search_compute_limit'] <= 1000
    assert result['claims'][:len(left['assessed'])] == left['assessed']


def test_completed_structure_return_reuses_paid_scores_without_fit_or_prediction(monkeypatch):
    agent, frame, _, _ = structural_scene()
    paid = agent.execute_research_decision(frame, 'structural_research', SearchBudget(50000), [])
    assert paid['execution_complete'] and paid['new_claims']
    paid_scores = {item['candidate'].program: item['validation_loss'] for item in paid['new_claims']}

    def no_new_work(*args, **kwargs):
        raise AssertionError('A complete paid return cannot repeat fitting or prediction')

    monkeypatch.setattr(agent, '_execute_prepared_research_segment', no_new_work)
    monkeypatch.setattr(_Fitter, 'fit', no_new_work)
    monkeypatch.setattr('hegel_machine.cognition.unified_research.loss', no_new_work)
    work, actions = SearchBudget(0), []
    reused = agent.compare_research_decision_return(frame, 'structural_research',
        paid['new_claims'], work, actions, paid_scores=paid_scores, reused_paid_return=True)
    assert reused['status'] == 'completed_reused_paid_return'
    assert reused['execution_complete'] and reused['full_comparison'] and reused['reused_paid_return']
    assert reused['completion_scope'] == 'paid_native_return_and_full_pool_comparison'
    assert reused['claims'] == paid['claims']
    assert reused['selected'] == paid['selected']
    assert reused['claims'][:len(frame['assessed'])] == frame['assessed']
    assert work.spent == reused['compute_cost'] == reused['new_fit_compute_cost'] == 0
    assert reused['new_comparison_compute_cost'] == reused['search_compute_cost'] == 0
    assert actions[-1]['action'] == 'consume_paid_prepared_research_return'


def test_insufficient_new_comparison_allowance_preserves_pool_without_search(monkeypatch):
    agent, frame, _, _ = structural_scene()
    def no_search(*args, **kwargs):
        raise AssertionError('Search cannot consume the complete-comparison reserve')
    monkeypatch.setattr(agent, '_execute_prepared_research_segment', no_search)
    work = SearchBudget(16 * len(frame['comparison']))
    result = agent.execute_research_decision(frame, 'structural_research', work, [])
    assert result['status'] == 'complete_segment_unfunded'
    assert not result['execution_complete'] and work.spent == 0
    assert result['claims'] == frame['assessed']


def test_normal_research_view_uses_the_same_preparation(monkeypatch):
    rows = [Observation({'x': x}, 2. * x + 1.) for x in (-2., -1., 0., 2.)]
    validation = [Observation({'x': x}, 2. * x + 1.) for x in (-1.5, 1.5)]
    spec = TaskSpec('affine', 'Read measured response', ('x',), (), ())
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    original, frames = agent.prepare_research_decision, []
    def prepare(*args, **kwargs):
        frame = original(*args, **kwargs)
        frames.append(frame)
        return frame
    monkeypatch.setattr(agent, 'prepare_research_decision', prepare)
    work = SearchBudget(1000)
    result = agent._research_view(spec, rows, validation, work, [], [], ResearchGraph())
    assert len(frames) == 1 and frames[0]['terminal']
    assert result == frames[0]['assessed']
    assert work.spent == frames[0]['prepare_compute_cost']
    spent = work.spent
    repeated = agent.execute_research_decision(frames[0], 'structural_research', work, [])
    assert repeated['status'] == 'already_explained' and work.spent == spent


def retained_nonlinear_scene(*, independent=True, training_shift=0., validation_shift=0.):
    program = Program('mul', children=(Program('feature', name='x'),) * 2)
    training = [Observation({'x': x}, x * x + training_shift)
                for x in (-3., -2., -1., 1., 2., 3.)]
    validation = ([Observation({'x': x}, x * x + validation_shift) for x in (-2.5, .5, 2.5)]
                  if independent else [])
    parent = {'candidate': Candidate('retained_square', program, 'retained_response'),
              'kind': 'retained_response', 'family_id': 'T03', 'training_loss': 0.}
    spec = TaskSpec('retained', 'Compare an existing nonlinear answer', ('x',), (), ())
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    frame = agent.prepare_research_decision(spec, training, validation, SearchBudget(10000), [],
        parent=parent)
    return agent, frame


@pytest.mark.parametrize('independent', [True, False])
def test_complete_retained_answer_stops_both_entries_without_claiming_confirmation(monkeypatch, independent):
    agent, frame = retained_nonlinear_scene(independent=independent)
    old = deepcopy(frame['assessed'])
    assert frame['terminal'] and frame['comparison_complete']
    assert frame['independent_validation'] is independent
    assert agent._best(old)['candidate'].candidate_id == 'retained_square'
    assert old[0]['validation_loss'] > agent.config.minimum_improvement  # Affine did not solve it.

    def no_search(*args, **kwargs):
        raise AssertionError('The completely compared existing answer needs confirmation, not another search')

    monkeypatch.setattr(agent, '_execute_prepared_research_segment', no_search)
    # A retained prepared frame may predate the completion flag repair.
    frame['terminal'] = False
    work, actions = SearchBudget(5000), []
    result = agent.execute_research_decision(frame, 'structural_research', work, actions)
    assert result['status'] == 'already_explained'
    assert result['execution_complete'] and result['full_comparison']
    assert result['claims'] == old and result['new_claims'] == []
    assert not result['independent_confirmation']
    assert work.spent == result['compute_cost'] == 0 and actions == []

    monkeypatch.setattr(agent, 'prepare_research_decision', lambda *a, **k: frame)
    ordinary = agent._research_view(frame['spec'], frame['training'], frame['validation'],
        work, actions, [], ResearchGraph(), parent=frame['parent'])
    assert ordinary == old and work.spent == 0 and actions == []


@pytest.mark.parametrize('training_shift,validation_shift', [(1., 0.), (0., 1.)])
def test_retained_fit_or_comparison_failure_still_allows_research(monkeypatch, training_shift, validation_shift):
    agent, frame = retained_nonlinear_scene(
        training_shift=training_shift, validation_shift=validation_shift)
    assert frame['comparison_complete'] and not frame['terminal']
    frame['directions'].append('program_replay')
    entered = []

    def search(prepared, action, budget, actions, **kwargs):
        entered.append(action)
        return [], SearchBudget(0)

    monkeypatch.setattr(agent, '_execute_prepared_research_segment', search)
    result = agent.execute_research_decision(frame, 'program_replay', SearchBudget(5000), [])
    assert entered == ['program_replay'] and result['status'] == 'completed'
    assert result['claims'] == frame['assessed']


def test_successful_scored_candidate_does_not_hide_an_incomplete_pool(monkeypatch):
    agent, frame = retained_nonlinear_scene()
    frame.update(terminal=False, comparison_complete=False,
        uncompared=[deepcopy(frame['assessed'][-1])])
    assert not agent._research_pool_explained(frame)

    def no_search(*args, **kwargs):
        raise AssertionError('Pending pool comparison must be resolved before a new segment')

    monkeypatch.setattr(agent, '_execute_prepared_research_segment', no_search)
    work = SearchBudget(5000)
    result = agent.execute_research_decision(frame, 'structural_research', work, [])
    assert result['status'] == 'preparation_comparison_unresolved'
    assert not result['execution_complete'] and not result['full_comparison']
    assert result['claims'] == [*frame['assessed'], *frame['uncompared']]
    assert work.spent == 0


def test_unknown_action_cannot_silently_dispatch_another_segment():
    agent, frame, _, _ = structural_scene()
    work = SearchBudget(5000)
    with pytest.raises(ValueError, match='Unsupported'):
        agent.execute_research_decision(frame, 'choose_a_winner', work, [])
    assert work.spent == 0


def test_unscored_prelude_candidates_remain_pending_and_are_not_dropped(monkeypatch):
    agent, original, _, _ = structural_scene()
    frame = agent.prepare_research_decision(original['spec'], original['training'],
        original['validation'], SearchBudget(len(original['training'])), [],
        parent=original['parent'])
    assert not frame['comparison_complete'] and frame['uncompared']
    def no_search(*args, **kwargs):
        raise AssertionError('An incomplete prelude cannot supply a complete matched start')
    monkeypatch.setattr(agent, '_execute_prepared_research_segment', no_search)
    work = SearchBudget(50000)
    result = agent.execute_research_decision(frame, 'structural_research', work, [])
    assert result['status'] == 'preparation_comparison_unresolved'
    assert not result['full_comparison'] and not result['execution_complete']
    assert frame['uncompared'][0] in result['claims'] and work.spent == 0
