"""Restored first-selection evidence cannot become a gratuitous research rerun."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from scripts.accept_partial_role_structural_seed import recover_paid_scene, run_segment


def saved_report():
    program = Program('feature', name='q').to_dict()
    old_parent = Program('const', value=10.).to_dict()
    groups = [{'object_id': 'a', 'q': 'q'}]
    graph = {'fit_rows': 2, 'output_node': 'answer', 'term_nodes': ['q'],
        'graph': {'nodes': {'q': {'answer': program},
            'answer': {'answer': program, 'loss': .0625}}}}
    rows = [{'features': {'q': value}, 'target': value + .25, 'history': []}
            for value in (0., 1., 2.)]
    return {'task_id': 'first_registered_task',
        'actions': [
            {'action': 'adopt_discovered_objects', 'groups': groups, 'object_model_id': 'objects'},
            {'action': 'research_action_feedback', 'baseline_program': program,
             'baseline_kind': 'partial_object:window_3_totals', 'before_loss': .0625,
             'fitting_rows': 2, 'evidence_rows': 1, 'evidence_basis': 'common_unfitted_validation',
             'alternatives': [{'revision': {'original_program': old_parent}}]},
            {'action': 'choose_research_action', 'selected_action': 'structural_research',
             'compute_remaining': 2000},
            {'action': 'compare_structural_explanations', 'features': ['q']},
            {'action': 'execute_dependent_research', 'parent_program': old_parent,
             'parent_candidate_id': 'actual_bound_parent'},
            {'action': 'compare_research_directions', 'alternatives': [
                {'kind': 'retained_response', 'validation_loss': 60.0625}]},
            {'action': 'propose_mechanism_revisions', 'alternatives': [{'revision': {
                'original_program': program, 'original_research_program': graph}}]}],
        'batches': [{'observations': rows[:2], 'transitions': []},
                    {'observations': rows[2:], 'transitions': []},
                    {'observations': [{'features': {'q': 7.}, 'target': 999., 'history': []}]}],
        'object_research': {'model_id': 'objects', 'support_rows': 2, 'acquisition_rows': 2,
            'groups': groups, 'schema': {'objects': ['a'], 'fields': {'a': {'q': 'q'}}}},
        'object_continuation_state': {'selected': {'candidate': {'program': Program('const', value=999.).to_dict()}},
            'candidate_frontiers': [{'entries': [{'kind': 'partial_object:window_3_totals',
                'candidate': {'program': Program('const', value=888.).to_dict()},
                'partial_object_knowledge': {'source_model_id': 'objects', 'used_fields': ['q']}}]}]}}


def test_paid_scene_uses_original_graph_and_split_not_later_parent_or_rows():
    report = saved_report()
    before = deepcopy(report)
    scene = recover_paid_scene(report)
    assert [row.target for row in scene['training']] == [.25, 1.25]
    assert [row.target for row in scene['validation']] == [2.25]
    assert scene['baseline']['candidate'].program == Program('feature', name='q')
    assert scene['baseline']['research_program']['fit_rows'] == 2
    assert scene['references']['original_research_program'] == [
        'actions', 6, 'alternatives', 0, 'revision', 'original_research_program']
    assert scene['parent']['candidate'].candidate_id == 'actual_bound_parent'
    assert scene['references']['validation_batch'] == 1
    assert report == before


def test_equal_native_first_selection_stops_before_any_structural_fit(monkeypatch):
    from hegel_machine.cognition import unified_structure

    def unexpected_research(*args, **kwargs):
        raise AssertionError('The unchanged first decision must not repeat its fitted search')

    monkeypatch.setattr(unified_structure, 'propose_structures', unexpected_research)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    scene = recover_paid_scene(saved_report())
    scene['parent'] = deepcopy(scene['baseline'])
    result = run_segment(scene, agent)
    assert result['status'] == 'no_changed_decision'
    assert result['structural_research_executed'] is False
    assert result['new_candidate_count'] == result['old_fits_repeated'] == 0
    assert result['observation_cost'] == result['measurement_cost'] == 0
    assert result['compute_cost'] == result['restoration_scoring_compute'] == 6
    assert result['actions'][0]['selected_kind'] == 'partial_object:window_3_totals'
    assert result['original_loss'] == result['restored_loss'] == .0625


def test_actual_bound_parent_competes_and_native_proposal_receives_original_graph(monkeypatch):
    from hegel_machine.cognition import unified_structure

    scene = recover_paid_scene(saved_report())
    received = []

    def propose(spec, training, budget, *, parent, **kwargs):
        received.append(parent)
        assert budget.spend(10)
        fitted = Program('add', children=(Program('feature', name='q'), Program('const', value=.25)))
        return [{'candidate': Candidate('new_structure', fitted, 'structural'),
                 'kind': 'structural:test', 'family_id': 'T08', 'training_loss': 0.}]

    monkeypatch.setattr(unified_structure, 'propose_structures', propose)
    result = run_segment(scene, UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7)))
    assert result['status'] == 'changed_seed_researched'
    assert received[0]['research_program'] == scene['baseline']['research_program']
    assert result['original_parent']['candidate']['candidate_id'] == 'actual_bound_parent'
    assert result['new_candidate_count'] == 1 and result['best_new_loss'] == 0.
    assert result['improved'] is True
    assert result['compute_cost'] == 17  # Two restored scores, one actual proposal, one validation.
