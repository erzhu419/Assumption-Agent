"""The ordinary archive can revise its current winner without borrowing a fit."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import replay_transfer_partition
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from scripts import review_paid_object_pool as runner
from test_parent_bound_staged_replay import fixture
from test_paid_object_pool import paired_parameter_scene
from test_native_method_return_route import save


@pytest.mark.parametrize('organization', ['archive', 'fixed_pr2', 'pr3'])
def test_complete_pool_winner_becomes_optional_bound_plan_in_every_arm(tmp_path, monkeypatch, organization):
    body, training, feedback = fixture()
    parent = replay_transfer_partition(body['staged_research']['baseline'], training, SearchBudget(100000))
    parent['validation_loss'] = loss(parent['candidate'].program, feedback, 0., .5)
    recipe = body['staged_research']['continuation']
    names = tuple(training[0].features)
    template = AbstractTemplate('old', 'source', parent['candidate'].program, names, names, ())
    old = ('source', recipe, template, [dict(zip(names, names))])
    frame = {'assessed': [parent], 'uncompared': [], 'comparison_complete': True, 'terminal': False,
        'comparison': feedback, 'fallback': 0., 'prepare_compute_cost': 0,
        'replay_plans': [old], 'policy': SimpleNamespace(set_action_knowledge=lambda *args: None)}
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', lambda *args, **kwargs: frame)
    agent = CommonObjectResearchAgent(ResearchConfig(max_candidates=4, minimum_improvement=1e-7),
        workspace=tmp_path, organization=organization)
    spec = TaskSpec('current', '', names, (), ())
    actions = []
    prepared = agent.prepare_research_decision(spec, training, feedback, SearchBudget(100000), actions)
    bound = prepared['replay_plans'][0][1]
    assert prepared['replay_plans'][1:] == [old] and len(prepared['assessed']) == 1
    assert research_program_fit_contract(bound)['program'] == parent['candidate'].program
    assert bound['parent_parameter_revision']['new_parent_binding']
    assert actions[-1]['action'] == 'prepare_current_parent_parameter_replay'
    assert not actions[-1]['paid_initializer_reused']
    # Use the actual ordinary replay executor, not a benchmark-only caller.
    outputs = agent._replay_program_claims(spec, training, SearchBudget(100000), [],
        prepared=prepared['replay_plans'], feedback_rows=feedback)
    assert len(outputs) <= 4
    assert loss(outputs[0]['candidate'].program, training + feedback, 0., .5) < 1e-7
    assert outputs[0]['transfer_partition_search']['reused_initial_fits'] == 0
    assert recipe == old[1] and bound != recipe


def test_unfunded_optional_parent_plan_keeps_existing_queue(tmp_path, monkeypatch):
    body, training, feedback = fixture()
    parent = replay_transfer_partition(body['staged_research']['baseline'], training, SearchBudget(100000))
    parent['validation_loss'] = .1
    names = tuple(training[0].features)
    template = AbstractTemplate('old', 'source', parent['candidate'].program, names, names, ())
    old = ('source', body['staged_research']['continuation'], template, [dict(zip(names, names))])
    frame = {'assessed': [parent], 'uncompared': [], 'comparison_complete': True, 'terminal': False,
        'comparison': feedback, 'fallback': 0., 'prepare_compute_cost': 0, 'replay_plans': [old]}
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', lambda *args, **kwargs: frame)
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='archive')
    spec = TaskSpec('current', '', names, (), ())
    assert agent.prepare_research_decision(spec, training, feedback, SearchBudget(1), [])['replay_plans'] == [old]


def test_saved_precall_view_excludes_later_success_and_does_not_claim_current_fit(tmp_path, monkeypatch):
    request, prior, parent, rows, _ = paired_parameter_scene(tmp_path, monkeypatch)
    common = runner.load(prior / 'common/receipt.json')
    source = tmp_path / 'successor'
    source.mkdir()
    prior_frontier = runner.load(prior / 'retained_frontier.json')
    current_parent = prior_frontier['selected_ref']
    scores = [{'candidate_ref': ref, 'loss': item['mse']}
        for item in common['scores'] for ref in item['candidate_refs']]
    scores.append({'candidate_ref': current_parent, 'loss': prior_frontier['selected_mse']})
    save(source / 'registration.json', runner.load(prior / 'common/registration.json'))
    save(source / 'public_spec.json', runner.load(prior / 'common/public_spec.json'))
    save(source / 'prepared_state.json', {'comparison_ref': runner.ref(source / 'common_comparison.json'),
        'training_refs': common['data_refs'][:1], 'validation_refs': common['data_refs'][1:],
        'training_rows': len(rows) - 1, 'working_rows': 1,
        'public_spec_ref': runner.ref(source / 'public_spec.json')})
    save(source / 'pre_call_evidence.json', {'model_ref': current_parent,
        'score_loss': prior_frontier['selected_mse'], 'data_refs': common['data_refs'],
        'basis': {'full_pool_size': len(scores)}})
    save(source / 'common_comparison.json', {'scores': scores, 'data_refs': common['data_refs'],
        'full_comparison': True, 'rows': len(rows), 'complete_pool_size': len(scores),
        'pre_call_evidence_ref': runner.ref(source / 'pre_call_evidence.json')})
    save(source / 'receipt.json', {'selected_ref': {'file': 'later_success_not_a_source.json'}, 'loss': 0.})
    output = tmp_path / 'pre_call_view'
    restored = runner.restore_pre_call_parent(source, output)
    assert restored['selected_ref'] == current_parent
    frontier, _, recovered, _, refs, _, _, _ = runner.parent_inputs(output)
    assert len(refs) == len(scores) and recovered == rows
    assert frontier['appended_comparison_refs'] == []
    assert restored['parent_fit_scope'] == 'scored_existing_program_on_current_data'
    assert restored['new_fits'] == restored['new_predictions'] == restored['new_compute_cost'] == 0


def test_scored_old_source_is_not_passed_as_a_paid_current_initializer(tmp_path, monkeypatch):
    request, source, parent, rows, scores = paired_parameter_scene(tmp_path, monkeypatch)
    config = runner.load(request)
    actual_ref = config['previous_result_ref']
    actual = runner.read_method_reference(actual_ref, source)
    save(Path(actual_ref['file']), {**actual, 'parent_fit_scope': 'scored_existing_program_on_current_data'})
    output = tmp_path / 'parameter'
    cfg = runner.register_parameter(request, output)
    assert cfg['paid_initializer_cost_ref'] is None
    def run(agent, spec, retained, paid, budget, actions, **kwargs):
        assert 'paid_initial' not in kwargs
        assert kwargs['fit_rows'] == rows[:cfg['training_rows']]
        assert len(kwargs['finalists']) == runner.load(source / 'retained_frontier.json')['complete_current_pool_occurrences']
        return {'status': 'no_improvement', 'new_claims': [], 'compared_claims': [retained],
            'selected': retained, 'full_comparison': True, 'execution_complete': True,
            'action': {'research': {'execution_complete': True}}}
    monkeypatch.setattr(UnifiedResearchAgent, '_revise_paid_structure', run)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_parent(output)
    assert result['full_comparison'] and result['compute_cost'] == 0
