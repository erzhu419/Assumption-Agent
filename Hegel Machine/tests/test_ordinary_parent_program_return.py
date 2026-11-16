"""Ordinary replay returns its actual parent and method, not an episode guess."""
from copy import deepcopy

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.method_action_research import (
    _existing_method_execution_return, read_method_reference,
)
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.transfer_partitions import replay_transfer_partition
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_parent_bound_staged_replay import fixture


def scene(tmp_path, monkeypatch):
    from types import SimpleNamespace
    body, training, feedback = fixture()
    parent = replay_transfer_partition(body['staged_research']['baseline'], training, SearchBudget(100000))
    parent['validation_loss'] = loss(parent['candidate'].program, feedback, 0., .5)
    recipe = body['staged_research']['continuation']
    names = tuple(training[0].features)
    template = AbstractTemplate('source', 'old', parent['candidate'].program, names, names, ())
    original = [('old', recipe, template, [dict(zip(names, names))])]
    frame = {'assessed': [parent], 'uncompared': [], 'comparison_complete': True,
        'terminal': False, 'comparison': feedback, 'fallback': 0.,
        'prepare_compute_cost': 0, 'replay_plans': original,
        'policy': SimpleNamespace(set_action_knowledge=lambda *args: None)}
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', lambda *a, **k: frame)
    agent = CommonObjectResearchAgent(ResearchConfig(max_candidates=4, compute_budget=100000),
        workspace=tmp_path, organization='pr3', source_task_ref={'target': 'current'})
    spec = TaskSpec('current', '', names, (), ())
    agent.prepare_research_decision(spec, training, feedback, SearchBudget(100000), [])
    return agent, frame, original, spec, parent, recipe, training, feedback


def actual_return(agent):
    record = agent._native_applications[-1]
    evidence = read_method_reference(record['evidence_ref'], agent.workspace)
    before = read_method_reference(evidence['before_entry_ref'], agent.workspace)['research_program']
    after = read_method_reference(evidence['after_entry_ref'], agent.workspace)['research_program']
    recognized = _existing_method_execution_return(evidence, before, after, agent.workspace)
    return record, evidence, recognized


def test_later_pool_winner_and_original_recipe_are_actual_replay_inputs(tmp_path, monkeypatch):
    agent, frame, original, spec, parent, recipe, training, feedback = scene(tmp_path, monkeypatch)
    initial = agent._native_revision_before_ref
    # Another real fitted parent wins later. Its source fit is retained as a
    # starting model; it is not declared paid on these current observations.
    other_body, other_rows, _ = fixture(radius=1.2)
    later = replay_transfer_partition(other_body['staged_research']['baseline'], other_rows, SearchBudget(100000))
    assert later['candidate'].program != parent['candidate'].program
    later['validation_loss'] = .001
    frame.update(assessed=[later], terminal=False, replay_plans=original)
    agent.prepare_research_decision(spec, training, feedback, SearchBudget(100000), [])
    method = frame['replay_plans'][0][1]
    binding = method['parent_parameter_revision']
    assert binding['parent_ref'] != initial
    assert read_method_reference(binding['parent_ref'], tmp_path)['candidate']['program'] == later['candidate'].program.to_dict()
    assert read_method_reference(binding['source_method_ref'], tmp_path) == recipe
    outputs = agent._replay_program_claims(spec, training, SearchBudget(100000), [],
        prepared=frame['replay_plans'][:1], feedback_rows=feedback)
    assert outputs and outputs[0]['transfer_partition_search']['reused_initial_fits'] == 0
    agent._assess(outputs, feedback, SearchBudget(1000), 0., .5)
    record, evidence, recognized = actual_return(agent)
    assert recognized is not None and recognized['execution_complete']
    assert evidence['research_parent_ref'] == binding['parent_ref']
    execution = read_method_reference(evidence['method_execution_ref'], tmp_path)
    assert execution['action'] == 'program_replay'
    assert execution['comparison']['old_pool_size'] == 1
    assert not recognized['proposal']['candidates']
    assert not list(tmp_path.glob('**/*pool*.json'))


@pytest.mark.parametrize('incomplete_search', [False, True])
def test_partial_execution_or_comparison_returns_unknown_not_new_method(tmp_path, monkeypatch, incomplete_search):
    agent, frame, _, spec, _, _, training, feedback = scene(tmp_path, monkeypatch)
    # Enough for one real fit but not the complete bounded search.
    budget = SearchBudget(3000 if incomplete_search else 100000)
    outputs = agent._replay_program_claims(spec, training, budget, [],
        prepared=frame['replay_plans'][:1], feedback_rows=feedback)
    assert outputs
    agent._assess(outputs, feedback, SearchBudget(1000 if incomplete_search else 0), 0., .5)
    _, evidence, recognized = actual_return(agent)
    assert recognized is not None and not recognized['execution_complete']
    assert recognized['proposal']['candidates'] == []
    execution = read_method_reference(evidence['method_execution_ref'], tmp_path)
    assert execution['full_comparison'] is incomplete_search
    assert execution['search_complete'] is not incomplete_search


def test_nonwinning_application_reaches_ordinary_topic_without_composition_producer(tmp_path, monkeypatch):
    agent, frame, _, spec, parent, _, training, feedback = scene(tmp_path, monkeypatch)
    outputs = agent._replay_program_claims(spec, training, SearchBudget(100000), [],
        prepared=frame['replay_plans'][:1], feedback_rows=feedback)
    agent._assess(outputs, feedback, SearchBudget(1000), 0., .5)
    result = {'status': 'returned', 'program': parent['candidate'].program.to_dict(),
        'research_program': deepcopy(parent['research_program']), 'costs': {'compute_cost': 0}}
    agent._record_actual_program_return(result)
    agent._consume_program_return(result, [])
    review = result['native_method_return']['review']['research_result']['method_action_research']
    assert len(review['existing_method_executions']) == 1
    existing = read_method_reference(review['existing_method_executions'][0]['return_ref'], tmp_path)
    assert existing['execution_complete']
    assert existing['proposal']['candidates'] == []
    assert not review.get('native_method_composition_refs')
    # Re-reading the same paid return does not repeat scientific fitting or
    # create another physical source; addressed application refs stay stable.
    original_ref = deepcopy(agent._native_applications[0]['return_ref'])
    agent._assess(outputs, feedback, SearchBudget(0), 0., .5)
    assert agent._native_applications[0]['return_ref'] == original_ref
    assert read_method_reference(original_ref, tmp_path)['execution_complete']


@pytest.mark.parametrize('complete_pool', [False, True])
def test_prepared_segment_comparison_uses_its_real_whole_pool_status(tmp_path, monkeypatch, complete_pool):
    agent, frame, _, spec, _, _, training, feedback = scene(tmp_path, monkeypatch)
    outputs = agent._replay_program_claims(spec, training, SearchBudget(100000), [],
        prepared=frame['replay_plans'][:1], feedback_rows=feedback)
    frame.update(spec=spec, independent_validation=False,
        comparison_complete=complete_pool, native_full_pool_comparison=complete_pool)
    compared = agent.compare_research_decision_return(frame, 'program_replay', outputs,
        SearchBudget(1000), [])
    _, _, recognized = actual_return(agent)
    assert recognized is not None
    assert recognized['execution_complete'] is complete_pool
    assert compared['full_comparison'] is complete_pool
    # Restored claims from a different episode are addressed objects, not
    # indices into this agent's transient application list.
    resumed = deepcopy(outputs)
    agent._native_applications = []
    assert agent._assess(resumed, feedback, SearchBudget(1000), 0., .5)
