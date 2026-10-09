"""Return-only isolation and paid-prefix accounting for the demand study."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path

from hegel_machine.cognition.models import ResearchConfig
from test_scoped_retention_comparison import LinearLab


def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_workflow_demand_study.py'
    spec = importlib.util.spec_from_file_location('workflow_demand_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_intervention_removes_only_outcome_receipts_and_keeps_competing_workflows():
    study = runner()
    state = {'knowledge': {'research_workflows': {'methods': [
        {'method_id': 'method_0', 'nodes': [{'action': 'fit'}],
         'knowledge_outcomes': [{'retained_gain': 1.}]},
        {'method_id': 'method_1', 'nodes': [{'action': 'measure'}]}], 'next_method_index': 2},
        'concepts': [{'concept_id': 'learned', 'source_returns': [{'retained_normalized_gain': 1.}]}]},
        'results': [{'research': {'costs': {'compute_cost': 123}, 'batches': ['paid']}}],
        'pending_revisits': [{'consumer': 0}]}
    retained, removed = (study._fork_state(state, arm) for arm in study.ARMS)
    assert retained == state
    expected = deepcopy(state)
    expected['knowledge']['research_workflows']['methods'][0]['knowledge_outcomes'] = []
    assert removed == expected
    removed['knowledge']['research_workflows']['methods'][0]['nodes'].clear()
    assert retained == state


def test_actual_learning_has_shared_costs_and_all_training_precedes_matched_terminal():
    study = runner()
    events, evaluations = [], []
    plan = study.protocol(63009001)
    plan['return_tasks'] = [{'coefficients': (1., 0.), 'seed': 100},
                            {'coefficients': (1., 0.), 'seed': 101}]
    plan['competition_tasks'] = [{'coefficients': (0., 1.), 'seed': 102}]
    plan['tasks'] = [{'coefficients': (1., 0.), 'seed': 103},
                     {'coefficients': (1., 1.), 'seed': 104}]
    plan['research_config'] = asdict(ResearchConfig(observation_budget=96, compute_budget=24000,
        minimum_improvement=1e-7, max_rounds=1, seed=63009001))
    plan['measurement_budget'] = 0
    plan['evaluation'] = {'length': 8, 'trajectories': 1, 'seed_offset': 1000003}
    result = study.run_study(plan, lab_factory=lambda task:
        LinearLab(**task, events=events, evaluations=evaluations))
    start = events.index('evaluation')
    assert all(event == 'learning' for event in events[:start])
    assert events[start:] == ['evaluation'] * 7
    expected = Counter((task['seed'], task['seed'] + 1000003, 8, 1)
        for task in [*plan['return_tasks'], *plan['competition_tasks']])
    expected.update((task['seed'], task['seed'] + 1000003, 8, 1)
        for task in plan['tasks'] for _ in study.ARMS)
    assert Counter(evaluations) == expected
    assert all(action['action'] != 'replay_research_workflow'
        for action in result['partial_demand']['actions'])
    retained, removed = (result['results'][arm] for arm in study.ARMS)
    cleared = deepcopy(retained['initial_workflows'])
    for method in cleared['methods']:
        if 'knowledge_outcomes' in method:
            method['knowledge_outcomes'] = []
    assert removed['initial_workflows'] == cleared
    for key in study.COST_KEYS:
        assert result['common_prefix_costs'][key] == sum(
            phase['costs'][key] for phase in result['prefix'].values())
        for phase in [*result['prefix'].values(), *result['results'].values()]:
            paid = sum(task['costs'][key] for task in phase['tasks'])
            paid += phase['screening_compute_cost'] if key == 'compute_cost' else 0
            assert phase.get('task_sequence_costs', phase['costs'])[key] == paid
            assert phase['evaluation_costs'][key] == sum(
                task['evaluation_costs'][key] for task in phase['tasks'])
        for arm in result['results'].values():
            assert arm['costs'][key] == arm['task_sequence_costs'][key] + result['common_prefix_costs'][key]
            assert arm['continuation_costs'][key] + result['partial_demand']['costs'][key] == arm['task_sequence_costs'][key]
            assert arm['costs'][key] <= arm['resource_limits'][key]
    assert result['external_model_calls'] == 0
    # Lack of confirmed source credit is a recorded observation, not an execution failure.
    assert 'demand_matches_confirmed_return' in result
    assert 'producer_choice_changed' in result


def test_registered_cases_keep_public_protocol_fixed_and_new_seeds_unique():
    roster = runner().cohort_protocol()
    assert len(roster['cases']) == 12
    assert len({case['seed'] for case in roster['cases']}) == 12
    assert all(case['partial_probe_seed'] == case['return_tasks'][0]['seed'] for case in roster['cases'])
    assert all(case['tasks'][0]['seed'] != case['return_tasks'][0]['seed'] for case in roster['cases'])
    assert all(case['demand_collection'] == 'independent incoming partially researched consumer'
        for case in roster['cases'])
