"""Actual Common startup repairs preserve mature dynamics and paid comparisons."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_native_representation_routing import PublicOnlyLab


def scene(tmp_path, *, validation_flip=False):
    def threshold(value):
        return Program('ge', children=(Program('feature', name='u'),
                                        Program('const', value=value)))

    mature = Program('if', children=(Program('ge', children=(
        Program('lag', name='__target__', lag=1), Program('const', value=.5))),
        threshold(.2), threshold(.6)))
    original = Program('state_replay', children=(Program('if', children=(
        Program('available', lag=1), mature, threshold(.1))),))
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=replace(InquiryLLMConfig(), compute_budget=20000, max_model_calls=0))
    sequences = [[.4, .4, .8, .3, .1], [.8, .4, .1, .7, .3],
                 [-.2, .8, .3, .1, .4], [-.3, .9, .4, .1, .8]]
    agent.rows, agent.observations = [], []
    for index, values in enumerate(sequences):
        history, state, rows = [], 0., []
        for value in values:
            state = float(value >= (.2 if state else .6))
            rows.append({'features': {'u': value}, 'history': list(history),
                         'target': 1. - state if index == 3 and validation_flip else state})
            history.append({'u': value})
        start = len(agent.rows)
        agent.observations.append({'action': {'kind': 'trajectory', 'channels': [],
            'sequence': [{'u': value} for value in values]}, 'rows': rows,
            'row_indices': list(range(start, start + len(rows)))})
        agent.rows.extend(rows)
    agent._save('old.py', export_temporal_program(original))
    arguments = {'operator': 'research_route', 'research_question': {
        'id': 'startup:old', 'kind': 'program_revision', 'parent_id': 'task',
        'question': 'Revise the model using the paid reset counterexamples', 'depends_on': [],
        'origin': {'kind': 'fresh_confirmation_failure', 'row_indices': [0]},
        'answer_type': 'program', 'use_sites': [
            {'kind': 'supply_component', 'model': 'old.py', 'slot': 'representation'}]},
        'theories': [{'id': 'old', 'program': original.to_dict()},
                    {'id': 'constant', 'program': Program('const', value=0.).to_dict()}],
        'native_representation_plan': {'parent_theory_ids': ['old', 'constant'],
            'validation_records': [3], 'compute_limit': 10000, 'max_candidates': 4}}
    return agent, original, arguments


def test_common_completes_local_repair_and_reuses_its_validation_scores(tmp_path, monkeypatch):
    agent, original, arguments = scene(tmp_path)
    old_source = (agent.workspace / 'old.py').read_bytes()

    def broad_search(*args, **kwargs):
        pytest.fail('a completed local zero-error repair needs no broad re-search')

    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims', broad_search)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', broad_search)
    assessed = Counter()
    assess = UnifiedResearchAgent._assess

    def count_assess(self, claims, *args, **kwargs):
        for claim in claims:
            assessed[claim['candidate'].candidate_id] += 1
        return assess(self, claims, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, '_assess', count_assess)
    result = agent._execute('run', arguments, PublicOnlyLab())
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    selected = native['selected']
    assert selected['training_mse'] == selected['validation_mse'] == selected['development_mse'] == 0.
    assert selected['best_retained_validation_mse'] == 0.  # Validation alone would retain the old tie.
    assert selected['best_retained_development_mse'] > 0. and not selected['retained']
    assert [item['direction'] for item in native['search_directions']] == ['startup_revision']
    assert native['search_stop']['reason'] == 'precision_satisfied_after_local_repair'
    assert native['search_stop']['unexecuted_directions'] == ['discrete_state', 'retained_composition']
    assert not native['search_stop']['independent_confirmation']
    assert native['comparison_completed'] and native['all_parent_candidates_retained']
    assert all(count == 1 for count in assessed.values())
    direction = native['search_directions'][0]
    assert direction['validation_compute'] == len(native['candidates']) * 5
    assert selected['selection']['fitting_records'] == [0, 1, 2]
    assert selected['selection']['selection_records'] == [0, 1, 2, 3]
    assert selected['selection']['diagnostic_validation_records'] == [3]
    revised = Program.from_dict(selected['program'])
    assert revised.op == 'state_replay'
    assert revised.children[0].children[:2] == original.children[0].children[:2]
    assert revised.children[0].children[2] != original.children[0].children[2]
    assert all(item['state_research']['fit_record_indices'] == [0, 1, 2]
               for item in native['candidates'])
    namespace = {}
    exec((agent.workspace / result['selected_model']).read_text(), namespace)
    assert all(namespace['predict'](row) == row['target'] for row in agent.rows)
    assert (agent.workspace / 'old.py').read_bytes() == old_source
    assert set(result['retained_parent_theory_ids']) == {'old', 'constant'}
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
    assert agent.model is None and not result['parent_revision_completed']
    before = dict(agent.costs)
    repeated = agent._execute('run', arguments, PublicOnlyLab())
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert agent.costs == before


def test_validation_failure_keeps_broad_research_available_without_refitting_on_validation(tmp_path, monkeypatch):
    original_agent, _, original_arguments = scene(tmp_path / 'supported')
    first = original_agent._execute('run', original_arguments, PublicOnlyLab())
    original = load_native_representation_result(first['native_representation_ref'], original_agent.workspace)
    agent, _, arguments = scene(tmp_path / 'refuted', validation_flip=True)
    continued = []

    def state_search(*args, **kwargs):
        continued.append('discrete_state')
        return []

    def composition(*args, **kwargs):
        continued.append('retained_composition')
        return []

    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims', state_search)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', composition)
    result = agent._execute('run', arguments, PublicOnlyLab())
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert continued == ['discrete_state', 'retained_composition']
    assert 'search_stop' not in native
    assert native['comparison_completed']
    assert {Program.from_dict(item['program']) for item in native['candidates']} == {
        Program.from_dict(item['program']) for item in original['candidates']}
    assert all(item['training_mse'] == 0. and item['validation_mse'] > 0.
               for item in native['candidates'])
    assert native['selected']['selection']['independent_confirmation'] is False


@pytest.mark.parametrize('compute_limit, funded', [(300, True), (180, False)])
def test_complete_startup_entry_uses_unspent_total_when_equal_share_is_too_small(
        tmp_path, monkeypatch, compute_limit, funded):
    from hegel_machine.cognition import state_search

    agent, _, arguments = scene(tmp_path)
    arguments['native_representation_plan']['compute_limit'] = compute_limit
    execute = state_search.closed_startup_claims
    attempts = []

    def tracked(*args, budget, **kwargs):
        claims = execute(*args, budget=budget, **kwargs)
        attempts.append({'limit': budget.limit, 'spent': budget.spent,
                         'denied': budget.last_denied_request})
        return claims

    def broad(*args, **kwargs):
        if funded:
            pytest.fail('a funded zero-error local repair should finish before broad search')
        return []

    monkeypatch.setattr(state_search, 'closed_startup_claims', tracked)
    monkeypatch.setattr(state_search, 'closed_state_claims', broad)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', broad)
    result = agent._execute('run', arguments, PublicOnlyLab())
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    direction = native['search_directions'][0]
    assert direction['direction'] == 'startup_revision'
    assert attempts[0]['spent'] == 0 and attempts[0]['denied'] is not None
    assert native['compute_limit'] == compute_limit
    assert native['compute_cost'] <= compute_limit
    assert native['comparison_completed']
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    if funded:
        assert len(attempts) == 2 and attempts[1]['spent'] > 0
        reallocation = direction['allocation_revision']
        assert reallocation['initial_compute_limit'] == attempts[0]['limit']
        assert reallocation['declined_entry'] == attempts[0]['denied']
        assert reallocation['allocated_compute_limit'] == attempts[1]['limit']
        assert direction['compute_cost'] == attempts[1]['spent']
        assert native['selected']['development_mse'] == 0.
        assert native['research_completion'] == 'bounded_search_completed'
        assert native['search_stop']['completed_direction'] == 'startup_revision'
    else:
        assert len(attempts) == 1
        assert 'allocation_revision' not in direction
        assert native['research_completion'] == 'bounded_search_incomplete'


def test_eight_parent_1609_row_scene_funds_existing_startup_repair_inside_30000(tmp_path, monkeypatch):
    agent, original, arguments = scene(tmp_path)
    agent.config = replace(agent.config, compute_budget=30000)
    agent.observations, agent.rows = [], []
    patterns = [[.4, .4, .8, .3, .1], [.8, .4, .1, .7, .3],
                [-.2, .8, .3, .1, .4], [-.3, .9, .4, .1, .8]]
    # Many short reset trajectories keep the real 1609-row/8-parent entry
    # arithmetic without creating quadratic long-history fixture files.
    for index, length in enumerate([24] * 65 + [25, 24]):
        pattern = patterns[index % len(patterns)]
        history, state, rows = [], 0., []
        for step in range(length):
            value = pattern[step % len(pattern)] + index / 1_000_000
            state = float(value >= (.2 if state else .6))
            rows.append({'features': {'u': value}, 'history': list(history), 'target': state})
            history.append({'u': value})
        first = len(agent.rows)
        agent.observations.append({'action': {'kind': 'trajectory', 'channels': [],
            'sequence': history}, 'rows': rows,
            'row_indices': list(range(first, first + length))})
        agent.rows.extend(rows)
    parents = []
    guard, mature, _ = original.children[0].children
    for index in range(8):
        startup = Program('ge', children=(Program('feature', name='u'),
            Program('const', value=(index - 4) / 10)))
        program = Program('state_replay', children=(Program('if', children=(guard, mature, startup)),))
        parents.append({'id': f'parent{index}', 'program': program.to_dict()})
    arguments['theories'] = parents
    arguments['native_representation_plan'].update(parent_theory_ids=[item['id'] for item in parents],
        validation_records=[66], compute_limit=30000)

    def no_broad(*args, **kwargs):
        pytest.fail('this completed local repair must precede broad reconstruction')

    monkeypatch.setattr('hegel_machine.cognition.state_search.closed_state_claims', no_broad)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', no_broad)
    result = agent._execute('run', arguments, PublicOnlyLab())
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    direction, = native['search_directions']
    allocation = direction['allocation_revision']
    assert native['fitting_rows'] == 1585 and native['validation_rows'] == 24
    assert len(native['retained_comparison']) == 8
    assert allocation['initial_compute_limit'] == 5597
    assert allocation['declined_entry']['required_compute'] == 7925
    assert allocation['allocated_compute_limit'] == 16792
    assert native['compute_limit'] == 30000 and native['compute_cost'] <= 30000
    assert direction['compute_cost'] > 0 and direction['last_denied_request'] is None
    assert native['selected']['development_mse'] == 0.
    assert native['comparison_completed'] and native['all_parent_candidates_retained']
    assert native['research_completion'] == 'bounded_search_completed'
    assert native['search_stop']['completed_direction'] == 'startup_revision'
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0


def test_short_restored_question_history_cannot_overwrite_existing_round_one(tmp_path):
    agent, _, arguments = scene(tmp_path)
    first = agent._execute('run', arguments, PublicOnlyLab())
    changed = deepcopy(arguments)
    changed['native_representation_plan']['validation_records'] = [2]
    second = agent._execute('run', changed, PublicOnlyLab())
    assert len(second['exported_models']) >= 3
    saved = {entry['model']: (agent.workspace / entry['model']).read_bytes()
        for result in (first, second) for entry in result['exported_models']}
    assert all('/round_0001/' in entry['model'] for entry in second['exported_models'])
    owner = next(question for question in agent.questions if question['id'] == 'startup:old')
    owner['native_representation_research'] = owner['native_representation_research'][:1]
    changed['native_representation_plan']['validation_records'] = [1]
    third = agent._execute('run', changed, PublicOnlyLab())
    assert len(owner['native_representation_research']) == 2
    assert third['export_round_index'] == 2
    assert third['native_representation_ref']['file'].endswith('record_0002.json')
    assert all('/round_0002/' in entry['model'] for entry in third['exported_models'])
    assert all((agent.workspace / path).read_bytes() == source for path, source in saved.items())
    assert set(saved) <= set(agent.candidate_paths)
    assert len(agent.candidate_paths) == len(saved) + len(third['exported_models'])
