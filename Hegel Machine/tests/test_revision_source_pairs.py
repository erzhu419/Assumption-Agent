"""A registered source-pair frontier retains the full source intake and fees."""
from copy import deepcopy
from itertools import combinations
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import revision_relations as relations
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.theory_consolidation import _allocation


def source_records():
    records = []
    for index in range(11):
        records.append({
            'episode_id': f'endpoint:{index}',
            'source_id': 'physical:left' if index < 9 else 'physical:right',
            'before_ref': {'file': 'retained.json', 'json_path': [index, 'before']},
            'after_ref': {'file': 'retained.json', 'json_path': [index, 'after']},
            'before_program': Program('const', value=0.).to_dict(),
            'after_program': Program('const', value=float(index)).to_dict(),
            'role_signature': {}, 'output_unit': 'V', 'time_contract': ['current'],
            'context_ref': {'source': index}, 'context_family_ref': 'fixture_family',
        })
    return records


def selected_pairs(records):
    return [[records[left]['episode_id'], records[right]['episode_id']]
            for left in range(4, 9) for right in (9, 10)]


def parameter_binding_scene():
    records = source_records()
    preparation = SearchBudget(1000)
    prepared = [relations._prepare(record, Program.from_dict(record['after_program']),
                                  preparation, side='after') for record in records]
    candidate = relations._compile_revision([prepared[4], prepared[9]],
                                             'retained_pair', preparation)
    return candidate, prepared


def test_source_binding_denial_stays_unknown_without_entering_parameter_or_condition_search(monkeypatch):
    from hegel_machine.cognition import revision_parameter_relations as parameters

    candidate, prepared = parameter_binding_scene()
    old_context = {'id': 'old_context', 'kind': 'program_context_relation',
                   'definition': {'body': 'retained_without_parameter_learning'}}
    original = deepcopy((candidate, old_context, prepared))

    def unexpected_learning(*args, **kwargs):
        raise AssertionError('incomplete source binding cannot supply parameter or condition samples')

    monkeypatch.setattr(parameters, 'learn_parameter_relations', unexpected_learning)
    monkeypatch.setattr(parameters, 'learn_joint_parameter_relations', unexpected_learning)
    budget = SearchBudget(1)
    searches, candidates = relations._learn_parameter_updates(
        [old_context, candidate], prepared, budget, {})
    assert budget.spent == 1
    search, = searches
    assert search['status'] == 'UNKNOWN' and search['budget_exhausted']
    assert search['reason'] == 'source_binding_budget_exhausted'
    assert search['source_record_count'] is None
    assert search['source_binding_compute_cost'] == budget.spent
    assert search['actual_compute_cost'] == 0 and not search['attempts']
    assert candidate['parameter_relation_search']['reason'] == search['reason']
    assert candidate['parameter_relation_search']['source_record_count'] is None
    assert candidates == [old_context, candidate]
    assert all(candidate[key] == value for key, value in original[0].items())
    assert old_context == original[1] and prepared == original[2]


@pytest.mark.parametrize('matched_count', [0, 9])
def test_complete_source_binding_keeps_real_empty_or_one_sided_sample_shortage(matched_count):
    candidate, prepared = parameter_binding_scene()
    for index, source in enumerate(prepared):
        if matched_count == 0 or index in (0, 10):
            source['record']['context_family_ref'] = 'another_family'
    budget = SearchBudget(1000)
    searches, candidates = relations._learn_parameter_updates([candidate], prepared, budget, {})
    search, = searches
    assert search['source_record_count'] == matched_count
    assert search['reason'] == 'two_physical_sources_with_training_and_held_revisions_required'
    assert search['budget_exhausted'] is False
    assert not search['attempts'] and not search['accepted']
    if matched_count:
        assert [record['source_id'] for record in search['source_records']].count('physical:right') == 1
    assert candidates == [candidate] and candidate['parameter_relations'] == []


def test_source_binding_denial_reaches_construction_status_without_relabeling_real_sample_shortage(monkeypatch):
    records = source_records()
    records[10]['context_family_ref'] = 'another_family'
    allocation = {'compute': 2000, 'max_groups': 1,
                  'source_pairs': [[records[4]['episode_id'], records[9]['episode_id']]]}
    frontier = {}

    def retain_construction(candidates, sources, budget, allocation):
        frontier['spent'] = budget.spent
        return [], candidates

    with monkeypatch.context() as patch:
        patch.setattr(relations, '_learn_parameter_updates', retain_construction)
        constructed = relations.consolidate_revision_relations(
            records, allocation=allocation, object_type='program_revision_relation')
    assert constructed['status'] == 'completed' and not constructed['budget_exhausted']
    cut = relations.consolidate_revision_relations(records,
        allocation={**allocation, 'compute': frontier['spent'] + 1},
        object_type='program_revision_relation')
    assert cut['status'] == 'UNKNOWN' and cut['budget_exhausted']
    assert cut['actual_compute_cost'] == frontier['spent'] + 1
    search, = cut['parameter_relation_searches']
    assert search['reason'] == 'source_binding_budget_exhausted'
    assert search['source_record_count'] is None and not search['attempts']
    assert all(cut['candidates'][0][key] == value
               for key, value in constructed['candidates'][0].items())

    completed = relations.consolidate_revision_relations(
        records, allocation=allocation, object_type='program_revision_relation')
    assert completed['status'] == 'completed' and not completed['budget_exhausted']
    search, = completed['parameter_relation_searches']
    assert search['source_record_count'] == 10
    assert search['reason'] == 'two_physical_sources_with_training_and_held_revisions_required'
    assert not search['budget_exhausted'] and not search['attempts']


def compile_spy(monkeypatch):
    """Charge a fixed construction unit; never predict, fit, or infer a law."""
    entered, completed = [], []

    def compile_pair(sources, identifier, budget):
        pair = [source['record']['episode_id'] for source in sources]
        entered.append(pair)
        relations._spend(budget, 7)
        completed.append(pair)
        return None

    monkeypatch.setattr(relations, '_compile_revision', compile_pair)
    return entered, completed


@pytest.mark.parametrize('reverse', [False, True])
def test_exact_ten_pairs_keep_all_eleven_preparations_and_declared_order(monkeypatch, reverse):
    records = source_records()
    original = deepcopy(records)
    pairs = selected_pairs(records)
    if reverse:
        pairs.reverse()
    entered, completed = compile_spy(monkeypatch)
    prepare = relations._prepare
    prepared = []

    def record_preparation(record, program, budget, **kwargs):
        prepared.append(record['episode_id'])
        return prepare(record, program, budget, **kwargs)

    monkeypatch.setattr(relations, '_prepare', record_preparation)
    result = relations.consolidate_revision_relations(records, allocation={
        'compute': 1000, 'max_groups': 10, 'source_pairs': pairs},
        object_type='program_revision_relation')
    assert prepared == [record['episode_id'] for record in records for _ in range(2)]
    assert entered == completed == pairs
    assert result['actual_compute_cost'] == 33 + 10 * (1 + 7)
    assert not result['budget_exhausted']
    assert [item['episode_ids'] for item in result['excluded']] == pairs
    assert records == original


def test_omitted_selector_keeps_original_combinations_and_eight_pair_boundary(monkeypatch):
    records = source_records()
    entered, completed = compile_spy(monkeypatch)
    result = relations.consolidate_revision_relations(records, allocation={
        'compute': 1000, 'max_groups': 8}, object_type='program_revision_relation')
    expected = [[records[left]['episode_id'], records[right]['episode_id']]
                for left, right in combinations(range(11), 2)
                if records[left]['source_id'] != records[right]['source_id']][:8]
    assert entered == completed == expected
    # The ninth compatible pair is visited before the unchanged max_groups stop.
    assert result['actual_compute_cost'] == 33 + 39 + 8 * 7
    assert result['budget_exhausted'] and len(result['excluded']) == 38


def test_pair_selection_keeps_actual_budget_denial_and_partial_result(monkeypatch):
    records = source_records()
    pairs = selected_pairs(records)
    entered, completed = compile_spy(monkeypatch)
    result = relations.consolidate_revision_relations(records, allocation={
        'compute': 48, 'max_groups': 10, 'source_pairs': pairs},
        object_type='program_revision_relation')
    assert entered == pairs[:2] and completed == pairs[:1]
    assert result['budget_exhausted'] and result['status'] == 'UNKNOWN'
    assert result['actual_compute_cost'] == 42
    assert [item['episode_ids'] for item in result['excluded']] == pairs[:1]


@pytest.mark.parametrize('mismatch', ['same_source', 'output_unit'])
def test_selected_pair_still_obeys_original_source_and_unit_contracts(monkeypatch, mismatch):
    records = source_records()
    left, right = (0, 1) if mismatch == 'same_source' else (4, 9)
    if mismatch == 'output_unit':
        records[right]['output_unit'] = 'W'
    entered, _ = compile_spy(monkeypatch)
    pair = [records[left]['episode_id'], records[right]['episode_id']]
    result = relations.consolidate_revision_relations(records, allocation={
        'compute': 1000, 'max_groups': 10, 'source_pairs': [pair]},
        object_type='program_revision_relation')
    assert not entered and result['actual_compute_cost'] == 34
    assert result['excluded'] == [{'episode_ids': pair, 'status': 'UNKNOWN',
        'reason': ('distinct_episode_sources_required' if mismatch == 'same_source'
                   else 'incompatible_output_unit_or_context')}]


def test_unavailable_prepared_endpoint_is_not_replaced_or_silently_omitted(monkeypatch):
    records = source_records()
    records[9].pop('output_unit')
    entered, _ = compile_spy(monkeypatch)
    pair = selected_pairs(records)[0]
    result = relations.consolidate_revision_relations(records, allocation={
        'compute': 1000, 'max_groups': 10, 'source_pairs': [pair]},
        object_type='program_revision_relation')
    assert not entered and result['actual_compute_cost'] == 31
    assert result['status'] == 'UNKNOWN'
    assert result['excluded'][-1] == {'episode_ids': pair, 'status': 'UNKNOWN',
                                    'reason': 'requested_pair_source_unavailable'}


def test_common_allocation_preserves_exact_pairs_and_original_funding_clip():
    agent = SimpleNamespace(research_mission={'purpose': 'theory_learning',
        'theory_learning_allocation': {'compute': 75000},
        'resource_usage': {'compute': 14564}}, config=SimpleNamespace(compute_budget=1800000),
        costs={'compute_cost': 1356559})
    requested = {'allocation': {'compute': 25000, 'max_groups': 10,
                                'source_pairs': selected_pairs(source_records())}}
    result = _allocation(agent, requested)
    assert result == requested['allocation']
    result['source_pairs'][0].reverse()
    assert result['source_pairs'] != requested['allocation']['source_pairs']
    assert agent.research_mission['resource_usage']['compute'] == 14564


def test_common_operator_consumes_pairs_and_keeps_them_in_its_cache_basis(tmp_path, monkeypatch):
    from hegel_machine.cognition import theory_consolidation
    from test_theory_consolidation import agent_scene, sources

    agent, lab = agent_scene(tmp_path)
    records = source_records()
    monkeypatch.setattr(theory_consolidation, '_revision_entries',
                        lambda *args: deepcopy(records))
    entered, completed = compile_spy(monkeypatch)
    pairs = selected_pairs(records)
    arguments = {'operator': 'theory_consolidation', 'theme': 'registered_ten',
        'object_type': 'program_revision_relation', 'theories': sources(),
        'source_theory_ids': ['source_a', 'source_b'],
        'mission': {'purpose': 'theory_learning',
                    'theory_learning_allocation': {'compute': 20000}},
        'allocation': {'compute': 1000, 'max_groups': 10, 'source_pairs': pairs}}
    result = agent._execute('run', deepcopy(arguments), lab)
    assert entered == completed == pairs
    assert result['new_compute_cost'] == 113
    saved = agent.theory_consolidation['executions']['registered_ten']
    assert saved['request_basis']['allocation']['source_pairs'] == pairs
    before_compute = agent.costs['compute_cost']
    repeated = agent._execute('run', deepcopy(arguments), lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == before_compute
    assert entered == pairs
    arguments['allocation']['source_pairs'] = list(reversed(pairs))
    reordered = agent._execute('run', arguments, lab)
    assert not reordered['cache_hit'] and reordered['new_compute_cost'] == 113
    assert entered == pairs + list(reversed(pairs))
    assert not agent.observations and agent.costs['model_calls'] == 0 and lab.paid == 0
