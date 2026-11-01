"""Observed role/temporal changes become ordinary executable method proposals."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_programs import (
    _entry, propose_research_step_revisions, replay_research_program,
    research_program_replay_inputs,
)
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import (
    propose_transfer_partitions, quote_transfer_partition, transfer_partition_view,
)
from test_native_method_return_route import route, save
from test_transfer_partitions import observations, plan


def payload(terms, coefficients):
    program = _sum_terms(terms, coefficients[1:], coefficients[0])
    fit = {'program': program, 'fit_terms': terms, 'terms': terms,
           'coefficients': coefficients, 'loss': 0.}
    return _entry(fit, [], 20, 100)['research_program']


def binding_scene():
    before = payload([Program('feature', name=name) for name in ('a', 'c')], [0., 1., 0.])
    after = payload([Program('feature', name=name) for name in ('b', 'd')], [7., 99., 0.])
    template = AbstractTemplate('inputs', 'typed_source', Program('feature', name='value'),
        ('value', 'aux'), ('a', 'c'), (), object_roles={'value': (0, 'value'), 'aux': (0, 'aux')})
    schema = {'objects': ['A', 'B'],
              'fields': {'A': {'value': 'a', 'aux': 'c'}, 'B': {'value': 'b', 'aux': 'd'}}}
    before.update(input_template=template.to_dict(), source_schema=schema)
    after['source_schema'] = deepcopy(schema)
    rows = [Observation({'a': (i * 11) % 17, 'b': (i * 7) % 13, 'c': 0., 'd': 0.},
                        2. + 3. * ((i * 7) % 13)) for i in range(36)]
    return before, after, rows


def temporal_scene():
    x, zero = Program('feature', name='x'), Program('feature', name='zero')
    before = payload([x, zero], [0., 1., 0.])
    after = payload([Program('window_sum', lag=4, children=(x,)), zero], [7., 99., 0.])
    target = _sum_terms([Program('window_sum', lag=3, children=(x,))], [2.], .5)
    history, rows = [], []
    for index in range(48):
        frame = {'x': ((index * 13) % 19) / 7., 'zero': 0.}
        row = Observation(frame, 0., tuple(history))
        rows.append(replace(row, target=target.predict(row)))
        history.append(frame)
    return before, after, rows


def direction_scene():
    before = plan(dormant=True)
    fit = research_program_fit_contract(before)
    def direction(node, name):
        return replace(node, name=name) if node.op == 'neighbor_sum' else replace(
            node, children=tuple(direction(child, name) for child in node.children))
    # The successful source used right; a different target requires left.
    after = payload([direction(term, 'right') for term in fit['fit_terms']], [4., 7., 5., 0.])
    target = _sum_terms([direction(term, 'left') for term in fit['fit_terms']], [2., 3., 0.], .5)
    rows = []
    for index, row in enumerate(observations(radius=.4)):
        fields = {**row.features, 'x0': .2, 'x1': (-1 if index % 2 else 1) * .2 + .2}
        measured = replace(row, features=fields)
        rows.append(replace(measured, target=target.predict(measured)))
    return before, after, rows


@pytest.mark.parametrize('scene,kind', [(binding_scene, 'Pi2'), (temporal_scene, 'Pi3')])
def test_normal_proposal_and_replay_changes_actual_fit_with_complete_columns(scene, kind):
    before, after, rows = scene()
    original = deepcopy(before)
    proposed = propose_research_step_revisions(before, rows,
        revision_evidence={'after': after, 'before_ref': 'paid-before', 'after_ref': 'paid-after'})
    candidate, = proposed['candidates']
    body = candidate['research_program']
    assert body['transfer_partition']['kind'] == kind
    assert body['transfer_partition']['max_attempts'] <= 4
    quote = quote_transfer_partition(body, rows)
    budget = SearchBudget(quote['required_compute'])
    selected = replay_research_program(body, rows, budget)
    baseline = replay_research_program(proposed['baseline_research_program'], rows, SearchBudget(10000))
    assert selected['training_loss'] < 1e-12 < baseline['training_loss']
    fit = research_program_fit_contract(selected['research_program'])
    assert len(fit['fit_terms']) == 2 and fit['coefficients'][-1] == 0.
    assert fit['coefficients'][1] == pytest.approx(3. if kind == 'Pi2' else 2., abs=1e-7)
    assert before == original
    assert selected['transfer_partition_search']['completed']
    assert selected['transfer_partition_search']['compute_cost'] == budget.spent <= quote['required_compute']
    assert candidate['replay_inputs'] != proposed['baseline_replay_inputs']
    assert selected['transfer_partition_search']['attempts'] <= 4
    if kind == 'Pi3':
        assert fit['fit_terms'][0].lag == 3  # source after used 4; target data decides


@pytest.mark.parametrize('scene,kind', [(binding_scene, 'Pi2'), (temporal_scene, 'Pi3'),
                                     (direction_scene, 'Pi3')])
def test_actual_native_return_route_publishes_the_same_executable_edit(tmp_path, scene, kind):
    before, after, rows = scene()
    source = {'physical_source': 'actual_local_research'}
    evidence = save(tmp_path / 'revision.json', {'source_task_ref': source,
        'before_entry_ref': save(tmp_path / 'before.json', {'research_program': before}),
        'after_entry_ref': save(tmp_path / 'after.json', {'research_program': after})})
    actual = save(tmp_path / 'returned.json', {'status': 'completed',
        'execution_complete': True, 'revision_evidence_ref': evidence})
    common = CommonObjectDependencyMethods(tmp_path / 'common', [],
        budget=SearchBudget(100000), actions=[]).common
    record_native_parent_research_return(common, source_task_ref=source, result_ref=actual,
        mission={'purpose': 'method_learning', 'allocation': {'compute': 100000}})
    result = route(common, allowance=100000)
    published, = result['method_action_research']['transfer_partition_proposals']
    candidate = read_method_reference(published['candidate_refs'][0], common.workspace)
    body = candidate['research_program']
    assert body['transfer_partition']['kind'] == kind
    selected = replay_research_program(body, rows, SearchBudget(100000))
    assert selected['training_loss'] < 1e-12
    assert route(common, allowance=100000)['cache_hit']
    assert common.costs['observation_cost'] == common.costs['model_calls'] == 0


def test_observed_direction_revision_searches_target_direction_and_keeps_symbol_binding():
    before, after, rows = direction_scene()
    proposal = propose_transfer_partitions(before, after)
    body, = [item['research_program'] for item in proposal['candidates']
        if item['research_program']['transfer_partition']['kind'] == 'Pi3']
    assert len(body['transfer_partition']['alternatives']) == 2
    result = replay_research_program(body, rows, SearchBudget(100000))
    assert result['training_loss'] < 1e-12
    fit = research_program_fit_contract(result['research_program'])
    assert len(fit['fit_terms']) == 3 and fit['coefficients'][-1] == 0.
    assert any(term.name == 'left' for term in fit['fit_terms'])
    assert transfer_partition_view(result['research_program'])['status'] == 'ready'
    parameter, = result['research_program']['selection_steps'][0]['parameters']
    old = before['selection_steps'][0]['parameters'][0]
    assert parameter['symbol'] == old['symbol'] and parameter['radius'] == old['radius']
    assert parameter['relation'] == 'left' and parameter['radius_intervals'] == []
    assert parameter['prior_radius_intervals'] == old['radius_intervals']

    # The actual interval and kernel consumers must not promote the old
    # direction's supported interval into evidence for this new direction.
    from hegel_machine.cognition.relation_kernel_priors import recall_relation_kernels
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    research = RelationIntervalResearch(result['candidate'].program, [], SearchBudget(10000),
        research_program=result['research_program'])
    assert research.parameters[0]['relation'] == 'left'
    assert research.parameters[0]['radius_intervals'] == []
    assert research.parameters[0]['prior_radius_intervals'] == old['radius_intervals']
    symbolic = Program.from_dict(result['research_program']['selection_steps'][0]['parameter_template'])
    recalled = recall_relation_kernels(symbolic, [parameter], [{
        'status': 'supported', 'source_concept_id': 'saved-discrete-revision',
        'research_program': result['research_program']}], SearchBudget(10000))
    assert recalled == {'r': []}


def test_direction_revision_preserves_other_equal_radius_group_and_source():
    before = plan(other=True, dormant=True)
    original = deepcopy(before)
    fit = research_program_fit_contract(before)
    revised = list(fit['fit_terms'])
    revised[1] = replace(revised[1], name='right')
    after = payload(revised, fit['coefficients'])
    body, = [item['research_program'] for item in propose_transfer_partitions(before, after)['candidates']
        if item['research_program']['transfer_partition']['kind'] == 'Pi3']
    revised[1] = replace(revised[1], name='left')
    target = _sum_terms(revised, [2., 3., .7, 0.], .5)
    rows = []
    for i, row in enumerate(observations(radius=.4, other=True)):
        row = replace(row, features={**row.features,
            'x1': (-1 if i % 2 else 1) * .2})
        rows.append(replace(row, target=target.predict(row)))
    result = replay_research_program(body, rows, SearchBudget(100000))
    assert result['training_loss'] < 1e-12
    parameters = result['research_program']['selection_steps'][0]['parameters']
    assert parameters[0]['relation'] == 'left' and parameters[0]['radius_intervals'] == []
    assert parameters[1] == before['selection_steps'][0]['parameters'][1]
    assert before == original


def test_no_observed_revision_or_untyped_binding_does_not_open_new_research():
    before, after, _ = binding_scene()
    assert not propose_transfer_partitions(before, deepcopy(before))['candidates']
    unrelated = deepcopy(after)
    unrelated['source_schema']['objects'] = ['B', 'A']
    assert not propose_transfer_partitions(before, unrelated)['candidates']
    del before['input_template']
    assert not propose_transfer_partitions(before, after)['candidates']
    before, after, _ = temporal_scene()
    # A new fitted term on unrelated measurements is not evidence for a temporal edit.
    after = payload([Program('feature', name='other'), Program('feature', name='zero')], [0., 1., 0.])
    assert not propose_transfer_partitions(before, after)['candidates']


def test_incomplete_compute_is_not_a_negative_and_frozen_caps_apply():
    before, after, rows = temporal_scene()
    body = propose_transfer_partitions(before, after)['candidates'][0]['research_program']
    selected = replay_research_program(body, rows, SearchBudget(4 * len(rows)))
    assert selected is not None
    summary = selected['transfer_partition_search']
    assert summary['status'] == 'compute_unresolved' and not summary['completed']
    assert summary['remaining'] > 0 and not summary['independent_support']
    body['transfer_partition']['max_attempts'] = 1
    selected = replay_research_program(body, rows, SearchBudget(100000))
    assert selected['transfer_partition_search']['attempts'] == 1
    assert selected['transfer_partition_search']['remaining'] == 2


def test_target_feature_binding_reaches_alternatives_and_method_identity():
    before, after, rows = temporal_scene()
    body = propose_transfer_partitions(before, after)['candidates'][0]['research_program']
    mapping = {'x': 'measured', 'zero': 'unused'}
    rebound = [Observation({mapping[k]: v for k, v in row.features.items()}, row.target,
        tuple({mapping[k]: v for k, v in frame.items()} for frame in row.history)) for row in rows]
    result = replay_research_program(body, rebound, SearchBudget(100000), feature_mapping=mapping)
    assert result['training_loss'] < 1e-12
    inputs = research_program_replay_inputs(body, feature_mapping=mapping)
    assert 'measured' in str(inputs['transfer_partition']['alternatives'])
    changed = deepcopy(body)
    changed['transfer_partition']['alternatives'] = changed['transfer_partition']['alternatives'][:1]
    assert research_program_replay_inputs(body) != research_program_replay_inputs(changed)


def test_typed_ordinary_call_binds_newly_addressed_objects_on_renamed_successor():
    before, after, rows = binding_scene()
    body = propose_transfer_partitions(before, after)['candidates'][0]['research_program']
    template = AbstractTemplate.from_dict(body['input_template'])
    assert set(template.source_features) == {'a', 'b', 'c', 'd'}
    schema = {'objects': ['U', 'V'],
        'fields': {'U': {'value': 'u', 'aux': 'unused_u'},
                   'V': {'value': 'v', 'aux': 'unused_v'}}}
    mapping = next(template.mappings(['u', 'v', 'unused_u', 'unused_v'], object_schema=schema))
    binding = dict(zip(template.source_features, (mapping[role] for role in template.roles)))
    rebound = [Observation({binding[k]: v for k, v in row.features.items()}, row.target) for row in rows]
    result = replay_research_program(body, rebound, SearchBudget(100000), feature_mapping=binding)
    assert result['training_loss'] < 1e-12


def test_temporal_revision_keeps_relation_symbols_and_dormant_columns():
    before = plan(dormant=True)
    saved = before['selection_steps'][0]
    fit = research_program_fit_contract(before)
    terms = [fit['fit_terms'][0], Program('window_sum', lag=3, children=(fit['fit_terms'][1],)),
             fit['fit_terms'][2]]
    after = payload(terms, [0., 2., 1., 0.])
    proposal = propose_transfer_partitions(before, after)
    body, = [item['research_program'] for item in proposal['candidates']
              if item['research_program']['transfer_partition']['kind'] == 'Pi3']
    rows, history = [], []
    target = research_program_fit_contract(after)['program']
    for row in observations(radius=.4):
        current = replace(row, history=tuple(history))
        rows.append(replace(current, target=target.predict(current)))
        history.append(row.features)
    result = replay_research_program(body, rows, SearchBudget(100000))
    assert result['training_loss'] < 1e-12
    view = transfer_partition_view(result['research_program'])
    assert view['status'] == 'ready'
    assert [item['symbol'] for item in view['groups']] == ['r']
    assert result['research_program']['selection_steps'][0]['parameters'] == saved['parameters']
    assert len(research_program_fit_contract(result['research_program'])['fit_terms']) == 3
