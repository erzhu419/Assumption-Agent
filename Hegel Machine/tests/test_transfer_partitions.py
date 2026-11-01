"""One bound relation is reidentified without copying the after answer."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_uncertainty import _bind
from hegel_machine.cognition.research_programs import _entry
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import (
    continue_transfer_partition, propose_transfer_partitions, quote_transfer_partition, replay_transfer_partition,
    transfer_partition_view,
)


def second_stage_fixture():
    payload = plan(.75, weight=3., dormant=True)
    payload['selection_steps'][0]['parameters'][0].update(
        radius_intervals=[[.6, 1.0]], prior_radius_intervals=[[0., 1.2]])
    payload['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'max_attempts': 48, 'max_candidates': 12, 'second_stage': {
            'operation': 'resolve_relation_intervals',
            'trigger': 'fitted_training_with_unexplained_return', 'compute_limit': 15000}}
    target = research_program_fit_contract(plan(.85, weight=3., dormant=True))['program']
    def measured(distances):
        result = []
        for i, distance in enumerate(distances):
            fields = {'drive': (i % 5) * .3, 'zero': 0., 'f0': 1., 'f1': 0.,
                'x0': 0., 'x1': distance, 'q0': .2 * (i % 7), 'q1': 1. + .13 * (i % 4)}
            row = Observation(fields, 0.)
            result.append(replace(row, target=target.predict(row)))
        return result
    training = measured([.2, .4, .6, 1.1] * 6)
    feedback = measured([.8, .9] * 4)
    fit = research_program_fit_contract(payload)
    entry = _entry(fit, [{**payload['selection_steps'][0], 'program': fit['program']}], len(training), 1)
    entry['training_loss'] = 0.
    entry['research_program']['transfer_partition'] = deepcopy(payload['transfer_partition'])
    loss = sum((entry['candidate'].program.predict(row) - row.target) ** 2 for row in feedback) / len(feedback)
    return entry, training, feedback, loss


def test_paid_return_drives_interval_second_stage_without_refitting(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    entry, training, feedback, measured = second_stage_fixture()
    before = deepcopy(entry)
    monkeypatch.setattr(module._Fitter, 'fit', lambda *args: pytest.fail('completed fit repeated'))
    budget = SearchBudget(15000)
    result = continue_transfer_partition(entry, training, feedback, budget,
        tolerance=1e-7, feedback_loss=measured)
    assert result['status'] == 'paid_interval_supported'
    assert result['execution_complete'] and not result['budget_incomplete']
    assert result['first_training_loss'] < 1e-12 and result['feedback_loss'] < 1e-12
    assert result['compute_cost'] == budget.spent and result['new_fits'] == result['new_observations'] == 0
    selected = result['selected']
    fit = research_program_fit_contract(selected['research_program'])
    assert len(fit['fit_terms']) == 3 and fit['coefficients'][-1] == 0.
    assert fit['coefficients'] == research_program_fit_contract(entry['research_program'])['coefficients']
    assert selected['research_program']['transfer_partition']['released_groups'] == ['relation_parameter_0']
    assert transfer_partition_view(selected['research_program'])['status'] == 'ready'
    assert entry == before


def test_ordinary_replay_executes_native_paid_interval_followup(monkeypatch):
    from hegel_machine.cognition.research_programs import replay_research_program
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch

    entry, training, feedback, _ = second_stage_fixture()
    body = deepcopy(entry['research_program'])
    calls = []
    native = RelationIntervalResearch.alternatives

    def record(self, paid_rows, budget, tolerance=1e-7, **kwargs):
        calls.append((list(paid_rows), kwargs))
        return native(self, paid_rows, budget, tolerance, **kwargs)

    monkeypatch.setattr(RelationIntervalResearch, 'alternatives', record)
    quoted = quote_transfer_partition(body, training, feedback_rows=feedback)
    budget = SearchBudget(quoted['required_compute'])
    result = replay_research_program(body, training, budget,
        feedback_rows=feedback, tolerance=1e-7)
    followup = result['transfer_partition_followup']
    assert followup['status'] == 'paid_interval_supported'
    assert calls and calls[0][0] == [*training, *feedback]
    assert calls[0][1]['experiments'] == [feedback]
    assert calls[0][1]['parameter_index'] == 0
    assert followup['execution_complete'] and followup['selected_new']
    assert followup['first_training_loss'] < 1e-12 and followup['feedback_loss'] < 1e-12
    assert followup['data_role'] == 'exposed_adaptive_development'
    assert not followup['independent_support'] and followup['new_fits'] == 0
    assert budget.spent <= quoted['required_compute']
    assert result['training_loss'] == followup['first_training_loss']
    assert result['validation_loss'] == followup['feedback_loss']


def test_followup_respects_actual_return_and_first_stage_status():
    entry, training, feedback, measured = second_stage_fixture()
    budget = SearchBudget(15000)
    absent = continue_transfer_partition(entry, training, [], budget, tolerance=1e-7)
    assert absent['status'] == 'paid_return_required' and budget.spent == 0
    satisfied = continue_transfer_partition(entry, training, feedback, budget,
        tolerance=1e-7, feedback_loss=0.)
    assert satisfied['status'] == 'paid_return_already_supported' and budget.spent == 0
    bad_fit = deepcopy(entry)
    bad_fit['training_loss'] = 1.
    result = continue_transfer_partition(bad_fit, training, feedback, budget,
        tolerance=1e-7, feedback_loss=measured)
    assert result['status'] == 'first_stage_training_unexplained' and budget.spent == 0
    actual = continue_transfer_partition(entry, training, feedback, budget, tolerance=1e-7)
    assert actual['status'] == 'paid_interval_supported' and actual['prior_feedback_loss'] == measured


def test_interval_budget_limit_is_unknown_not_a_scientific_negative():
    entry, training, feedback, measured = second_stage_fixture()
    entry['research_program']['transfer_partition']['second_stage']['compute_limit'] = 1
    result = continue_transfer_partition(entry, training, feedback, SearchBudget(100000),
        tolerance=1e-7, feedback_loss=measured)
    assert result['status'] == 'compute_unresolved' and result['budget_incomplete']
    assert not result['execution_complete'] and result['selected'] is entry
    assert result['compute_cost'] <= 1


def test_completed_fixed_readout_failure_retains_comparable_scores():
    entry, training, feedback, _ = second_stage_fixture()
    changed = [replace(row, target=row.target + 7.) for row in feedback]
    measured = sum((entry['candidate'].program.predict(row) - row.target) ** 2 for row in changed) / len(changed)
    result = continue_transfer_partition(entry, training, changed, SearchBudget(15000),
        tolerance=1e-7, feedback_loss=measured)
    assert result['status'] == 'fixed_readout_interval_unexplained'
    assert result['execution_complete'] and not result['selected_new']
    assert result['selected'] is entry and result['first_training_loss'] == 0.
    assert result['feedback_loss'] == measured
    assert result['all_paid_loss'] == pytest.approx(len(changed) * measured / (len(training) + len(changed)))
    assert result['fixed_readout_only'] and not result['independent_support']


def test_target_residual_contract_does_not_inherit_source_precision_stop():
    entry, training, feedback, _ = second_stage_fixture()
    # Supported public geometry can require distinctions below the old fixed
    # 1e-6 source precision. There is still an observed returned counterexample.
    payload = entry['research_program']
    original = research_program_fit_contract(payload)
    saved = payload['selection_steps'][0]
    saved['parameters'][0].update(radius=.8500001,
        radius_intervals=[[.85, .8500009]], prior_radius_intervals=[[.85, .8500009]])
    templates = [Program.from_dict(value) for value in saved['parameter_terms']]
    terms = [_bind(term, {'r': Program('const', value=.8500001)}) for term in templates]
    program = _sum_terms(terms, original['coefficients'][1:], original['coefficients'][0])
    updated = _entry({**original, 'program': program, 'fit_terms': terms},
        [{**saved, 'program': program}], len(training), 1)
    updated['research_program']['transfer_partition'] = deepcopy(payload['transfer_partition'])
    updated['training_loss'] = 0.
    truth_terms = [_bind(term, {'r': Program('const', value=.8500005)}) for term in templates]
    truth = _sum_terms(truth_terms, original['coefficients'][1:], original['coefficients'][0])
    returned = []
    for i, row in enumerate(feedback):
        changed = Observation({**row.features, 'x1': .8500003 if i % 2 else .8500007}, 0.)
        returned.append(replace(changed, target=truth.predict(changed)))
    result = continue_transfer_partition(updated, training, returned, SearchBudget(15000), tolerance=1e-7)
    assert result['status'] == 'paid_interval_supported'
    assert result['feedback_loss'] < 1e-12


def relation(symbol='r', prefix=''):
    return Program('neighbor_sum', children=(Program('arg', name=symbol),
        *(Program('feature', name=prefix + name) for name in ('f0', 'x0', 'q0', 'f1', 'x1', 'q1'))))


def plan(radius=.4, *, weight=1., other=False, dormant=False):
    templates = [Program('feature', name='drive'), relation()]
    parameters = [{'symbol': 'r', 'radius': radius, 'radius_intervals': [[radius - .01, radius + .01]],
                   'source_concept_id': None, 'relation': 'symmetric'}]
    if other:
        templates.append(relation('s', 'b'))
        parameters.append({'symbol': 's', 'radius': radius, 'radius_intervals': [[radius - .01, radius + .01]],
                           'source_concept_id': None, 'relation': 'symmetric'})
    if dormant:
        templates.append(Program('feature', name='zero'))
    coefficients = [0., 2., weight, *([.7] if other else []), *([0.] if dormant else [])]
    bindings = {item['symbol']: Program('const', value=item['radius']) for item in parameters}
    terms = [_bind(term, bindings) for term in templates]
    program = _sum_terms(terms, coefficients[1:], coefficients[0])
    fit = {'program': program, 'fit_terms': terms, 'terms': terms, 'coefficients': coefficients, 'loss': 0.}
    steps = [{'operation': 'resolve_relation_intervals', 'question': 'Resolve actual parameter binding',
        'program': program, 'loss': 0., 'parameters': parameters,
        'parameter_template': _sum_terms(templates, coefficients[1:], coefficients[0]).to_dict(),
        'parameter_terms': [term.to_dict() for term in templates]}]
    return _entry(fit, steps, 20, 100)['research_program']


def observations(radius=.85, weight=3., count=36, *, other=False):
    target = research_program_fit_contract(plan(radius, weight=weight, other=other, dormant=True))['program']
    rows = []
    for index in range(count):
        fields = {'drive': (index % 5) * .3, 'zero': 0., 'f0': 1., 'f1': 0.,
                  'x0': 0., 'x1': .1 + (index % 12) * .1,
                  'q0': .2 * (index % 7), 'q1': 1. + .13 * (index % 4)}
        if other:
            fields.update({('b' + key): value for key, value in list(fields.items()) if key in ('f0', 'f1', 'x0', 'x1', 'q0', 'q1')})
        row = Observation(fields, 0.)
        rows.append(replace(row, target=target.predict(row)))
    return rows


def proposal(*, other=False, dormant=True):
    return propose_transfer_partitions(plan(other=other, dormant=dormant),
        plan(.85, weight=3., other=other, dormant=dormant),
        source_ref={'file': 'paid/revision.json'}, before_ref='before', after_ref='after')


def test_symbols_keep_equal_numbers_distinct_and_retain_zero_columns():
    source = plan(other=True, dormant=True)
    view = transfer_partition_view(source)
    assert view['status'] == 'ready'
    assert [group['symbol'] for group in view['groups']] == ['r', 's']
    assert len(view['parameter_terms']) == 4
    assert view['groups'][0]['occurrences'] == [{'term_node': 'term:1', 'path': []}]
    proposed = proposal(other=True)
    assert len(proposed['candidates']) == 2
    assert [item['research_program']['transfer_partition']['released_groups'] for item in proposed['candidates']] == [['r'], ['s']]
    assert all(item['research_program']['transfer_partition']['source_revision']['isolated_group_effect'] is False
               for item in proposed['candidates'])


def test_no_change_or_missing_actual_binding_does_not_invent_a_revision():
    before = plan()
    assert not propose_transfer_partitions(before, deepcopy(before))['candidates']
    del before['selection_steps'][0]['parameter_terms']
    assert transfer_partition_view(before)['status'] == 'binding_unavailable'
    assert not propose_transfer_partitions(before, plan(.8))['candidates']


def test_after_values_are_not_search_initialization():
    first = proposal()['candidates'][0]['research_program']
    other = propose_transfer_partitions(plan(dormant=True), plan(9., weight=8., dormant=True),
        source_ref={'file': 'paid/revision.json'}, before_ref='before', after_ref='after')['candidates'][0]['research_program']
    assert first == other
    assert first['selection_steps'][0]['parameters'][0]['radius'] == .4


def test_joint_readout_fits_escape_source_interval_and_preserve_declared_zero_column():
    proposed = proposal()
    rows = observations()
    current = proposed['candidates'][0]['research_program']
    quote = quote_transfer_partition(current, rows)
    budget = SearchBudget(quote['required_compute'])
    selected = replay_transfer_partition(current, rows, budget)
    assert selected is not None
    assert selected['training_loss'] < 1e-12
    assert budget.spent <= quote['required_compute']
    fit = research_program_fit_contract(selected['research_program'])
    assert len(fit['fit_terms']) == 3 and fit['coefficients'][-1] == 0.
    binding = selected['research_program']['selection_steps'][0]
    assert .4 not in [item['radius'] for item in binding['parameters']]
    assert binding['parameters'][0]['radius'] > .41
    assert fit['coefficients'][2] == pytest.approx(3., abs=1e-7)
    summary = selected['transfer_partition_search']
    assert summary['completed'] and summary['compute_cost'] == budget.spent
    assert summary['attempts'] <= 48 and summary['independent_support'] is False
    baseline = replay_transfer_partition(proposed['baseline_research_program'], rows, SearchBudget(10000))
    assert baseline['training_loss'] > .01
    assert transfer_partition_view(selected['research_program'])['status'] == 'ready'


def test_fixed_legal_search_contract_is_not_silently_enlarged_and_partial_is_unresolved():
    current = proposal()['candidates'][0]['research_program']
    current['transfer_partition']['max_attempts'] = 3
    result = replay_transfer_partition(current, observations(), SearchBudget(10000), max_attempts=1000)
    assert result['transfer_partition_search']['attempts'] <= 3
    assert result['transfer_partition_search']['remaining'] > 0
    quote = quote_transfer_partition(current, observations())
    one_fit_budget = quote['initialization_compute'] + quote['geometry_compute'] + 5 * len(observations())
    budget = SearchBudget(one_fit_budget)
    partial = replay_transfer_partition(current, observations(), budget)
    assert partial is not None
    assert partial['transfer_partition_search']['status'] == 'compute_unresolved'
    assert not partial['transfer_partition_search']['completed']
    assert budget.denied_requests > 0


def test_different_outer_composition_is_a_joint_source_not_pure_group_evidence():
    before = plan()
    after = plan(.85)
    fit = research_program_fit_contract(after)
    changed = [Program('window_sum', lag=2, children=(term,)) for term in fit['fit_terms']]
    program = _sum_terms(changed, fit['coefficients'][1:], fit['coefficients'][0])
    after = _entry({**fit, 'program': program, 'fit_terms': changed}, [], 20, 1)['research_program']
    after['structural_replay'] = {'radius': .85}
    result = propose_transfer_partitions(before, after)
    assert result['candidates'][0]['edit']['source_revision_kind'] == 'joint_variant'
    proposal_fit = research_program_fit_contract(result['candidates'][0]['research_program'])
    assert proposal_fit['fit_terms'] == research_program_fit_contract(before)['fit_terms']


def test_unordered_object_roles_match_without_reordering_execution():
    before = plan()
    after = plan(.85)
    fit = research_program_fit_contract(after)
    term = fit['fit_terms'][1]
    changed = replace(term, children=(term.children[0], *term.children[4:], *term.children[1:4]))
    terms = [fit['fit_terms'][0], changed]
    program = _sum_terms(terms, fit['coefficients'][1:], fit['coefficients'][0])
    after = _entry({**fit, 'program': program, 'fit_terms': terms}, [], 20, 1)['research_program']
    result = propose_transfer_partitions(before, after)
    assert result['candidates'][0]['edit']['source_revision_kind'] == 'relation_and_readout_revision'
    assert research_program_fit_contract(result['candidates'][0]['research_program'])['fit_terms'] == research_program_fit_contract(before)['fit_terms']


def test_pi0_keeps_binding_for_later_method_revision_and_pi1_has_current_interval():
    proposed = proposal()
    rows = observations()
    baseline = replay_transfer_partition(proposed['baseline_research_program'], rows, SearchBudget(10000))
    assert transfer_partition_view(baseline['research_program'])['status'] == 'ready'
    result = replay_transfer_partition(proposed['candidates'][0]['research_program'], rows, SearchBudget(10000))
    saved = result['research_program']['selection_steps'][0]['parameters'][0]
    assert saved['source_radius_intervals'] == [[.39, .41000000000000003]]
    assert saved['radius_intervals'][0][0] <= saved['radius'] <= saved['radius_intervals'][0][1]
    assert saved['prior_radius_intervals'][0][0] == 0.
    assert saved['interval_meaning'] == 'current_geometry_cell_not_confidence_interval'


def test_feature_rebinding_is_used_by_parameter_fit():
    current = proposal()['candidates'][0]['research_program']
    rows = observations()
    mapping = {name: 'new_' + name for name in rows[0].features}
    rebound = [replace(row, features={mapping[key]: value for key, value in row.features.items()}) for row in rows]
    result = replay_transfer_partition(current, rebound, SearchBudget(10000), feature_mapping=mapping)
    assert result['training_loss'] < 1e-12
    assert result['candidate'].program.required_features() <= set(mapping.values())


def test_two_equal_bindings_of_same_kernel_are_not_merged_by_source_identity():
    source = plan(other=True)
    # Identical present coordinates do not imply the two declared parameters
    # must remain shared on a later task.
    from hegel_machine.cognition.research_drafts import _rebind_step
    source = _rebind_step(source, {'b' + name: name for name in ('f0', 'f1', 'x0', 'x1', 'q0', 'q1')})
    for item in source['selection_steps'][0]['parameters']:
        item['source_concept_id'] = 'same_old_source'
    source['transfer_partition'] = {'kind': 'Pi1', 'released_groups': ['r'],
        'max_attempts': 6, 'max_candidates': 2}
    result = replay_transfer_partition(source, observations(), SearchBudget(10000))
    assert result is not None
    parameters = result['research_program']['selection_steps'][0]['parameters']
    assert [item['symbol'] for item in parameters] == ['r', 's']
    assert parameters[1]['radius'] == .4
    assert all(item['source_concept_id'] == 'same_old_source' for item in parameters)
