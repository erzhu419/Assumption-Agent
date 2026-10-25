"""Exact retained fits are computation reuse, not new empirical support."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, component, constant, edit_candidate, feature


def _retained(template, rows, *, scope=None):
    report = {}
    fitted = fit_parameter_slots(template, {}, rows, SearchBudget(1000), .5, report=report)
    assert report['search_completed']
    return {'template': template, 'rows': tuple(rows), 'risk_quantile': .5,
        **(scope or {}), 'source_ref': {'file': 'completed.json', 'json_path': ['fit']},
        'report': report, 'outputs': [
            {'parameters': [revisions._at(program, path).value for path in template.parameter_paths],
             'method': method, 'training_loss': report['losses'][program],
             'source_ref': {'file': 'completed.json', 'json_path': ['outputs', index]}}
            for index, (method, program) in enumerate(zip(report['candidate_methods'], fitted))]}


def _linear_template():
    definition = Definition('edit', ('old', 'gain', 'offset'),
        binary('add', Program('arg', name='old'),
            binary('add', binary('mul', Program('arg', name='gain'), feature()),
                Program('arg', name='offset'))))
    call = Program('call', name='edit', definition=definition,
        children=(binary('mul', constant(5.), feature()), constant(0.), constant(0.)))
    return AbstractTemplate('fit', 'edit', call, (), (), ((1,), (2,)))


def _scope():
    declaration = component('original_parent', constant(5.))
    return {key: declaration[key] for key in ('role_signature', 'output_unit', 'context_ref')}


def test_proposer_reuses_complete_methods_across_parent_ids_without_fitting_or_prediction(monkeypatch):
    parent = binary('mul', constant(5.), feature())
    declaration = component('original_parent', parent)
    checks, groups = examples(parent), []
    fitting = {**FITTING, 'operations': ['refit'], 'max_structures': 1}
    original_fit = revisions._fit_revision_binding

    def collect(candidate, prepared, fixed, rows, quantile, budget, **kwargs):
        template = kwargs.get('template') or revisions._revision_fitting_template(candidate, prepared, fixed)
        result = original_fit(candidate, prepared, fixed, rows, quantile, budget, template=template)
        programs, _, report = result
        assert report['search_completed']
        groups.append({'template': template, 'rows': tuple(rows), 'risk_quantile': quantile,
            **{key: declaration[key] for key in ('role_signature', 'output_unit', 'context_ref')},
            'source_ref': {'file': 'retained.json', 'json_path': [len(groups)]},
            'report': deepcopy(report), 'outputs': [
                {'parameters': [revisions._at(program, path).value for path in template.parameter_paths],
                 'method': method, 'training_loss': report['losses'][program],
                 'source_ref': {'file': 'retained.json', 'json_path': [len(groups), index]}}
                for index, (program, method) in enumerate(zip(programs, report['candidate_methods']))]})
        return result

    monkeypatch.setattr(revisions, '_fit_revision_binding', collect)
    original = revisions.propose_revision(edit_candidate(), declaration, [], {'compute': 4000},
        evidence_checks=checks, fitting=fitting)
    assert groups and original['proposals']
    inputs = deepcopy(groups)

    def forbidden(*args, **kwargs):
        raise AssertionError('the retained complete fit must not execute again')

    monkeypatch.setattr(revisions, '_fit_revision_binding', forbidden)
    # Source-trigger checking still predicts the actual parent. The numerical
    # reuse operation itself is tested separately below with predict forbidden.
    reused = revisions.propose_revision(edit_candidate(), {**declaration, 'id': 'other_parent_identity'}, [],
        {'compute': 4000}, evidence_checks=checks, fitting=fitting, retained_fits=groups)
    assert [item['program'] for item in reused['proposals']] == [item['program'] for item in original['proposals']]
    assert [item['fitting']['training_loss'] for item in reused['proposals']] == [
        item['fitting']['training_loss'] for item in original['proposals']]
    assert all(item['fitting']['fit_reuse_ref'] and item['fitting']['search_completed']
               for item in reused['proposals'])
    assert 0 < reused['actual_compute_cost'] < original['actual_compute_cost'] + 300
    assert reused['cursor']['pending_fits'] == []
    assert reused['retained_fit_reuse_count'] == len(groups)
    assert not reused['support_updated'] and not reused['independent_confirmation']
    assert groups == inputs


def test_matcher_keeps_rank_unknown_and_restores_exact_three_candidates_without_predict(monkeypatch):
    template = _linear_template()
    # Identical u leaves gain/intercept unidentifiable but the finite solve is complete.
    rows = [Observation({'u': 2.}, value) for value in (10., 12., 14.)]
    record = _retained(template, rows, scope=_scope())
    assert record['report']['status'] == 'UNKNOWN' and record['report']['search_completed']
    saved = deepcopy(record)
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('reuse must not predict'))
    budget = SearchBudget(3000)
    fitted, paths, report = revisions._reuse_revision_fit(template, rows, .5, _scope(), [record], budget)
    assert paths == template.parameter_paths
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert report['status'] == 'UNKNOWN' and report['parameter_identifiability'] == 'UNKNOWN'
    assert report['search_completed'] and report['actual_compute_cost'] == 0
    assert report['original_compute_cost'] == saved['report']['actual_compute_cost']
    assert [report['losses'][program] for program in fitted] == [item['training_loss'] for item in record['outputs']]
    assert 0 < budget.spent < 1000 and record == saved


@pytest.mark.parametrize('change', ['initial', 'fixed', 'paths', 'rows', 'history', 'quantile',
                                    'unit', 'role', 'context', 'incomplete', 'missing_method', 'sharing'])
def test_changed_fit_contract_misses_instead_of_reusing_a_final_program(change):
    template = _linear_template()
    rows = [Observation({'u': value}, value * 2. + 7.) for value in (-2., -1., 1., 2.)]
    record = _retained(template, rows, scope=_scope())
    current, actual_rows, quantile, scope = deepcopy(template), deepcopy(rows), .5, _scope()
    if change in {'initial', 'fixed'}:
        children = list(current.pattern.children)
        index = 1 if change == 'initial' else 0
        children[index] = constant(1.)
        current.pattern = Program('call', name=current.pattern.name, definition=current.pattern.definition,
                                  children=tuple(children))
    elif change == 'paths':
        current.parameter_paths = tuple(reversed(current.parameter_paths))
    elif change == 'rows':
        actual_rows.reverse()
    elif change == 'history':
        actual_rows[0] = Observation(rows[0].features, rows[0].target, ({'u': 9.},))
    elif change == 'quantile':
        quantile = .7
    elif change == 'unit':
        scope['output_unit'] = 'A'
    elif change == 'role':
        scope['role_signature']['u']['time'] = 'previous'
    elif change == 'context':
        scope['context_ref'] = {'scene': 'other'}
    elif change == 'incomplete':
        record['report']['search_completed'] = False
    elif change == 'missing_method':
        record['outputs'].pop()
    else:
        body = current.pattern.definition.body
        definition = Definition('edit', ('old', 'gain', 'offset'),
            binary('add', body, Program('arg', name='gain')))
        current.pattern = Program('call', name='edit', definition=definition, children=current.pattern.children)
    assert revisions._reuse_revision_fit(current, actual_rows, quantile, scope, [record], SearchBudget(3000)) is None


def test_reuse_metadata_work_exhausts_shared_budget_instead_of_running_free():
    template, rows = _linear_template(), [Observation({'u': 2.}, 12.)]
    record = _retained(template, rows, scope=_scope())
    budget = SearchBudget(1)
    with pytest.raises(revisions._Incomplete):
        revisions._reuse_revision_fit(template, rows, .5, _scope(), [record], budget)
    assert budget.spent == 1


def test_missed_or_incomplete_retained_fit_uses_the_original_fitter(monkeypatch):
    parent = binary('mul', constant(5.), feature())
    declaration, checks = component('parent', parent), examples(parent)
    fit, calls, captured = revisions._fit_revision_binding, [], []

    def capture(candidate, prepared, fixed, rows, quantile, budget, **kwargs):
        template = kwargs.get('template') or revisions._revision_fitting_template(candidate, prepared, fixed)
        result = fit(candidate, prepared, fixed, rows, quantile, budget, template=template)
        captured.append(_retained(template, rows, scope=_scope()))
        calls.append(quantile)
        return result

    monkeypatch.setattr(revisions, '_fit_revision_binding', capture)
    fitting = {**FITTING, 'operations': ['refit'], 'max_structures': 1}
    revisions.propose_revision(edit_candidate(), declaration, [], {'compute': 4000},
        evidence_checks=checks, fitting=fitting)
    original_calls = len(calls)
    assert original_calls and captured
    for record in captured:
        record['report']['search_completed'] = False
    result = revisions.propose_revision(edit_candidate(), declaration, [], {'compute': 4000},
        evidence_checks=checks, fitting=fitting, retained_fits=deepcopy(captured))
    assert len(calls) == 2 * original_calls
    assert result['proposals'] and result['retained_fit_reuse_count'] == 0
    assert all('fit_reuse_ref' not in item['fitting'] for item in result['proposals'])


def _template_with_unused_before_argument(*, free=False):
    original = _linear_template()
    definition = Definition('edit', ('before_only', *original.pattern.definition.arguments),
                            original.pattern.definition.body)
    call = Program('call', name='edit', definition=definition,
        children=(constant(0.), *original.pattern.children))
    return AbstractTemplate('fit', 'edit', call, (), (),
        (*(((0,),) if free else ()), (2,), (3,)))


def test_unused_fixed_before_subtree_is_not_a_new_fit_and_restoration_keeps_current_argument(monkeypatch):
    template = _template_with_unused_before_argument()
    rows = [Observation({'u': u}, 7. * u + 2.) for u in (-2., -1., 1., 2.)]
    record = _retained(template, rows, scope=_scope())
    # The actual retained parent can change these before-only subtrees while
    # the after Definition never reads them, including their apparent history.
    unused = Program('if', children=(Program('available', lag=4),
        binary('add', Program('lag', name='u', lag=4), constant(9.)), feature()))
    current = deepcopy(template)
    current.pattern = Program('call', name='edit', definition=template.pattern.definition,
                             children=(unused, *template.pattern.children[1:]))
    before = deepcopy((record, current))
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('reuse must not predict'))
    fitted, paths, report = revisions._reuse_revision_fit(
        current, rows, .5, _scope(), [record], SearchBudget(3000))
    assert paths == current.parameter_paths
    assert all(item.children[0] == unused for item in fitted)
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert report['fit_reuse_ref'] == record['source_ref']
    assert [report['losses'][item] for item in fitted] == [x['training_loss'] for x in record['outputs']]
    assert (record, current) == before


@pytest.mark.parametrize('change', ['initial', 'path'])
def test_unused_free_argument_and_its_declared_slot_are_still_exact_inputs(change):
    template = _template_with_unused_before_argument(free=True)
    rows = [Observation({'u': u}, 7. * u + 2.) for u in (-2., -1., 1., 2.)]
    record = _retained(template, rows, scope=_scope())
    current = deepcopy(template)
    if change == 'initial':
        current.pattern = Program('call', name='edit', definition=template.pattern.definition,
                                  children=(constant(1.), *template.pattern.children[1:]))
    else:
        current.parameter_paths = ((1, 0), (2,), (3,))
    assert revisions._reuse_revision_fit(current, rows, .5, _scope(), [record], SearchBudget(3000)) is None
