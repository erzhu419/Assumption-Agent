"""Old grammar and whole-context fits; controlled examples are not discovery."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.revision_relations import propose_revision
from test_revision_relations import component, edit_candidate, constant, feature, binary, revision


FITTING = {'operations': ['refit', 'representation', 'relation', 'composition', 'robustification', 'idealization'],
           'risk_quantile': .5, 'max_history': 2, 'max_structures': 6}


def examples(parent, offset=7.):
    return [{'source_ref': {'file': 'paid_context_fixture.json', 'json_path': [index]},
        'observation': {'features': {'u': value},
                        'target': parent.predict(Observation({'u': value}, 0.)) + offset,
                        'history': []}, 'tolerance': 1e-8}
        for index, value in enumerate((-2., -1., 1., 2.))]


def test_missing_component_is_fitted_on_whole_after_while_the_parent_stays_fixed():
    candidate = edit_candidate()
    parent = binary('mul', constant(5.), feature())
    declaration = component('parent', parent)
    checks = examples(parent)
    original = deepcopy((candidate, declaration, checks))
    result = propose_revision(candidate, declaration, [], allocation={'compute': 5000},
                              evidence_checks=checks, fitting=FITTING)
    assert result['proposals']
    passing = [item for item in result['proposals'] if item['fitting']['training_loss'] < 1e-12]
    assert passing
    program = Program.from_dict(passing[0]['program'])
    assert program.children[0] == parent
    assert program.predict(Observation({'u': 3.}, 0.)) == 22. or abs(
        program.predict(Observation({'u': 3.}, 0.)) - 22.) < 1e-8
    assert passing[0]['fitting']['scope'] == 'whole_after_paid_fit_not_confirmation'
    assert not passing[0]['independent_confirmation'] and result['actual_compute_cost'] <= 5000
    assert (candidate, declaration, checks) == original


def test_world_response_is_not_used_as_the_internal_offset_label():
    candidate = edit_candidate()
    parent = binary('mul', constant(5.), feature())
    fitted = propose_revision(candidate, component('parent', parent), [], allocation={'compute': 5000},
        evidence_checks=examples(parent, offset=11.), fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1})
    best = min(fitted['proposals'], key=lambda item: item['fitting']['training_loss'])
    program = Program.from_dict(best['program'])
    assert abs(program.children[1].value - 11.) < 1e-8
    assert program.children[0] == parent
    assert best['fitting']['fitted_rows'] == 4


def test_fitting_budget_exhaustion_keeps_the_actual_partial_cost_and_recovery_state():
    parent = binary('mul', constant(5.), feature())
    result = propose_revision(edit_candidate(), component('parent', parent), [],
        allocation={'compute': 55}, evidence_checks=examples(parent), fitting=FITTING)
    assert result['status'] == 'UNKNOWN'
    assert 0 < result['actual_compute_cost'] <= 55
    assert not result['support_updated'] and not result['independent_confirmation']


def test_startup_program_is_synthesized_inside_the_learned_context_without_changing_mature_parent():
    condition = Program('available', lag=1)
    lag = Program('lag', name='u', lag=1)
    sources = [revision(str(index), Program('if', children=(condition, before, startup)),
        before_program=before.to_dict(), counterexamples=[{
            'source_ref': {'file': 'paid_source_fixture.json', 'json_path': [index]},
            'observation': {'features': {'u': 2.}, 'target': 2., 'history': []}}])
        for index, (before, startup) in enumerate([(lag, feature()),
            (binary('mul', constant(2.), lag), Program('abs', children=(feature(),)))])]
    candidate = edit_candidate(sources)
    parent = Program('lag', name='u', lag=2)
    checks = []
    for index, value in enumerate((-2., -1., 1., 2.)):
        for depth in (0, 1, 2):
            history = [{'u': 10. + step} for step in range(depth)]
            checks.append({'source_ref': {'file': 'paid_startup_fixture.json', 'json_path': [len(checks)]},
                'observation': {'features': {'u': value}, 'history': history,
                    'target': 3. * value + 7. if depth < 2 else 10.}, 'tolerance': 1e-8})
    result = propose_revision(candidate, component('parent', parent), [],
        allocation={'compute': 12000}, evidence_checks=checks, fitting=FITTING)
    matching = [Program.from_dict(item['program']) for item in result['proposals']
                if item['fitting']['training_loss'] < 1e-12]
    assert matching
    program = matching[0]
    assert program.children[0].lag == 2 and program.children[1] == parent
    assert abs(program.predict(Observation({'u': 3.}, 0.)) - 16.) < 1e-8
    assert program.predict(Observation({'u': 3.}, 0., ({'u': 10.}, {'u': 11.}))) == 10.


def test_repeated_open_hole_keeps_one_parameter_binding_during_context_fitting():
    sources = []
    for index, old in enumerate((Program('abs', children=(feature(),)), binary('sub', feature(), constant(3.)))):
        extra = constant(index + 1.)
        after = binary('add', old, binary('add', extra, extra))
        sources.append(revision(str(index), after, before_program=old.to_dict(), counterexamples=[{
            'source_ref': {'file': 'paid_shared_hole_fixture.json', 'json_path': [index]},
            'observation': {'features': {'u': 2.}, 'history': [],
                'target': after.predict(Observation({'u': 2.}, 0.))}}]))
    parent = binary('mul', constant(5.), feature())
    result = propose_revision(edit_candidate(sources), component('parent', parent), [],
        allocation={'compute': 5000}, evidence_checks=examples(parent, offset=14.),
        fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1})
    best = min(result['proposals'], key=lambda item: item['fitting']['training_loss'])
    program = Program.from_dict(best['program'])
    assert program.children[1].children[0] == program.children[1].children[1]
    assert len(best['fitting']['parameters']) == 1
    assert abs(program.children[1].children[0].value - 7.) < 1e-8


def test_generated_affine_structure_cannot_assign_response_units_to_an_incompatible_feature():
    parent = constant(5.)
    declaration = component('parent', parent)
    declaration['role_signature']['u']['unit'] = 'A'
    result = propose_revision(edit_candidate(), declaration, [], allocation={'compute': 5000},
        evidence_checks=examples(parent), fitting={**FITTING, 'operations': ['refit'], 'max_structures': 2})
    assert result['proposals']
    assert any(item['reason'] == 'incompatible_generated_unit'
               for item in result['cursor']['fitting']['excluded_structures'])
    assert all(not Program.from_dict(item['program']).observed_variables() for item in result['proposals'])


def test_real_transfer_and_pr2_route_consume_context_fits_without_adopting_the_parent(tmp_path):
    from test_theory_consolidation import agent_scene, assert_no_new_science
    from test_question_routing import CONTEXT, future_question, probe, route, theory
    from hegel_machine.cognition.theory_consolidation import _save
    from hegel_machine.cognition.question_routing import register_theories
    agent, lab = agent_scene(tmp_path)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', parent)])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)]
    arguments = {'operator': 'theory_transfer', 'candidate_id': candidate['id'], 'theory_id': 'fitted_revision',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 5000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 2}}
    result = agent._execute('run', deepcopy(arguments), lab)
    assert result['generated_theories']
    assert any(item['status'] == 'refuted' for item in result['generated_theories'])
    passing = [item for item in result['generated_theories'] if item['status'] == 'candidate_registered']
    assert passing
    chosen = passing[0]
    assert chosen['binding']['fitting']['scope'] == 'whole_after_paid_fit_not_confirmation'
    assert agent.theory_index['parent']['program'] == parent.to_dict()
    question = future_question(agent)
    question['candidate_theory_ids'] = ['parent', chosen['theory_id']]
    routed = route(agent, lab, question=question, experiments=[probe(agent)])
    assert routed['selection']['selected_experiment']
    assert routed['parent_use']['status'] == 'parent_query_prepared'
    assert agent.model is None and not agent.confirmations
    used = deepcopy(agent.costs)
    repeated = agent._execute('run', arguments, lab)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0 and agent.costs == used
    assert_no_new_science(agent, lab)


def test_local_nonlinear_fit_stays_pending_with_global_budget_left_and_resumes_to_completion(monkeypatch):
    from hegel_machine.cognition.abstractions import _constant_paths
    from hegel_machine.cognition import revision_relations
    from hegel_machine.cognition.models import SearchBudget

    # Four product terms have eight declared free coefficients. Their zero
    # start is flat for one-coordinate moves, so the finite nonlinear pass must
    # assess both signs of every coefficient. The first binding's local share
    # can run out while the ordinary constant binding still fits and returns.
    terms = [binary('mul', constant(0.), constant(0.)) for _ in range(4)]
    nonlinear = terms[0]
    for term in terms[1:]:
        nonlinear = binary('add', nonlinear, term)
    components = [component('unfitted_nonlinear', nonlinear,
        parameter_paths=[list(path) for path in _constant_paths(nonlinear)])]
    parent = binary('mul', constant(5.), feature())
    candidate, declaration = edit_candidate(), component('parent', parent)
    checks = examples(parent)
    fitting = {**FITTING, 'operations': ['refit'], 'max_structures': 1}
    original = deepcopy((candidate, declaration, components, checks))
    # Explicitly bound the first *local* pass; added typed-occurrence accounting
    # should not turn this regression into a narrow global-budget coincidence.
    # The original fitter still executes, and every unit it spends is charged.
    fit = revision_relations._fit_revision_binding
    local_passes = []

    def first_local_share(candidate, prepared, fixed, rows, quantile, budget, **kwargs):
        if local_passes:
            return fit(candidate, prepared, fixed, rows, quantile, budget, **kwargs)
        assert any(item['id'] == 'unfitted_nonlinear' for name, item in prepared.items() if name not in fixed)
        local = SearchBudget(min(52, budget.remaining))
        output = fit(candidate, prepared, fixed, rows, quantile, local, **kwargs)
        assert budget.spend(local.spent)
        local_passes.append({'spent': local.spent, 'limit': local.limit})
        return output

    monkeypatch.setattr(revision_relations, '_fit_revision_binding', first_local_share)
    first = propose_revision(candidate, declaration, components, allocation={'compute': 500},
                             evidence_checks=checks, fitting=fitting)
    assert first['status'] == 'UNKNOWN' and first['reason'] == 'bounded_parameter_fitting_incomplete'
    assert not first['search_completed'] and not first['budget_exhausted']
    assert 0 < first['actual_compute_cost'] < 500 and first['attempted_bindings'] == 2
    assert 0 < local_passes[0]['spent'] <= local_passes[0]['limit'] == 52
    pending, = first['cursor']['pending_fits']
    assert first['incomplete_fit_positions'] == [pending['position']]
    assert any(item['fitting']['status'] == 'UNKNOWN' for item in first['proposals'])
    assert any(item['fitting']['status'] == 'completed' and item['fitting']['training_loss'] < 1e-12
               for item in first['proposals'])
    retained_cursor = deepcopy(first['cursor'])
    resumed = propose_revision(candidate, declaration, components, allocation={'compute': 400},
        evidence_checks=checks, fitting=fitting, cursor=first['cursor'])
    assert first['cursor'] == retained_cursor
    assert resumed['status'] == 'completed' and resumed['search_completed']
    assert not resumed['budget_exhausted'] and 0 < resumed['actual_compute_cost'] <= 400
    assert resumed['attempted_bindings'] == 1 and resumed['cursor']['pending_fits'] == []
    assert resumed['incomplete_fit_positions'] == []
    assert all(item['search_position']['layer'] == pending['position']['layer']
               and item['search_position']['offset'] == pending['position']['offset']
               and item['fitting']['search_completed']
               and item['fitting']['status'] == 'UNKNOWN'
               and item['fitting']['parameter_identifiability'] == 'UNKNOWN'
               for item in resumed['proposals'])
    # Completing the finite trials is not a successful nonlinear fit: this
    # binding's retained training error remains visible, without new support.
    assert resumed['proposals'] and all(item['fitting']['training_loss'] == 49.
                                        for item in resumed['proposals'])
    for result in (first, resumed):
        assert not result['support_updated'] and not result['independent_confirmation']
        assert all(Program.from_dict(item['program']).children[0] == parent
                   for item in result['proposals'])
    assert (candidate, declaration, components, checks) == original


def test_pending_fit_resumes_before_the_unfinished_new_binding_frontier():
    from hegel_machine.cognition.abstractions import _constant_paths

    terms = [binary('mul', constant(0.), constant(0.)) for _ in range(4)]
    nonlinear = terms[0]
    for term in terms[1:]:
        nonlinear = binary('add', nonlinear, term)
    components = [component('unfitted_nonlinear', nonlinear,
        parameter_paths=[list(path) for path in _constant_paths(nonlinear)])]
    # These are genuinely distinct available bindings, not unfinished fits.
    # Their remaining frontier must not postpone the earlier nonlinear work.
    components.extend(component(f'constant_{index}', constant(index)) for index in range(20))
    parent = binary('mul', constant(5.), feature())
    candidate, declaration = edit_candidate(), component('parent', parent)
    checks = examples(parent)
    fitting = {**FITTING, 'operations': ['refit'], 'max_structures': 1}
    first = propose_revision(candidate, declaration, components, allocation={'compute': 200},
        evidence_checks=checks, fitting=fitting)
    assert first['status'] == 'UNKNOWN' and first['budget_exhausted']
    pending = deepcopy(first['cursor']['pending_fits'][0])
    assert first['cursor']['layer'] < len(components) - 1
    assert any(item['search_position']['layer'] == pending['position']['layer']
               and item['fitting']['status'] == 'UNKNOWN' for item in first['proposals'])
    retained_cursor = deepcopy(first['cursor'])

    resumed = propose_revision(candidate, declaration, components, allocation={'compute': 400},
        evidence_checks=checks, fitting=fitting, cursor=first['cursor'])
    assert first['cursor'] == retained_cursor
    assert resumed['proposals']
    retried = resumed['proposals'][0]
    assert {key: retried['search_position'][key] for key in ('layer', 'offset')} == pending['position']
    assert retried['fitting']['search_completed']
    assert retried['fitting']['status'] == 'UNKNOWN'
    assert retried['fitting']['parameter_identifiability'] == 'UNKNOWN'
    assert pending not in resumed['cursor']['pending_fits']
    assert 0 < resumed['actual_compute_cost'] <= 400
    assert Program.from_dict(retried['program']).children[0] == parent
    assert retried['fitting']['training_loss'] == 49.
    assert not resumed['support_updated'] and not resumed['independent_confirmation']
