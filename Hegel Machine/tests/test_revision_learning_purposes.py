"""Learning purposes change actual proposal admission, not scientific gates."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.parameter_roles import parameter_family_correspondence
from hegel_machine.cognition.revision_relations import consolidate_revision_relations, propose_revision
from test_revision_context_fitting import FITTING
from test_revision_relations import binary, component, constant, feature, revision


def sources(*, structural):
    records = []
    for source, values in enumerate(((2., 3., 4.), (5., 6., 7.))):
        for value in values:
            before = binary('mul', constant(value), feature())
            after = binary('add', feature(), constant(value ** 2)) if structural else binary(
                'mul', constant(value ** 2), feature())
            records.append(revision(f'{source}:{value}', after, source_id=f'physical_{source}',
                before_program=before.to_dict(), counterexamples=[{
                    'source_ref': {'file': 'controlled_paid.json', 'json_path': [len(records)]},
                    'observation': {'features': {'u': -2.}, 'target': value ** 2 - 2.
                        if structural else -2. * value ** 2, 'history': []},
                    'tolerance': 1e-10}]))
    return records


def build(records):
    return consolidate_revision_relations(records, allocation={'compute': 25000,
        'max_groups': 1, 'parameter_relations': {'operations': ['refit'],
            'max_structures': 2, 'risk_quantile': .5, 'tolerance': 1e-10}},
        object_type='program_revision_relation')


def target_checks():
    return [{'source_ref': {'file': 'controlled_target.json', 'json_path': [index]},
        'observation': {'features': {'u': value}, 'target': value + 7., 'history': []},
        'tolerance': 1e-10} for index, value in enumerate((-2., -1., 1., 2.))]


def test_unshared_structural_revision_survives_failed_exact_map_and_enters_whole_fit():
    records = sources(structural=True)
    original = deepcopy(records)
    built = build(records)
    candidate, = built['candidates']
    assert candidate['purpose'] == 'structural_revision'
    assert not candidate.get('parameter_relations')
    assert built['parameter_relation_searches']
    assert all(not search['accepted'] for search in built['parameter_relation_searches'])
    assert all(item['before_exact'] and item['after_exact'] for item in candidate['source_reconstruction'])
    assert records == original

    parent = component('actual_parent', binary('mul', constant(5.), feature()))
    unchanged = deepcopy((candidate, parent))
    result = propose_revision(candidate, parent, [], allocation={'compute': 5000},
        evidence_checks=target_checks(),
        fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1})
    assert result['proposals']
    assert min(item['fitting']['training_loss'] for item in result['proposals']) < 1e-20
    assert all(item['parent_binding']['component_id'] == parent['id'] for item in result['proposals'])
    assert 0 < result['actual_compute_cost'] <= 5000
    assert not result['support_updated'] and not result['independent_confirmation']
    assert (candidate, parent) == unchanged


def test_constant_refit_is_a_parameter_proposal_and_requires_actual_whole_fit():
    built = build(sources(structural=False))
    candidate, = built['candidates']
    assert candidate['purpose'] == 'parameter_proposal'
    assert candidate['source_change_kind'] == 'parameter_refit'
    assert candidate['structural_change'] is False
    assert not candidate.get('parameter_relations')
    parent = component('actual_parent', binary('mul', constant(5.), feature()))
    pool = [component('existing_value', constant(7.), output_unit='dimensionless')]
    result = propose_revision(candidate, parent, pool, allocation={'compute': 5000},
                              evidence_checks=target_checks())
    assert not result['proposals']
    assert result['reason'] == 'whole_after_fitting_required_for_parameter_proposal'
    assert result['status'] == 'UNKNOWN'


def test_full_exact_parameter_maps_keep_their_direct_frozen_consumer():
    from test_revision_parameter_consumption import scaled

    records = sources(structural=False)
    for record, value in zip(records, (2., 3., 4., 5., 6., 7.)):
        record['after_program'] = scaled(2. * value + 1.).to_dict()
    built = build(records)
    candidate, = built['candidates']
    assert candidate['parameter_relations']
    assert candidate['purpose'] == 'exact_edit_map'
    assert all(item['purpose'] == 'exact_edit_map' for item in candidate['parameter_relations'])
    assert not candidate['structural_change']


def test_basis_permutation_is_parameter_rebinding_instead_of_structure_discovery():
    lag = Program('lag', name='u', lag=1)
    before = binary('add', binary('mul', constant(2.), feature()),
                    binary('mul', constant(3.), lag))
    after = binary('add', binary('mul', constant(4.), lag),
                   binary('mul', constant(5.), feature()))
    record = revision('paired', after, before_program=before.to_dict())
    from hegel_machine.cognition.revision_relations import _revision_change_kind

    original = deepcopy((before, after, record))
    budget = SearchBudget(2000)
    diagnosed = _revision_change_kind(before, after, record, budget)
    assert diagnosed['status'] == 'known' and diagnosed['kind'] == 'parameter_rebinding'
    mapping = diagnosed['parameter_role_correspondence']['mappings']
    assert {tuple(item['source_path']): tuple(item['target_path']) for item in mapping} == {
        (0, 0): (1, 0), (1, 0): (0, 0)}
    assert (before, after, record) == original
    assert 0 < budget.spent <= budget.limit


def test_ambiguous_or_unfunded_role_correspondence_stays_unknown():
    before = binary('add', binary('mul', constant(2.), feature()),
                    binary('mul', constant(3.), feature()))
    after = binary('add', binary('mul', constant(4.), feature()),
                   binary('mul', constant(5.), feature()))
    declaration = component('current', before)
    result = parameter_family_correspondence(before, after, declaration['role_signature'],
        declaration['output_unit'], SearchBudget(2000))
    assert result['status'] == 'UNKNOWN'
    assert result['reason'] == 'unique_complete_parameter_role_correspondence_required'
    cut = parameter_family_correspondence(before, after, declaration['role_signature'],
        declaration['output_unit'], SearchBudget(1))
    assert cut['status'] == 'UNKNOWN' and cut['actual_compute_cost'] == 1


def test_unknown_same_token_recombination_remains_a_fittable_proposal():
    # abs(a*u) versus a*abs(u) has the same operator/name multiset. Coefficient
    # signs can change its meaning, so absence of a linear-role proof is UNKNOWN.
    records = []
    for index, value in enumerate((2., 3.)):
        before = Program('abs', children=(binary('mul', constant(value), feature()),))
        after = binary('mul', constant(-value), Program('abs', children=(feature(),)))
        records.append(revision(str(index), after, before_program=before.to_dict(),
            counterexamples=[{'source_ref': {'file': 'recombined_paid.json', 'json_path': [index]},
                'observation': {'features': {'u': -2.}, 'target': -2. * value, 'history': []},
                'tolerance': 1e-10}]))
    built = build(records)
    candidate, = built['candidates']
    assert candidate['purpose'] == 'parameter_proposal'
    assert candidate['source_change_kind'] == 'UNKNOWN' and candidate['structural_change'] is None
    assert all(item['before_exact'] and item['after_exact'] for item in candidate['source_reconstruction'])
    checks = deepcopy(target_checks())
    for check in checks:
        check['observation']['target'] = -7. * abs(check['observation']['features']['u'])
    parent = component('actual_parent', Program('abs', children=(binary('mul', constant(5.), feature()),)))
    result = propose_revision(candidate, parent, [], allocation={'compute': 5000},
        evidence_checks=checks, fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1})
    assert result['proposals']
    assert min(item['fitting']['training_loss'] for item in result['proposals']) < 1e-20


def test_actual_common_transfer_registers_structure_fit_and_returns_parameter_author_work(tmp_path):
    from hegel_machine.cognition.question_routing import register_theories
    from hegel_machine.cognition.theory_consolidation import _save
    from test_question_routing import CONTEXT, theory
    from test_theory_consolidation import agent_scene

    agent, lab = agent_scene(tmp_path)
    candidates = [{**build(sources(structural=structural))['candidates'][0],
                   'id': f'controlled:{structural}', 'version': 1} for structural in (True, False)]
    agent.theory_consolidation = {'candidates': {item['id']: item for item in candidates}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('actual_parent', parent)])
    checks = [{**check, 'context_ref': deepcopy(CONTEXT)} for check in target_checks()]
    arguments = {'operator': 'theory_transfer', 'candidate_id': candidates[0]['id'],
        'theory_id': 'controlled_fit', 'parent_theory_id': 'actual_parent',
        'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 6000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    before = deepcopy(agent.costs)
    result = agent._execute('run', deepcopy(arguments), lab)
    registered = [entry for entry in result['generated_theories'] if entry['status'] == 'candidate_registered']
    assert registered and min(entry['binding']['fitting']['training_loss'] for entry in registered) < 1e-20
    assert all(entry['theory_id'] in agent.theory_index for entry in registered)
    assert agent.costs['compute_cost'] - before['compute_cost'] == result['new_compute_cost'] <= 6000
    assert agent.costs['observation_cost'] == before['observation_cost']
    assert agent.theory_index['actual_parent']['program'] == parent.to_dict()
    arguments.update(candidate_id=candidates[1]['id'], theory_id='needs_fitting')
    arguments.pop('fitting')
    missing = agent._execute('run', arguments, lab)
    assert not missing['generated_theories']
    assert missing['reason'] == 'whole_after_fitting_required_for_parameter_proposal'
    assert missing['author_work'][0]['reason'] == missing['reason']
    assert missing['author_work'][0]['required'] == [
        'operations', 'risk_quantile', 'max_history', 'max_structures']
