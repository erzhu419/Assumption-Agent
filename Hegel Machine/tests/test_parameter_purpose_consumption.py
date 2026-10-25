"""A parameter initializer changes an actual fit without confirming an edit map."""

from copy import deepcopy

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.parameter_roles import describe_parameter_roles
from hegel_machine.cognition.revision_relations import propose_revision
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, component, constant, edit_candidate, feature


def proposal_candidate(purpose):
    candidate = edit_candidate()
    role = describe_parameter_roles(constant(3.), [[]], {}, 'V', SearchBudget(100))['roles'][()]
    candidate['parameter_relations'] = [{'output_hole': 'hole_1',
        'program': constant(3.).to_dict(), 'purpose': purpose,
        'parameter_coordinate': 'canonical_basis', 'input_roles': {}, 'output_role': role,
        'input_holes': [], 'evidence_inputs': []}]
    return candidate


def derived(result):
    return [proposal for proposal in result['proposals']
            if proposal['bindings']['hole_1'].get('parameter_relation')]


def test_initialization_enters_existing_whole_after_fit_and_keeps_exact_map_fixed():
    parent = binary('mul', constant(5.), feature())
    declaration, checks = component('parent', parent), examples(parent, offset=7.)
    fitting = {**FITTING, 'operations': ['refit'], 'max_structures': 1}
    results = {}
    for purpose in ('exact_edit_map', 'parameter_proposal'):
        candidate = proposal_candidate(purpose)
        original = deepcopy((candidate, declaration, checks))
        result = propose_revision(candidate, declaration, [], allocation={'compute': 5000},
            evidence_checks=checks, fitting=fitting)
        results[purpose] = derived(result)
        assert results[purpose]
        assert (candidate, declaration, checks) == original
        assert not result['support_updated'] and not result['independent_confirmation']
    exact = results['exact_edit_map']
    assert all(Program.from_dict(item['program']).children[1].value == 3. for item in exact)
    assert all(item['fitting']['training_loss'] > 1. for item in exact)
    fitted = min(results['parameter_proposal'], key=lambda item: item['fitting']['training_loss'])
    assert fitted['fitting']['training_loss'] < 1e-20
    assert abs(Program.from_dict(fitted['program']).children[1].value - 7.) < 1e-10
    attribution = fitted['bindings']['hole_1']['parameter_relation']
    assert attribution['predicted_parameter'] == 3.
    assert attribution['world_refit'] and not attribution['exact_edit_map_confirmed']
    assert attribution['scope'] == 'initialization_then_whole_after_fit_not_exact_map_support'


def test_parameter_proposal_requires_a_declared_whole_after_fit():
    parent = binary('mul', constant(5.), feature())
    result = propose_revision(proposal_candidate('parameter_proposal'), component('parent', parent),
        [], allocation={'compute': 5000}, evidence_checks=examples(parent))
    assert not derived(result)
    assert any(item['reason'] == 'whole_after_fitting_required_for_parameter_proposal'
               for item in result['excluded'])


def test_world_fit_cannot_change_an_exact_maps_basis_while_leaving_its_slot_fixed():
    from hegel_machine.cognition.revision_relations import _exact_parameter_roles_preserved

    mapped, free = Program('arg', name='mapped'), Program('arg', name='free')
    definition = Definition('actual_basis', ('mapped', 'free'),
        binary('mul', mapped, binary('mul', free, feature())))
    candidate = {'parent_holes': [], 'definition': definition.to_dict()}
    parent = component('parent', feature())
    # A canonical coefficient of six was applied as slot three because the
    # actual, initially bound basis was 2*u. The free scale is another slot.
    source_role = describe_parameter_roles(binary('mul', constant(6.), feature()), [(0,)],
        parent['role_signature'], parent['output_unit'], SearchBudget(100))['roles'][(0,)]
    attribution = {'purpose': 'exact_edit_map', 'output_role': source_role,
        'target_coordinate_scale': 2., 'predicted_parameter': 6., 'canonical_prediction': 6.,
        'applied_parameter': 3., 'index': 0}
    prepared = {'mapped': {'_program': constant(3.), 'parameter_relation': attribution},
                'free': {'_program': constant(2.)}}
    excluded, budget = [], SearchBudget(1000)
    assert _exact_parameter_roles_preserved(candidate, parent, prepared, budget, excluded)
    assert not excluded
    # A whole-after fit may move this other free component to four. Keeping
    # mapped=3 now gives canonical coefficient twelve, not the learned six.
    prepared['free'] = {'_program': constant(4.)}
    scored_components = deepcopy(prepared)
    assert not _exact_parameter_roles_preserved(candidate, parent, prepared, budget, excluded)
    assert excluded[-1]['reason'] == 'exact_parameter_basis_changed_during_world_fit'
    assert prepared == scored_components  # Do not repair an already scored Program.
    assert prepared['mapped']['_program'].value == 3.
    # An initializer makes no exact-map claim after world fitting.
    prepared['mapped']['parameter_relation']['purpose'] = 'parameter_proposal'
    assert _exact_parameter_roles_preserved(candidate, parent, prepared, budget, [])
    assert 0 < budget.spent <= budget.limit


def test_consumer_uses_current_parent_and_target_scales_instead_of_source_scale():
    from hegel_machine.cognition.revision_relations import (
        _align_parameter_components, _component, _parameter_components,
    )

    old, new, scale = (Program('arg', name=name) for name in ('old', 'new', 'scale'))
    parent = component('current_parent', binary('mul', constant(3.),
                                              binary('mul', constant(2.), feature())))
    source_input = describe_parameter_roles(binary('mul', constant(3.), feature()), [(0,)],
        parent['role_signature'], parent['output_unit'], SearchBudget(100))['roles'][(0,)]
    source_output = describe_parameter_roles(binary('mul', constant(.7),
        binary('mul', constant(10.), feature())), [(0,)], parent['role_signature'],
        parent['output_unit'], SearchBudget(100))['roles'][(0,)]
    candidate = {'id': 'coordinate_transfer', 'parent_holes': ['old'],
        'holes': [{'name': name, 'output_unit': 'dimensionless'} for name in ('old', 'new', 'scale')],
        'before_pattern': binary('mul', old, binary('mul', constant(2.), feature())).to_dict(),
        'definition': Definition('target_basis', ('new', 'scale'),
            binary('mul', new, binary('mul', scale, feature()))).to_dict(),
        'parameter_relations': [{'output_hole': 'new',
            'program': binary('add', Program('feature', name='old'), constant(1.)).to_dict(),
            'purpose': 'exact_edit_map', 'parameter_coordinate': 'canonical_basis',
            'input_roles': {'old': source_input}, 'output_role': source_output,
            'input_holes': ['old'], 'evidence_inputs': []}]}
    fixed = {'old': component('actual_old_slot', constant(3.), output_unit='dimensionless')}
    originals = deepcopy((candidate, parent, fixed))
    budget, excluded = SearchBudget(2000), []
    generated = _parameter_components(candidate, parent, fixed, budget, excluded=excluded)
    declaration, = generated['new']
    assert declaration['parameter_relation']['parent_inputs'] == {'old': 6.}
    assert declaration['parameter_relation']['predicted_parameter'] == 7.
    assert source_output['coordinate_scale'] == 10.
    chosen = {'new': deepcopy(declaration),
              'scale': component('actual_target_scale', constant(4.), output_unit='dimensionless')}
    prepared = {name: _component(item, budget) for name, item in chosen.items()}
    assert _align_parameter_components(candidate, parent, chosen, prepared, budget, excluded)
    attribution = prepared['new']['parameter_relation']
    assert attribution['canonical_prediction'] == 7.
    assert attribution['target_coordinate_scale'] == 4.
    assert attribution['applied_parameter'] == prepared['new']['_program'].value == 1.75
    assert not excluded
    assert (candidate, parent, fixed) == originals
    # Equal units do not turn a windowed input into the current-input basis.
    window_parent = component('window_parent', binary('mul', constant(3.),
        binary('mul', constant(2.), Program('window_sum', lag=2, children=(feature(),)))))
    excluded = []
    assert not _parameter_components(candidate, window_parent, fixed, budget, excluded=excluded)
    assert excluded[-1]['reason'] == 'current_parent_parameter_role_unaligned'
    assert 0 < budget.spent <= budget.limit
