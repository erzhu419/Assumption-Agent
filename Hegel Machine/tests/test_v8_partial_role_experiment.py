"""Common adaptation of actual typed object roles, without buying observations."""

from copy import deepcopy

from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.object_dependency_methods import propose_partial_role_experiment


def scene():
    names = ('selector', 'qa', 'qb', 'ua', 'ub')
    spec = TaskSpec('public', 'Measured object roles', names, (), (),
        feature_bounds={name: (0., 2.) for name in names}, max_probe_length=8)
    model = {'focus_control': 'selector', 'quantity_controls': {'qa': 'qa', 'qb': 'qb'},
        'drive_links': {'ua': {'quantity': 'qa'}, 'ub': {'quantity': 'qb'}},
        'focus_candidates': {'fa': {'quantities': ['qa', 'qb']},
                             'fb': {'quantities': ['qa', 'qb']}}}
    records = []
    for selector in (0., 1.):
        controls = {name: 1. for name in names}
        controls['selector'] = selector
        records.append({'controls': controls, 'before': {'qa': 1., 'qb': 1.},
            'after': {'qa': 1., 'qb': 1.}, 'target': 1., 'history': [],
            'position': 0, 'probe_id': f'paid:{selector}'})
    sources = [{'model_id': 'old-source', 'source_ref': {'file': 'paid_source.json', 'json_path': []},
        'dependency_methods': [{'status': 'discriminating', 'excitation_role': 'drive',
            'matched_prefix_rows': 0, 'scope': {'silent_resets_with_history_signal': False}}]}]
    return spec, model, records, sources


def propose(spec, model, records, sources, **changes):
    return propose_partial_role_experiment(spec, model, records,
        selected_values={'fa': 0., 'fb': 1.}, sources=sources,
        compute_limit=10000, remaining_rows=changes.get('remaining_rows', 24))


def test_partial_role_proposal_quotes_all_missing_corners_and_keeps_source_scope():
    spec, model, records, sources = scene()
    before = deepcopy((model, records, sources))
    result = propose(spec, model, records, sources)
    assert result['status'] == 'proposed_complete_partial_role_experiment'
    assert len(result['actions']) == result['required_observation_rows'] == 2
    assert result['full_pair_required']
    assert {action['sequence'][0]['selector'] for action in result['actions']} == {0., 1.}
    assert result['relation']['status'] == 'target_pair_unmeasured'
    assert result['compute_cost'] > 0 and result['new_observations'] == 0
    assert result['sources'][0]['source_ref'] == sources[0]['source_ref']
    assert result['source_success_threshold'] == 1e-6
    assert not result['object_assignment_transferred'] and not result['world_support_updated']
    assert (model, records, sources) == before


def test_existing_paid_full_pair_is_assessed_but_one_corner_gets_no_credit():
    spec, model, records, sources = scene()
    proposal = propose(spec, model, records, sources)
    for index, action in enumerate(proposal['actions']):
        controls = action['sequence'][0]
        records.append({'controls': controls, 'before': {'qa': 1., 'qb': 1.},
            'after': {'qa': 2., 'qb': 1.}, 'target': 2. + index,
            'history': [], 'position': 0, 'probe_id': f'paid_change:{index}'})
        result = propose(spec, model, records, sources)
        contrasts = [item for item in result['measured_methods']
            if item['excitation_role'] == 'drive' and item['quantity_candidate'] == 'qa']
        if index == 0:
            assert contrasts == []
        else:
            assert contrasts[0]['status'] == 'discriminating'
            assert len(contrasts[0]['paired_probe_ids']) == 2
            assert not result['world_support_updated']


def test_insufficient_full_pair_allowance_is_not_a_single_corner_purchase():
    spec, model, records, sources = scene()
    result = propose(spec, model, records, sources, remaining_rows=1)
    assert result['status'] == 'scoped_unknown'
    assert result['reason'] == 'dependency_pilot_unaffordable'
    assert 'actions' not in result and result['new_observations'] == 0


def test_anonymous_scalar_u_does_not_supply_object_sensor_roles():
    spec = TaskSpec('public', 'Anonymous scalar', ('u',), (), (), feature_bounds={'u': (-1., 1.)})
    result = propose_partial_role_experiment(spec, {}, [], compute_limit=1000, remaining_rows=24)
    assert result['status'] == 'UNKNOWN'
    assert 'model.drive_links' in result['required']
    assert 'paid_before_after_transition_records' in result['required']
    assert result['compute_cost'] == 0 and result['new_observations'] == 0


def test_missing_actual_transition_stays_unknown_instead_of_defaulting_units_or_roles():
    spec, model, records, sources = scene()
    records[0].pop('before')
    result = propose(spec, model, records, sources)
    assert result['status'] == 'UNKNOWN' and 'paid_transition_record:0' in result['required']
    assert result['compute_cost'] == 0
