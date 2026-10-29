"""A declared factorized apparatus stays the same across the source lifecycle."""

from dataclasses import asdict
import json

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from scripts import run_independent_state_method_source as entry
from scripts import run_state_method_pairing as study
from scripts.run_frozen_state_confirmation import restore_trajectories
from test_state_method_pairing import paid_fixture


def test_source_prefix_restore_and_evaluation_bind_both_factors(tmp_path, monkeypatch):
    paid, _, _ = paid_fixture(tmp_path / 'historical')
    old = study.read(paid)
    # Historical cases have an id as well as constructor arguments. Factor
    # names belong to the host case, never the public scientific interface.
    case = {'id': 'declared_factorial_cell', 'kind': 'factorized_hysteresis',
            'seed': 31, 'startup_seed': 79}
    controls = [index / 10 for index in range(-10, 11)]
    actions = [{'kind': 'trajectory', 'channels': [], 'sequence': [{'u': value}]}
               for value in controls]
    actions.extend({'kind': 'trajectory', 'channels': [],
                    'sequence': [{'u': value} for value in sequence * 6]}
                   for sequence in ([-.8, .8, .2, -.2], [.8, -.8, .3, -.3]))
    actions_ref = study.save(tmp_path / 'public_prefix.json', actions)
    rows = sum(len(action['sequence']) for action in actions)
    request = {'case': case, 'prefix_actions_ref': actions_ref, 'prefix_rows': rows,
        'parents': old['parents'], 'diagnostic_parent_id': 'constant',
        'evaluation_seed': 9901, 'evaluation_rows': 96, 'validation_records': [22],
        'research_compute_cap': 30000, 'per_cell_compute_cap': 120000,
        'per_cell_activity_seconds': 60, 'per_cell_python_seconds': 30,
        'source_activity_seconds': 120, 'source_python_seconds': 30,
        'source_comparison_compute_upper': rows * len(old['parents']),
        'mse_threshold': 1e-10, 'comparison_scope': 'both_retained_test_programs',
        'data_role': 'independent_successor'}
    expected_lab = AnonymousHistoryLab(kind=case['kind'], seed=case['seed'],
        startup_seed=case['startup_seed'])
    expected_prefix = [row for action in actions
                       for row in expected_lab.execute(action)['rows']]
    expected_evaluation = expected_lab.evaluation_rows(length=96, seed=9901)
    mature_only = AnonymousHistoryLab(kind='hysteresis', seed=case['seed'])
    assert expected_prefix != [row for action in actions for row in mature_only.execute(action)['rows']]
    instantiated = []

    class RegisteredLab(AnonymousHistoryLab):
        def __init__(self, **kwargs):
            instantiated.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(entry, 'AnonymousHistoryLab', RegisteredLab)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', RegisteredLab)
    seen = []

    def inspect_pair(registration, destination):
        assert study.read(registration['evaluation']['case_ref']) == case
        agent, lab, models = study._scene(registration, tmp_path / 'restored')
        assert {name + '.py' for name in models} == {'old.py', 'constant.py'}
        assert [asdict(row) for row in agent.rows] == expected_prefix
        assert lab.public_spec() == expected_lab.public_spec() == mature_only.public_spec()
        assert lab.evaluation_rows(length=96, seed=9901) == expected_evaluation
        # Static options may reference paid evidence, but cannot receive the
        # evaluator's independent factor values as a method-choice feature.
        static = study.read(registration['static_selection_ref'])
        assert 'startup_seed' not in json.dumps(static)
        assert 'factorized_hysteresis' not in json.dumps(static)
        factories = study.make_factories(registration, destination)
        reference = factories['prepare_evaluation']()
        receipt = study.read(reference)
        assert restore_trajectories(receipt['material']) == expected_evaluation
        assert receipt['new_response_rows'] == 96
        seen.append(True)
        return {'status': 'factorized_source_binding_verified'}

    monkeypatch.setattr(entry, 'run_pair', inspect_pair)
    result = entry.run(request, tmp_path / 'source')
    assert seen == [True]
    assert instantiated == [{'kind': case['kind'], 'seed': case['seed'],
        'startup_seed': case['startup_seed']}] * 3
    assert result['status'] == 'factorized_source_binding_verified'
    prefix = study.read(result['prefix_costs_ref'])
    assert [row for record in study.read(prefix['records_ref']) for row in record['rows']] == json.loads(json.dumps(expected_prefix))
    assert result['prefix_costs']['observation_cost'] == rows
    assert result['shared_comparison_costs']['compute_cost'] == rows * 2
