"""The paid-counterexample CLI path preserves its full scene through feedback."""

from copy import deepcopy
from dataclasses import asdict

from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from scripts import run_independent_state_method_source as entry
from scripts import run_state_method_pairing as study
from test_completed_state_readout_restore import completed_fixture


def test_paid_counterexamples_reach_shared_pool_pair_and_common_feedback(tmp_path, monkeypatch):
    panel = [Observation({'u': .75}, 0., ()),
             Observation({'u': .8}, 1., ({'u': .75},)),
             Observation({'u': .3}, 1., ({'u': .75}, {'u': .8}))]
    values, Lab, prior, old, _, state, original = completed_fixture(
        tmp_path, monkeypatch, new_panel=panel)
    old_bytes = (prior / '00/workspace/training.json').read_bytes()
    record = {'action': {'kind': 'trajectory', 'sequence': [row.features for row in panel]},
              'rows': [asdict(row) for row in panel], 'row_indices': list(range(3))}
    material_ref = study.save(tmp_path / 'paid_counterexample.json', record)
    extra_costs = {key: 0 for key in old['costs']}
    extra_costs.update(compute_cost=3, observation_cost=3)
    costs_ref = study.save(tmp_path / 'extra_costs.json', extra_costs)
    source_author = study.read(original['finite_native_author_ref'])
    # A restored author's catalogue is an address chain, as in real continuations.
    catalog_ref = study.save(tmp_path / 'restored_catalog.json', {
        'models': list(dict.fromkeys([*source_author['predictor_paths'],
                                      *state['exported_candidate_paths']])),
        'source_catalog_ref': source_author['predictor_catalog_ref']})
    source_author['predictor_catalog_ref'] = catalog_ref
    author_ref = study.save(tmp_path / 'restored_author.json', source_author)
    request = {'training_ref': original['training_ref'], 'finite_native_author_ref': author_ref,
        'append_record_refs': [material_ref], 'additional_paid_cost_refs': [costs_ref],
        'expected_rows': 23, 'restore_confirmation_seed': 992,
        'case_ref': {'file': str(tmp_path / 'inputs.json'), 'json_path': ['case']},
        'public_spec_ref': {'file': str(tmp_path / 'inputs.json'), 'json_path': ['public']},
        'prior_costs_ref': costs_ref, 'mse_threshold': 1e-10,
        'shared_compute_cap': 10000, 'research_compute_cap': 10000,
        'per_cell_compute_cap': 100000, 'per_cell_python_seconds': 120.,
        'per_cell_activity_seconds': 180., 'evaluation_seed': 993, 'evaluation_rows': 20}
    directory = tmp_path / 'paid_pair'
    opened = []

    class CurrentLab(Lab):
        def evaluation_rows(self, *, length, seed):
            assert (length, seed) == (20, 993)
            manifest = read_json_checkpoint(directory / 'study/frozen_outputs.json')
            assert set(manifest['models']) == {'00', '10'}
            assert not manifest['evaluation_labels_opened']
            opened.append(True)
            # The synthetic apparatus has an independently defined reset rule;
            # no predictor or measured score is substituted for these targets.
            rows = []
            for sequence in ([.8, .4, .1, .9, .3], [.7, .8, .3, .1, .4],
                             [.9, .3, .1, .7, .3], [-.4, .8, .2, .1, .9]):
                history, previous = [], 0.
                for index, value in enumerate(sequence):
                    threshold = .8 if index == 0 else (.2 if previous else .6)
                    previous = float(value >= threshold)
                    rows.append(Observation({'u': value}, previous, tuple(history)))
                    history.append({'u': value})
            return rows

    monkeypatch.setattr(entry, 'AnonymousHistoryLab', CurrentLab)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', CurrentLab)
    execute = CommonArchiveInquiryAgent._execute
    method_routes = []

    def execute_without_purchase(agent, action, arguments, lab):
        assert not (action == 'query' and 'actions' in arguments), (
            'the paid continuation must not buy research observations')
        if (action == 'run' and arguments.get('operator') == 'research_route'
                and arguments.get('research_question', {}).get('kind') == 'method_use'):
            method_routes.append(deepcopy(arguments))
        return execute(agent, action, arguments, lab)

    monkeypatch.setattr(CommonArchiveInquiryAgent, '_execute', execute_without_purchase)
    result = entry.run_paid_method_pair(request, directory)
    assert result['status'] == 'completed_development_pairing', result
    assert result['source_role'] == 'same_physical_source_exposed_counterexample_method_pairing'
    assert opened == [True]
    assert len(method_routes) == 1
    restored = study.read(result['restoration_ref'])
    assert restored['total_rows'] == 23 and restored['exposed_rows'] == 3
    shared = study.read(result['shared_comparison_ref'])
    pool = {item['path'] for item in shared['evaluations']}
    assert set(state['exported_candidate_paths']) <= pool
    assert set(source_author['predictor_paths']) <= pool
    assert shared['models_completed'] == shared['models_requested'] == len(pool)
    for item in shared['evaluations']:
        reuse = item['paid_score_reuse']
        if reuse['status'] == 'reused':
            # Common may first assess its incumbent, then reuse that exact
            # enlarged score when the same literal appears in the full pool.
            assert reuse['original_costs']['compute_cost'] == 3
            assert reuse['additional_scoring_compute'] == 0
        else:
            assert reuse['reused_prediction_rows'] == 20
            assert reuse['newly_predicted_rows'] == 3
    assert result['shared_new_costs']['compute_cost'] == 3 * len(pool)
    assert result['shared_new_costs']['observation_cost'] == 0
    for arm in ('00', '10'):
        outcome = read_json_checkpoint(directory / f'study/{arm}/outcome.json')
        assert outcome['execution_complete'] and outcome['registered_attempt_returned']
        assert outcome['loss'] == 0.
        assert outcome['costs']['observation_cost'] == outcome['costs']['model_calls'] == 0
        current = read_json_checkpoint(directory / f'study/{arm}/workspace/research_state.json')
        comparisons = [item for item in current['operator_results']
                       if item.get('operator') == 'research_choice']
        retained = [item for item in comparisons[-1]['evaluations'] if item['path'] in pool]
        assert {item['path'] for item in retained} == pool
        assert all(item['paid_score_reuse']['status'] == 'reused' for item in retained)
        # New scene bindings receive a new method identity; the inherited
        # executable body remains available with its original selection plan.
        checkpoint = read_json_checkpoint(directory / f'study/{arm}/workspace/training.json')
        old_methods = read_json_checkpoint(prior / '00/workspace'
            / old['theory_consolidation']['file'])['candidates']
        methods = read_json_checkpoint(directory / f'study/{arm}/workspace'
            / checkpoint['theory_consolidation']['file'])['candidates']
        assert outcome['method_ref']['id'] not in old_methods
        for identifier, original_method in old_methods.items():
            if original_method.get('kind') == 'python_research_method':
                retained_method = methods[identifier]
                for key in ('version', 'kind', 'execution', 'selection_plan', 'ordinary_source_ref'):
                    assert retained_method.get(key) == original_method.get(key)
    paired = study.read(result['study_ref'])
    assert paired['ordinary_use']['ordinary_research_route_executed']
    assert not paired['ordinary_use']['general_guard_supported']
    assert not paired['ordinary_use']['successors_started']
    assert method_routes[0]['method_action_revision']['pair_refs'] == paired['pair_refs']
    assert (prior / '00/workspace/training.json').read_bytes() == old_bytes
