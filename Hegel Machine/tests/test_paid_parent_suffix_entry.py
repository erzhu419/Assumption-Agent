"""Paid structural research is reused before one actual bound continuation."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from scripts import stage_partition_method as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.research_programs import replay_research_program
from test_parent_bound_staged_replay import fixture as method_fixture


def fixture(tmp_path, monkeypatch):
    method, training, feedback = method_fixture()
    source, output = tmp_path / 'source', tmp_path / 'suffix'
    parent = replay_research_program(method['staged_research']['baseline'], training, SearchBudget(100000))
    rows = [*training, *feedback]
    refs = []
    for name, observed in (('training', training), ('working', feedback)):
        save(source / (name + '.json'), {'observations': [asdict(row) for row in observed]})
        refs.append(runner.ref(source / (name + '.json')))
    save(source / 'public.json', asdict(TaskSpec('paid_suffix', 'Fixed toy source', tuple(training[0].features), (), ())))
    save(source / 'prepared_state.json', {'training_refs': [refs[0]], 'validation_refs': [refs[1]],
        'fallback': 0., 'public_spec_ref': runner.ref(source / 'public.json')})
    save(source / 'frozen_methods.json', {'Pi3': method['staged_research']['baseline'],
                                        'Pi1': method['staged_research']['continuation']})
    old = {'candidate': Candidate('old', Program('const', value=100.), 'old')}
    old_ref = runner.ref(source / 'old.json')
    save(source / 'old.json', encode_entry(old))
    save(source / 'common_comparison.json', {'full_comparison': True, 'rows': len(rows),
        'data_refs': refs, 'scores': [{'candidate_ref': old_ref,
                                     'loss': loss(old['candidate'].program, rows, 0., .5)}]})
    parallel = {'candidate': Candidate('parallel', Program('const', value=50.), 'parallel')}
    for arm, entry in (('Pi1', parallel), ('Pi3', parent)):
        location = runner.ref(source / arm / 'new_entries.json', 0)
        save(source / arm / 'new_entries.json', [encode_entry(entry)])
        score = loss(entry['candidate'].program, rows, 0., .5)
        save(source / arm / 'comparison.json', {'full_comparison': True,
            'new_scores': [{'candidate_ref': location, 'loss': score}]})
        save(source / arm / 'outcome.json', {'execution_complete': True, 'full_comparison': True,
            'development_rows': len(rows), 'compute_cost': 999, 'selected_ref': location})
    save(source / 'registration.json', {'source_task_ref': {'source_id': 'fixed_paid_source'},
        'feature_mappings': [{}], 'training_rows': len(training), 'quality_threshold': 1e-7,
        'research_config': {'max_candidates': 4, 'minimum_improvement': 1e-7},
        'condition_validation_cohort_refs': [{'source_id': 'old_parallel_cohort'}],
        'frozen_conditions_ref': {'file': 'old_parallel_conditions.json'},
        'condition_prediction_ref': {'file': 'old_parallel_prediction.json'}})
    save(source / 'receipt.json', {'compute_cost': 2500})
    save(tmp_path / 'method.json', method)
    save(tmp_path / 'allocation.json', {'allocation_id': 'registered_suffix',
        'maximum_compute': 75000, 'new_observation_limit': 0})
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    return source, output, runner.ref(tmp_path / 'method.json'), runner.ref(tmp_path / 'allocation.json'), training


def test_paid_prefix_is_not_refit_and_fresh_suffix_returns_actual_sequential_evidence(tmp_path, monkeypatch):
    from hegel_machine.cognition import transfer_partitions
    from hegel_machine.cognition.staged_method_research import _parent_bound_continuation
    from hegel_machine.cognition.staged_method_research import _stage_return_pair
    from hegel_machine.cognition.method_feedback import load_paired_method_returns
    from scripts import return_partition_pair
    from test_native_paired_method_feedback import common, deliver, route

    source, output, method_ref, allocation_ref, training = fixture(tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in source.rglob('*.json')}
    artifact_refs = {}
    for arm, location in (('P0', {**method_ref, 'json_path': ['staged_research', 'baseline']}), ('Pi1', method_ref)):
        path = tmp_path / (arm + '_canonical.json')
        save(path, {'id': 'canonical:' + arm, 'version': 1, 'kind': 'native_research_segment',
            'execution': {'entry': 'replay_research_program', 'research_program_ref': location}})
        artifact_refs[arm] = runner.ref(path)
    cfg = runner.prepare_parent_suffix([source], method_ref, allocation_ref, output, artifact_refs=artifact_refs)
    assert cfg['native_method_artifact_refs'] == artifact_refs
    assert cfg['sources'][0]['pool_size'] == 3
    assert cfg['required_compute'] <= 75000
    assert cfg['sources'][0]['comparison_reserve'] == 4 * 64
    native, called = transfer_partitions.replay_transfer_partition, []

    def replay(body, rows, budget, **kwargs):
        assert body['transfer_partition']['kind'] == 'Pi1', 'the paid Pi3 prefix must never run again'
        assert rows == training and kwargs['paid_initial']['entry']['research_program']['fit_rows'] == len(training)
        called.append(body)
        return native(body, rows, budget, **kwargs)

    monkeypatch.setattr(transfer_partitions, 'replay_transfer_partition', replay)
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    receipt = runner.run_paid(output)
    assert receipt['status'] == 'paid_stage_paths_completed' and len(called) == 1
    assert receipt['compute_cost'] <= cfg['execution_compute_limit']
    actual = load(output / 'source_000/Pi1/outcome.json')
    current_cfg = load(output / 'source_000/registration.json')
    assert all(key not in current_cfg for key in ('condition_validation_cohort_refs',
               'frozen_conditions_ref', 'condition_prediction_ref'))
    staged = actual['staged_research_result']
    assert staged['baseline']['reused'] and not staged['continuation']['reused']
    assert staged['continuation']['all_paid_loss'] < staged['baseline']['all_paid_loss']
    assert actual['full_comparison'] and actual['source_pool_size'] == 3
    assert load(output / 'source_000/P0/outcome.json')['compute_cost'] == 0
    assert len(load(output / 'source_000/common_comparison.json')['scores']) == 3
    measured = _parent_bound_continuation(actual, runner.ref(output / 'source_000/Pi1/outcome.json'),
                                          load(tmp_path / 'method.json')['staged_research'])
    assert measured['status'] == 'measured' and measured['costs']['compute_cost'] > 0
    assert all(path.read_bytes() == content for path, content in originals.items())
    # Ordinary routing consumes its actual result; no separate science or
    # fabricated parallel-Pi1 label is needed for the condition learner.
    prior = tmp_path / 'prior'
    agent = common(prior)
    deliver(agent, tmp_path, 'previous', cheaper=True)
    route(agent)
    monkeypatch.setattr(return_partition_pair, 'paid_pre_call_profile',
                        lambda *args: {'data': {}, 'residual': {'loss': 1.}})
    returned = return_partition_pair.return_pair(output / 'source_000', output / 'source_000/method_return', prior)
    pair = load(output / 'source_000/method_return/pair.json')
    assert pair['data_role'] == 'development' and 'paired_future_ref' not in pair
    assert 'condition_validation_cohort_refs' not in pair
    challenger = load(pair['challenger_ref']['file'])
    assert challenger['method_artifact_ref'] == cfg['native_method_artifact_refs']['Pi1']
    assert challenger['actual_outcome_ref'] == runner.ref(output / 'source_000/Pi1/outcome.json')
    normalized, = load_paired_method_returns(agent, [runner.ref(output / 'source_000/method_return/pair.json')])
    contrast, evidence = _stage_return_pair(agent, normalized, 'challenger', 'baseline',
                                            stage=load(tmp_path / 'method.json')['staged_research'])
    assert contrast is not None and evidence['actual_parent_bound']
    assert evidence['preference']['preference'] == 'challenger'
    assert returned['compute_cost'] <= 3000
    assert runner.run_paid(output)['new_compute_cost'] == 0


@pytest.mark.parametrize('failure', ['wrong_body', 'incomplete_prefix', 'already_solved', 'insufficient_budget'])
def test_suffix_registration_keeps_unusable_sources_unresolved(tmp_path, monkeypatch, failure):
    source, output, method_ref, allocation_ref, _ = fixture(tmp_path, monkeypatch)
    if failure == 'wrong_body':
        methods = load(source / 'frozen_methods.json')
        methods['Pi3']['transfer_partition']['max_attempts'] = 1
        save(source / 'frozen_methods.json', methods)
    elif failure == 'incomplete_prefix':
        outcome = load(source / 'Pi3/outcome.json')
        save(source / 'Pi3/outcome.json', {**outcome, 'execution_complete': False})
    elif failure == 'already_solved':
        compared = load(source / 'common_comparison.json')
        compared['scores'][0]['loss'] = 0.
        save(source / 'common_comparison.json', compared)
    else:
        allocation = load(allocation_ref['file'])
        save(allocation_ref['file'], {**allocation, 'maximum_compute': 100})
    with pytest.raises(ValueError):
        runner.prepare_parent_suffix([source], method_ref, allocation_ref, output)
    assert not output.exists()
