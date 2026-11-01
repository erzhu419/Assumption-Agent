"""Paid acceptance restores the real sequential outputs without new science."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from scripts import review_paid_object_pool
from scripts import stage_partition_method as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.transfer_partitions import prepare_parent_parameter_revision
from test_parent_bound_staged_replay import fixture as method_fixture


def fixture(tmp_path, monkeypatch):
    body, training, feedback = method_fixture()
    source, producer, output = tmp_path / 'continuation', tmp_path / 'paired/Pi3', tmp_path / 'accepted'
    staged = body['staged_research']
    staged['continuation']['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000}
    parent = replay_research_program(staged['baseline'], training, SearchBudget(100000))
    parent_ref, producer_ref = runner.ref(producer / 'new_entry.json'), runner.ref(producer / 'outcome.json')
    source_method_ref = runner.ref(tmp_path / 'methods.json', 'Pi1')
    bound = prepare_parent_parameter_revision(parent, staged['continuation'],
        source_method_ref=source_method_ref, parent_ref=parent_ref)['research_program']
    continuation = replay_research_program(bound, training, SearchBudget(100000), feedback_rows=feedback)
    rows = [*training, *feedback]
    measured = [loss(entry['candidate'].program, rows, 0., .5) for entry in (parent, continuation)]
    save(tmp_path / 'methods.json', {'Pi3': staged['baseline'], 'Pi1': staged['continuation']})
    save(tmp_path / 'artifact.json', {'execution': {'research_program_ref': runner.ref(tmp_path / 'methods.json', 'Pi3')}})
    for side in ('baseline', 'challenger'):
        save(tmp_path / (side + '.json'), {'actual_outcome_ref': producer_ref,
            'method_artifact_ref': runner.ref(tmp_path / 'artifact.json')})
    save(tmp_path / 'pair.json', {side + '_ref': runner.ref(tmp_path / (side + '.json'))
                                 for side in ('baseline', 'challenger')})
    save(producer / 'new_entry.json', encode_entry(parent))
    save(producer / 'outcome.json', {'execution_complete': True, 'full_comparison': True,
        'selected_ref': parent_ref, 'development_rows': len(rows)})
    save(producer / 'comparison.json', {'full_comparison': True,
        'new_scores': [{'candidate_ref': parent_ref, 'loss': measured[0]}]})
    save(producer.parent / 'prepared_state.json', {'fallback': 0.})
    new_ref = runner.ref(source / 'new_entries.json', 0)
    save(source / 'new_entries.json', [encode_entry(continuation)])
    save(source / 'comparison.json', {'full_comparison': True, 'rows': len(rows),
        'new_scores': [{'candidate_ref': new_ref, 'loss': measured[1]}]})
    save(source / 'bound.json', bound)
    action_ref = runner.ref(source / 'diagnostics.json', 'native_action')
    save(source / 'diagnostics.json', {'native_action': {'full_comparison': True, 'research': {
        'execution_complete': True, 'parent_ref': parent_ref, 'source_method_ref': source_method_ref}}})
    save(source / 'public.json', asdict(TaskSpec('paid', 'Sequential paid scene',
        tuple(training[0].features), (), ())))
    save(source / 'registration.json', {'training_rows': len(training), 'quality_threshold': 1e-7,
        'source_task_ref': {'source_id': 'same_source'}, 'public_spec_ref': runner.ref(source / 'public.json')})
    save(source / 'receipt.json', {'execution_complete': True, 'full_comparison': True,
        'executed_method_ref': runner.ref(source / 'bound.json'), 'method_execution_ref': action_ref,
        'comparison_ref': runner.ref(source / 'comparison.json'), 'compute_cost': 999})
    proposal = {'basis': {'operation': 'revise_then_reidentify',
        'actual_result_ref': runner.ref(source / 'receipt.json'), 'research_parent_ref': parent_ref,
        'source_method_ref': source_method_ref, 'method_execution_ref': action_ref},
        'baseline_execution_ref': producer_ref, 'baseline_input_provenance_ref': runner.ref(tmp_path / 'pair.json'),
        'proposal': {'baseline_research_program': staged['baseline'], 'candidates': [{'research_program': body}]}}
    save(tmp_path / 'proposal.json', proposal)
    save(tmp_path / 'allocation.json', {'allocation_id': 'paid_chain_acceptance',
        'maximum_compute': 50, 'formation_reserve': 0, 'new_observation_limit': 0})
    old = {'candidate': Candidate('older', Program('const', value=100.), 'old'),
           'validation_loss': loss(Program('const', value=100.), rows, 0., .5)}
    pool = [old, dict(parent, validation_loss=measured[0]), dict(continuation, validation_loss=measured[1])]
    scores = {entry['candidate'].program: entry['validation_loss'] for entry in pool}
    monkeypatch.setattr(review_paid_object_pool, 'parent_inputs', lambda supplied:
        ({'full_comparison': True}, {}, rows, [], [], pool, pool[-1], scores))
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    monkeypatch.setattr(runner, 'write_text_incrementally', lambda *args: None)
    return source, output, runner.ref(tmp_path / 'proposal.json', 'proposal', 'candidates', 0, 'research_program'), \
        runner.ref(tmp_path / 'proposal.json'), runner.ref(tmp_path / 'allocation.json'), measured


def test_paid_native_chain_reuses_both_stages_scores_and_complete_pool(tmp_path, monkeypatch):
    from hegel_machine.cognition import transfer_partitions, evaluation
    source, output, method_ref, proposal_ref, allocation_ref, measured = fixture(tmp_path, monkeypatch)
    old = load(source / 'receipt.json')
    method = runner.read_method_reference(method_ref, output)
    scene = runner.parent_bound_paid_scene(source, method, proposal_ref)
    assert [scene['stage_evidence'][name]['joint_loss'] for name in ('baseline', 'continuation')] == measured
    cfg = runner.prepare_paid([source], method_ref, allocation_ref, output, proposal_ref=proposal_ref)
    assert cfg['sources'][0]['pool_size'] == 3
    assert cfg['proposal_ref'] == proposal_ref
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    monkeypatch.setattr(transfer_partitions, 'replay_transfer_partition',
                        lambda *args, **kwargs: pytest.fail('a paid stage was executed again'))
    monkeypatch.setattr(evaluation, 'loss', lambda *args, **kwargs: pytest.fail('a paid score was evaluated again'))
    receipt = runner.run_paid(output)
    actual = receipt['sources'][0]
    assert receipt['status'] == 'paid_stage_paths_completed'
    assert receipt['compute_cost'] == 1  # only the new conditional invocation
    assert actual['full_comparison'] and actual['old_pool_size'] == 3
    assert actual['comparison_compute_cost'] == 0 and actual['selected_loss'] == measured[1]
    assert all(actual['staged_research_result'][name]['reused'] for name in ('baseline', 'continuation'))
    assert not receipt['independent_support'] and receipt['new_observations'] == 0
    assert load(source / 'receipt.json') == old
    assert runner.run_paid(output)['new_compute_cost'] == 0


@pytest.mark.parametrize('changed', ['source_body', 'actual_binding', 'unfinished', 'parallel_parent', 'ambiguous_output'])
def test_missing_real_sequential_dependency_cannot_fall_back_to_parallel_pi1(tmp_path, monkeypatch, changed):
    source, output, method_ref, proposal_ref, allocation_ref, _ = fixture(tmp_path, monkeypatch)
    proposal = load(tmp_path / 'proposal.json')
    if changed == 'source_body':
        frozen = load(tmp_path / 'methods.json')
        frozen['Pi1']['transfer_partition']['max_attempts'] -= 1
        save(tmp_path / 'methods.json', frozen)
    elif changed == 'actual_binding':
        bound = load(source / 'bound.json')
        bound['transfer_partition']['released_groups'] = ['other_group']
        save(source / 'bound.json', bound)
    elif changed == 'unfinished':
        record = load(tmp_path / 'paired/Pi3/outcome.json')
        save(tmp_path / 'paired/Pi3/outcome.json', {**record, 'execution_complete': False})
    elif changed == 'parallel_parent':
        action = load(source / 'diagnostics.json')
        action['native_action']['research']['parent_ref'] = {'file': 'parallel_parent.json'}
        save(source / 'diagnostics.json', action)
    else:
        compared = load(source / 'comparison.json')
        compared['new_scores'].append(deepcopy(compared['new_scores'][0]))
        save(source / 'comparison.json', compared)
    with pytest.raises(ValueError):
        runner.prepare_paid([source], method_ref, allocation_ref, output, proposal_ref=proposal_ref)
    assert not output.exists()


def test_freeze_includes_the_paid_pool_restore_import(tmp_path, monkeypatch):
    import importlib.util
    from scripts import validate_partition_successor

    # The ordinary base runtime already provides review_paid_object_pool's
    # imports. Its newly required module must also exist in the frozen path.
    monkeypatch.setattr(validate_partition_successor, 'freeze_runtime',
        lambda output, source: (output / 'runtime/scripts').mkdir(parents=True))
    output = tmp_path / 'frozen'
    validate_partition_successor.freeze_entry(output, tmp_path / 'previous')
    location = output / 'runtime/scripts/review_paid_object_pool.py'
    spec = importlib.util.spec_from_file_location('frozen_paid_pool_restore', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.parent_inputs)
    assert module.ROOT == output / 'runtime'
