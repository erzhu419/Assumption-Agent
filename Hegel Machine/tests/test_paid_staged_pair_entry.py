"""The paid pair entry preserves same-call staged feedback and actual returns."""

from dataclasses import asdict

import pytest

from scripts import return_partition_pair
from scripts import validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from test_parent_bound_staged_replay import fixture


@pytest.mark.parametrize('complete,mapping_count', [(True, 1), (False, 1), (True, 2)])
def test_paid_pair_executes_bound_stages_and_saves_each_actual_result(tmp_path, monkeypatch,
                                                                    complete, mapping_count):
    body, training, feedback = fixture()
    methods = {'Pi3': body['staged_research']['baseline'], 'staged': body}
    rows, mappings = [*training, *feedback], [{} for _ in range(mapping_count)]
    for name, data in (('training', training), ('feedback', feedback)):
        save(tmp_path / (name + '.json'), {'observations': [asdict(row) for row in data]})
    train_ref, feedback_ref = [runner.ref(tmp_path / (name + '.json')) for name in ('training', 'feedback')]
    old_ref = runner.ref(tmp_path / 'old.json')
    save(tmp_path / 'old.json', encode_entry({'candidate': Candidate('old', Program('const', value=0.), 'old')}))
    save(tmp_path / 'pool.json', {'scores': [{'candidate_refs': [old_ref],
        'mse': sum(row.target ** 2 for row in rows) / len(rows)}]})
    names = {'baseline': 'Pi3', 'challenger': 'staged'}
    quote = runner.paid_pair_quote(methods, mappings, training, feedback, len(rows),
                                   evaluation_arm_names=names, return_reserve=100)
    save(tmp_path / 'public_spec.json', asdict(TaskSpec('paid_stages', 'Synthetic paid data',
        tuple(training[0].features), (), ())))
    save(tmp_path / 'frozen_methods.json', methods)
    save(tmp_path / 'registration.json', {**quote, 'evaluation_arm_names': names,
        'remaining_compute': quote['total_new_compute_upper'], 'previous_compute': 0,
        'source_pool_ref': runner.ref(tmp_path / 'pool.json'), 'training_refs': [train_ref],
        'feedback_refs': [feedback_ref], 'data_refs': [train_ref, feedback_ref],
        'research_config': {'max_candidates': 4, 'minimum_improvement': 1e-7},
        'feature_mappings': mappings, 'active_wall_seconds': 60, 'quality_threshold': 1e-7,
        'source_cost_refs': [], 'original_ledger': {}, 'ledger_scope': 'paid_development',
        'prior_method_review': 'existing_method_topic'})
    monkeypatch.setattr(runner, 'ROOT', tmp_path / 'runtime')
    native, calls, returns = runner.replay_research_program, [], []

    def replay(method, supplied, budget, **kwargs):
        staged = bool(method.get('staged_research'))
        assert supplied == training
        assert kwargs['feedback_rows'] == (feedback if staged else ())
        entry = native(method, supplied, budget, **kwargs)
        if staged:
            assert entry['staged_research_result']['continuation']['parent_stage'] == 'baseline'
            entry['staged_research_result']['execution_complete'] = complete
            # The entry must consume the actual staged verdict even if the
            # underlying bounded fitter itself completed successfully.
            entry['transfer_partition_search']['completed'] = True
            calls.append(entry['staged_research_result'])
        return entry

    monkeypatch.setattr(runner, 'replay_research_program', replay)
    monkeypatch.setattr(return_partition_pair, 'return_pair', lambda *args, **kwargs:
        (returns.append(kwargs) or {'status': 'recorded'}))
    receipt = runner.run_paid_pair(tmp_path)
    actual = load(tmp_path / 'staged/outcome.json')
    assert len(calls) == mapping_count
    assert actual['full_comparison'] and actual['source_pool_size'] == 1
    assert actual['execution_complete'] is complete
    assert [item['staged_research_result'] for item in actual['executions']] == calls
    if mapping_count == 1:
        assert actual['staged_research_result'] == calls[0]
    else:
        assert 'staged_research_result' not in actual  # no arbitrary mapping becomes the branch label
    saved_entries = load(tmp_path / 'staged/new_entries.json')
    assert all(entry['development_feedback_rows'] == len(feedback) for entry in saved_entries)
    assert receipt['status'] == ('paid_methods_completed' if complete else 'incomplete_method_not_negative')
    assert bool(returns) is complete
    assert receipt['compute_cost'] <= quote['total_new_compute_upper']
    assert receipt['new_observations'] == receipt['model_calls'] == 0
