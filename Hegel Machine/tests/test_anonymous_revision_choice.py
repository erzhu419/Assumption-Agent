"""Paired task and archive wiring, without scientific solutions or model calls."""

from copy import deepcopy
import json
from random import Random

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_anonymous_revision_choice as study
from test_recurrent_method_adoption import STOP_METHOD, source_study


def test_protocol_has_matching_short_evidence_and_distinguishable_long_paths(tmp_path):
    source_study(tmp_path, 'recurrent')
    plan = study.protocol(tmp_path)
    assert len(plan['cases']) == 4 and plan['arms'] == ['code_react', 'hegel']
    assert plan['limits']['execution_seconds'] == 3600.
    assert plan['limits']['max_model_calls'] == 96
    assert plan['limits']['token_budget'] == 10_000_000
    assert plan['limits']['observation_budget'] == 1152
    assert plan['limits']['confirmation_rows'] == 96
    assert plan['model'] == 'gpt-6-luna'
    assert plan['provider']['timeout_seconds'] == 600.
    assert plan['provider']['reasoning_effort'] == 'medium'
    assert {case['apparatus_seed'] for case in plan['cases']} == {71231031, 71232040}
    for block in range(2):
        pair = [case for case in plan['cases'] if case['parameter_pair'] == block]
        assert {case['kind'] for case in pair} == {'finite2', 'recurrent'}
        assert len({case['prefix_seed'] for case in pair}) == 1
        labs = [study.representation_laboratory(case) for case in pair]
        assert labs[0].public_spec() == labs[1].public_spec()
        rng = Random(pair[0]['prefix_seed'])
        actions = [{'kind': 'trajectory', 'channels': [],
                    'sequence': [{'u': rng.uniform(-1., 1.)} for _ in range(3)]}
                   for _ in range(24)]
        values = [[row['target'] for action in actions for row in lab.execute(action)['rows']]
                  for lab in labs]
        assert max(abs(a-b) for a, b in zip(*values)) < 1e-12
        # Host-only task check: this control sequence is never supplied to an actor.
        probe = {'kind': 'trajectory', 'channels': [],
                 'sequence': [{'u': .8}] + [{'u': 0.} for _ in range(23)]}
        values = [[row['target'] for row in lab.execute(probe)['rows']] for lab in labs]
        assert max(abs(a-b) for a, b in zip(*values)) > .01


def test_all_eight_consumers_receive_only_source_program_and_paid_prefix(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'choice'
    source_study(source, 'recurrent')
    original_source = (source / 'author/training.json').read_text()
    requests = []

    class StopProvider:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 1:
                requests.append(deepcopy(request))
                return {'action': 'run', 'arguments': {'operator': 'procedure',
                    'path': METHOD_PATH, 'max_steps': 12, 'confirm': True},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}
            return {'action': 'provider_error', 'error': {'fixture': 'End scripted test'},
                    'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    summary = study.run_choice(source, output, provider_factory=lambda workspace: StopProvider(),
                               auth_mode='scripted_fixture')
    assert len(summary['trials']) == len(requests) == 8
    assert summary['evidence_kind'] == 'scripted_interface_check'
    for request in requests:
        assert request['files'][METHOD_PATH] == STOP_METHOD
        assert request['costs']['observation_cost'] == 72
        assert request['costs']['measurement_cost'] == 0
        assert len(request['laboratory_observations']) == 24
        assert all(len(record['rows']) == 3 for record in request['laboratory_observations'])
        assert 'private_terminal_marker' not in json.dumps(request)
        assert 'private_target_fit' not in json.dumps(request)
        assert 'finite2' not in json.dumps(request)
    for row in summary['trials']:
        assert row['published_method_entry_attempts'] == 1
        assert row['published_method_executed_steps'] == row['published_method_stops'] == 1
        assert not row['published_method_confirmed']
        assert row['published_method_present_at_end']
        assert row['published_method_changed_at_end'] is False
        assert row['experiment_comparison_attempts'] == row['prospective_revision_attempts'] == 0
        assert row['acquired_prospective_revisions'] == 0
        assert not row['terminal']['passed']
        assert row['costs']['model_calls'] == 2
        assert row['costs']['observation_cost'] == 72
    plan = json.loads((output / 'protocol.json').read_text())
    assert plan['provider']['auth_mode'] == 'scripted_fixture'
    assert plan['execution_workers'] == 2
    snapshot = output / 'source_at_run/scripts/run_anonymous_revision_choice.py'
    assert snapshot.read_text() == (study.ROOT / 'scripts/run_anonymous_revision_choice.py').read_text()
    assert (source / 'author/training.json').read_text() == original_source
