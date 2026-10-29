"""A paid state revision rebases records and delivers the scored executable."""

from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from scripts import run_paid_state_revision as study
from test_native_discrete_state_route import scene
from test_native_representation_routing import PublicOnlyLab


def test_appended_reset_records_rebase_indices_without_editing_paid_sources(tmp_path):
    agent, _ = scene(tmp_path / 'scene')
    old, incoming = deepcopy(agent.observations[:3]), deepcopy(agent.observations[3:])
    incoming[0]['row_indices'] = list(range(len(incoming[0]['rows'])))
    incoming[0]['observation_index'] = 0
    old_copy, incoming_copy = deepcopy(old), deepcopy(incoming)
    combined = study.append_records(old, incoming)
    assert old == old_copy and incoming == incoming_copy
    assert combined[:3] == old
    assert combined[3]['observation_index'] == 3
    assert combined[3]['row_indices'] == list(range(30, 39))
    assert combined[3]['rows'] == incoming[0]['rows']
    assert combined[3]['rows'][0]['history'] == []
    assert combined[3]['rows'][1]['history'] == [incoming[0]['rows'][0]['features']]
    combined[3]['rows'][1]['history'][0]['u'] = -99.
    assert old == old_copy and incoming == incoming_copy


def test_real_paid_route_delivers_scored_predictor_and_bounded_costs(tmp_path, monkeypatch):
    # Source snapshot mechanics have separate controls. This check targets the
    # new runner's data handoff, real research route, exported readout and ledger.
    monkeypatch.setattr(study, 'clone_or_copy', lambda *args, **kwargs: None)
    agent, _ = scene(tmp_path / 'scene')
    source_files = []

    def reference(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        source_files.append((path, path.read_bytes()))
        return {'file': str(path), 'json_path': []}

    incoming = deepcopy(agent.observations[3])
    incoming['observation_index'] = 0
    incoming['row_indices'] = list(range(9))
    registration = {
        'activity_seconds': 30, 'research_compute_cap': 30000,
        'mse_threshold': 1e-10, 'max_candidates': 4, 'expected_rows': 39,
        'diagnostic_parent_id': 'old', 'validation_records': [1],
        'paid_records_ref': reference('old_records.json', agent.observations[:3]),
        'append_record_refs': [reference('new_record.json', incoming)],
        'public_spec_ref': reference('public.json', PublicOnlyLab().public_spec()),
        'parents': [{'candidate_id': 'old', 'program_ref': reference('old_program.json',
            Program('const', value=0.).to_dict())}],
        'counterexample_ref': reference('counterexample.json', {'status': 'all_refuted'}),
        'prior_costs_ref': {'file': 'paid_prior_costs.json', 'json_path': []},
    }
    output = tmp_path / 'run'
    result = study.run(registration, output)
    assert result['status'] == 'completed'
    assert result['native_status'] == 'native_representation_draft_selected'
    assert result['split']['validation_records'] == [1]
    assert result['split']['training_records'] == [0, 2, 3]
    assert result['selected']['export_matches_native_mse']
    assert not result['parent_replaced'] and not result['independent_confirmation']
    assert result['full_archive_comparison'] == 'not_run'
    native = load_native_representation_result(result['native_ref'], output / 'workspace')
    old_score = native['retained_comparison'][0]
    assert result['selected']['exported_development_mse'] < old_score['development_mse']
    assert any(item['program']['op'] == 'state_replay' for item in native['candidates'])
    assert any(item['direction'] == 'discrete_state' and item['compute_cost'] > 0
               for item in native['search_directions'])
    assert result['costs']['compute_cost'] == native['compute_cost'] + 2 * 39
    assert result['costs']['compute_cost'] <= registration['research_compute_cap'] + 2 * 39
    assert result['costs']['representation_compute'] == result['costs']['delivery_compute'] == 39
    assert result['costs']['new_observations'] == result['costs']['observation_cost'] == 0
    assert result['costs']['model_calls'] == 0
    selected = json.loads((output / 'selected_predictions.json').read_text())
    assert selected['rows'] == len(selected['predictions']) == 39
    assert selected['data_role'] == 'exposed_development'
    assert json.loads((output / 'summary.json').read_text()) == result
    assert all(path.read_bytes() == original for path, original in source_files)
