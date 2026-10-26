"""The registered driver exercises Common on small synthetic paid records."""

from dataclasses import asdict
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.registered_source_control import run_registered_representation_research
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program


def test_registered_development_runs_native_question_and_preserves_full_roster(tmp_path):
    parents = {}
    for name, value in [('one', 0.), ('two', .5)]:
        path = tmp_path / (name + '.py')
        path.write_text(export_temporal_program(Program('const', value=value)))
        parents[name] = str(path)
    rows = []
    for sequence in ([0., 1., 0., 1.], [1., 0., 1., 0.], [.5, 0., .5, 0.]):
        history = []
        for value in sequence:
            rows.append(asdict(Observation({'u': value}, float(len(history)), tuple(history))))
            history.append({'u': value})
    paid = tmp_path / 'paid.json'
    paid.write_text('[]')
    exposed = tmp_path / 'rows.json'
    exposed.write_text(json.dumps(rows))
    roster = tmp_path / 'roster.json'
    roster.write_text(json.dumps(list(parents.values())))
    registration = {'execution_admitted': True,
        'material': {'exposed_terminal_as_development': True, 'paid_records': str(paid),
            'exposed_rows': str(exposed), 'validation_trajectory_indices': [2]},
        'environment': {'kind': 'hysteresis', 'seed': 13}, 'parents': parents,
        'limits': {'compute': 50000, 'python_seconds': 20., 'new_observation_rows': 0,
            'research_units': 1, 'per_unit_compute': 10000, 'max_candidates': 2},
        'retained_rosters': [{'file': str(roster), 'workspace': str(tmp_path)}],
        'source_costs_ref': {'file': 'retained_old_costs.json'}}
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'completed'
    assert result['complete_retained_paths'] == 2
    assert result['comparison_completed'] == result['comparison_requested']
    assert result['new_observations'] == result['provider_calls'] == 0
    assert result['research_unit_count'] == 1
    saved = json.loads((tmp_path / 'run/final_state.json').read_text())
    assert result['questions_ref']['json_path'] == ['online_questions']
    assert any(q.get('native_representation_split', {}).get('validation_records') == [2]
               for q in saved['online_questions'])
    assert result['costs']['compute_cost'] <= 50000
    assert (tmp_path / 'run/final_state.json').is_file()
    records = json.loads((tmp_path / 'run/workspace/laboratory_observations.json').read_text())
    assert [index for record in records for index in record['row_indices']] == list(range(12))
