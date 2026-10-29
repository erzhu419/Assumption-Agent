"""A retained native unit must fund its return and preserve paid interruptions."""

from copy import deepcopy
import json
from time import sleep

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.registered_source_control import run_registered_representation_research
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_exposed_representation_material import paid_scene
from test_native_representation_inquiry import retained_procedure


def registered_scene(tmp_path, monkeypatch):
    from hegel_machine.cognition import registered_source_scene as source_module
    current, material = paid_scene(tmp_path)
    before = deepcopy(current['agent'].costs)
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'plan': {'cases': [{'id': 'fixture'}], 'limits': {}},
        'archive': {}, 'costs': before}))
    ref = lambda key: {'file': str(source), 'json_path': [key]}

    def restore(*args, **kwargs):
        assert kwargs['starting_costs'] == before
        current['agent'].config = kwargs['limits']
        return current

    monkeypatch.setattr(source_module, 'restore_retained_common_scene', restore)
    registration = {'execution_admitted': True, 'source_scene': {
        'plan_ref': ref('plan'), 'initial_archive_ref': ref('archive'), 'costs_ref': ref('costs'),
        'case_id': 'fixture', 'training_ref': {'file': str(tmp_path / 'prior/training.json')}},
        'material': {**material, 'exposed_terminal_as_development': True},
        'limits': {'compute': 50000, 'python_seconds': 30., 'activity_seconds': 2.,
            'native_host_seconds': 1., 'new_observation_rows': 0,
            'research_units': 1, 'per_unit_compute': 20000, 'max_candidates': 2},
        'source_costs_ref': ref('costs')}
    return current, registration, before


def test_unfunded_complete_unit_does_not_begin_native_or_reduce_the_pool(tmp_path, monkeypatch):
    current, registration, before = registered_scene(tmp_path, monkeypatch)
    registration['limits'].update(compute=1000, per_unit_compute=1000)

    def no_science(*args):
        raise AssertionError('unfunded research or comparison started')

    monkeypatch.setattr(current['agent'], '_execute', no_science)
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'insufficient_complete_research_budget'
    assert result['complete_unit_quote']['required_compute'] > 1000
    assert result['research_unit_count'] == 0 and result['comparison_completed'] is None
    assert result['costs'] == before
    assert result['complete_retained_paths'] == 1
    assert (tmp_path / 'run/final_state.json').exists()


def test_native_host_cutoff_preserves_actual_paid_fit_and_original_account(tmp_path, monkeypatch):
    current, registration, before = registered_scene(tmp_path, monkeypatch)
    registration['limits']['native_host_seconds'] = .08

    def fit_then_wait(self, spec, rows, budget, actions):
        from hegel_machine.cognition.research_programs import _Fitter
        fitter = _Fitter(rows, budget, .5)
        fitted = fitter.fit((Program('feature', name='u'),))
        assert fitted is not None
        sleep(.2)
        raise AssertionError('native hard window did not stop the unit')

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', fit_then_wait)
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'activity_window_exhausted'
    assert result['phase'] == 'native_unit_1'
    assert result['new_stage_costs']['compute_cost'] > 0
    assert result['new_stage_costs']['observation_cost'] == 0
    assert result['costs']['compute_cost'] > before['compute_cost']
    saved = json.loads((tmp_path / 'run/final_state.json').read_text())
    assert saved['costs'] == result['costs']
    study = json.loads(next((current['agent'].workspace / 'native_representation_research').glob(
        'record_*.json')).read_text())
    assert any(item['fits'] for item in study['interrupted_work']['fitters'])
    assert result['comparison_completed'] is None


def test_activity_cutoff_in_full_comparison_retains_paid_partial_scores(tmp_path, monkeypatch):
    current, registration, before = registered_scene(tmp_path, monkeypatch)
    registration['limits']['activity_seconds'] = .08
    agent = current['agent']

    def interrupted(action, arguments, laboratory):
        if arguments.get('operator') == 'research_route':
            return {'status': 'native_representation_candidates_ready', 'selected_model': None}
        assert arguments['operator'] == 'research_choice'
        assert 'old.py' in arguments['models']
        agent.costs['compute_cost'] += 7
        agent.operator_results.append({'operator': 'research_choice', 'truncated': True,
            'models_requested': len(arguments['models']), 'models_completed': 1,
            'evaluations': [{'path': 'old.py', 'paid_score_ref': 'already_paid_score'}]})
        sleep(.2)
        raise AssertionError('activity hard window did not stop comparison')

    monkeypatch.setattr(agent, '_execute', interrupted)
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'activity_window_exhausted'
    assert result['phase'] == 'complete_strong_pool_comparison'
    assert result['new_stage_costs']['compute_cost'] == 7
    assert agent.operator_results[-1]['evaluations'][0]['paid_score_ref'] == 'already_paid_score'
    saved = json.loads((tmp_path / 'run/final_state.json').read_text())
    assert saved['costs']['compute_cost'] == before['compute_cost'] + 7
    assert result['comparison_completed'] is None and result['complete_retained_paths'] == 1


def test_retained_driver_passes_source_procedures_and_keeps_their_old_models(tmp_path, monkeypatch):
    current, registration, _ = registered_scene(tmp_path, monkeypatch)
    parent, reference = retained_procedure(tmp_path)
    saved = json.loads((tmp_path / 'saved_parent_procedure.json').read_text())
    saved['candidates'][0]['model'] = 'retained.py'
    (tmp_path / 'saved_parent_procedure.json').write_text(json.dumps(saved))
    current['agent']._save('retained.py', export_temporal_program(parent))
    current['models'].append('retained.py')
    registration['material'].update(retained_research_program_refs=[reference],
        retained_replay_scope='retained_coordinates')
    seen = []

    def execute(action, arguments, laboratory):
        if arguments.get('operator') == 'research_route':
            plan = arguments['native_representation_plan']
            assert plan['retained_research_program_refs'] == [reference]
            assert plan['retained_replay_scope'] == 'retained_coordinates'
            assert len(plan['parent_theory_ids']) == 2
            seen.extend(plan['parent_theory_ids'])
            return {'status': 'native_representation_candidates_ready', 'selected_model': None}
        assert {'old.py', 'retained.py'} <= set(arguments['models'])
        return {'models_requested': len(arguments['models']), 'models_completed': len(arguments['models'])}

    monkeypatch.setattr(current['agent'], '_execute', execute)
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'completed' and len(seen) == 2
    assert result['complete_retained_paths'] == 2
    assert result['complete_unit_quote']['max_exports_per_unit'] == 7


def test_shared_preparation_scope_miss_never_rebuys_old_pool_scores(tmp_path, monkeypatch):
    from hegel_machine.cognition import registered_source_control as driver
    current, registration, before = registered_scene(tmp_path, monkeypatch)
    registration['source_scene']['shared_preparation_ref'] = {'file': 'paid_common_summary'}
    monkeypatch.setattr(driver, 'quote_joint_paid_scores', lambda scene: {
        'unpaid_scoring_compute': 7, 'reused_score_count': 0, 'unpaid_score_count': 1})
    monkeypatch.setattr(current['agent'], '_execute',
        lambda *args: (_ for _ in ()).throw(AssertionError('scope miss repurchased paid work')))
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'shared_preparation_scope_unavailable'
    assert result['costs'] == before and not result['execution_complete']


def test_normal_final_checkpoint_is_inside_activity_window_and_cutoff_preserves_once(tmp_path, monkeypatch):
    from hegel_machine.cognition import registered_source_control as driver
    current, registration, _ = registered_scene(tmp_path, monkeypatch)
    registration['limits'].update(research_units=0, activity_seconds=.08)
    monkeypatch.setattr(current['agent'], '_execute', lambda action, args, lab: {
        'models_requested': len(args['models']), 'models_completed': len(args['models'])})
    original = driver._write
    calls = []

    def slow_first_checkpoint(path, value):
        if path.name == 'training.json':
            calls.append(value['status'])
            if len(calls) == 1:
                sleep(.2)
        return original(path, value)

    monkeypatch.setattr(driver, '_write', slow_first_checkpoint)
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'activity_window_exhausted'
    assert result['phase'] == 'final_checkpoint' and not result['execution_complete']
    assert calls == ['development_completed', 'development_incomplete']
    assert result['timing']['activity_wall_seconds'] >= .08
    assert result['timing']['preservation_seconds'] > 0
    saved = json.loads((tmp_path / 'run/final_state.json').read_text())
    assert saved['status'] == 'development_incomplete'
    assert saved['costs'] == result['costs']


def test_frozen_scene_scope_is_checked_after_paid_material_admission_before_science(tmp_path, monkeypatch):
    current, registration, before = registered_scene(tmp_path, monkeypatch)
    registration['scope'] = {'paid_rows': len(current['agent'].rows) + 5,
        'pool_models': len(current['models']), 'max_exports': 2}
    monkeypatch.setattr(current['agent'], '_execute',
        lambda *args: (_ for _ in ()).throw(AssertionError('mismatched scene began science')))
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'requires_new_quote' and not result['execution_complete']
    assert result['costs'] == before and result['research_unit_count'] == 0
    assert result['scope_mismatch']['actual']['paid_rows'] == len(current['agent'].rows)
    assert result['scope_mismatch']['actual']['pool_models'] == len(current['models'])


def test_extra_actual_exports_are_preserved_and_stop_before_unquoted_comparison(tmp_path, monkeypatch):
    current, registration, before = registered_scene(tmp_path, monkeypatch)
    agent = current['agent']
    registration['scope'] = {'paid_rows': len(agent.rows) + 4,
        'pool_models': len(current['models']), 'max_exports': 1}

    def more_exports(action, arguments, laboratory):
        if arguments.get('operator') != 'research_route':
            raise AssertionError('unquoted candidate pool was compared')
        for index in range(2):
            path = f'extra_{index}.py'
            agent._save(path, export_temporal_program(Program('const', value=index + .1)))
            agent.candidate_paths.append(path)
        agent.costs['compute_cost'] += 13
        return {'status': 'native_representation_candidates_ready', 'selected_model': None}

    monkeypatch.setattr(agent, '_execute', more_exports)
    result = run_registered_representation_research(registration, tmp_path / 'run')
    assert result['status'] == 'requires_new_quote' and not result['execution_complete']
    assert result['comparison_completed'] is None
    assert result['scope_mismatch']['actual']['new_exports'] == 2
    assert result['new_stage_costs']['compute_cost'] == 13
    assert result['costs']['compute_cost'] == before['compute_cost'] + 13
    assert {'extra_0.py', 'extra_1.py'} <= set(agent.candidate_paths)
    assert current['models'] == ['old.py']
    saved = json.loads((tmp_path / 'run/final_state.json').read_text())
    assert saved['costs'] == result['costs']
