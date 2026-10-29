"""Paid frozen counterexamples enter the real retained Common route once."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.representation_diagnostics import representation_ambiguity
from hegel_machine.cognition.registered_source_scene import admit_exposed_representation_evidence
from hegel_machine.cognition.registered_source_control import run_registered_representation_research
from test_finite_native_representation_author import scene


def paid_scene(tmp_path):
    agent, lab, contract, _, _ = scene(tmp_path, 'pr3')
    rows, history = [], []
    for value, target in [(0., 0.), (1., 1.), (0., 0.), (1., 1.)]:
        rows.append(asdict(Observation({'u': value}, target, tuple(history))))
        history.append({'u': value})
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps({'rows': rows, 'all_predictors_frozen_before_labels': True,
        'costs_once': {'observation_cost': len(rows), 'model_calls': 0}}))
    evaluated = tmp_path / 'frozen_evaluation.json'
    evaluated.write_text(json.dumps({'frozen_model': {
        'workspace': str(agent.workspace), 'model': 'old.py'}, 'evaluation': {
        'returncode': 0, 'mse': .5, 'representation_diagnostic': representation_ambiguity(
            [[0.] for row in rows], [Observation(row['features'], row['target'], tuple(row['history']))
                                    for row in rows])}}))
    material = {'panel_ref': {'file': str(panel), 'json_path': []},
        'evaluation_ref': {'file': str(evaluated), 'json_path': []},
        'validation_records': [4], 'compute_limit': 20000, 'max_candidates': 2}
    return {'agent': agent, 'lab': lab, 'models': ['old.py']}, material


def test_paid_readout_opens_actual_child_without_repurchase_or_overwriting_parent_score(tmp_path):
    current, material = paid_scene(tmp_path)
    agent = current['agent']
    before_costs, before_parent = deepcopy(agent.costs), deepcopy(agent.evaluated_model)
    before_candidates = deepcopy(agent.research_candidates)
    count, rows = len(agent.observations), len(agent.rows)
    receipt = admit_exposed_representation_evidence(current, material)
    assert agent.costs == before_costs
    assert agent.evaluated_model == before_parent
    assert agent.research_candidates == before_candidates
    assert len(agent.rows) == rows + 4 and len(agent.observations) == count + 1
    assert receipt['new_observations'] == receipt['new_fits'] == receipt['new_scores'] == 0
    assert receipt['split']['validation_records'] == [4]
    assert count in receipt['split']['training_records']
    child = next(q for q in agent.questions if q['id'] == receipt['question_ref']['id'])
    assert child['status'] == 'open' and child['representation_diagnostic']['rows'] == 4
    assert child['representation_evidence_ref']['file'] == material['evaluation_ref']['file']
    assert child['use_sites'][0]['model'] == before_parent['source_model']
    reused = admit_exposed_representation_evidence(current, material)
    assert reused['cache_hit'] and len(agent.rows) == rows + 4
    assert agent.costs == before_costs and current['models'] == ['old.py']


def test_counterexample_trajectory_cannot_become_proposal_holdout_or_mutate_failed_admission(tmp_path):
    current, material = paid_scene(tmp_path)
    agent = current['agent']
    material['validation_records'] = [len(agent.observations)]
    before = deepcopy(agent.observations)
    with pytest.raises(ValueError, match='training development'):
        admit_exposed_representation_evidence(current, material)
    assert agent.observations == before


def test_terminal_fragment_cannot_be_spliced_into_an_old_reset(tmp_path):
    current, material = paid_scene(tmp_path)
    path = tmp_path / 'panel.json'
    data = json.loads(path.read_text())
    data['rows'][0]['history'] = [{'u': .3}]
    path.write_text(json.dumps(data))
    before = len(current['agent'].rows)
    with pytest.raises(ValueError, match='complete reset trajectories'):
        admit_exposed_representation_evidence(current, material)
    assert len(current['agent'].rows) == before


def test_retained_structure_sources_stay_addressed_and_reach_the_declared_plan(tmp_path, monkeypatch):
    from pathlib import Path
    from test_native_representation_inquiry import retained_procedure

    current, material = paid_scene(tmp_path)
    parent, reference = retained_procedure(tmp_path)
    source = Path(reference['file'])
    candidate = json.loads(source.read_text())['candidates'][0]
    candidate['model'] = 'saved.py'
    source.write_text(json.dumps({'unrelated_history': 'x' * 2000000, 'candidates': [candidate]}))
    original_read = Path.read_text

    def no_whole_history(path, *args, **kwargs):
        if path == source:
            pytest.fail('an addressed retained structure decoded its whole old state')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', no_whole_history)
    material.update(retained_research_program_refs=[reference],
                    retained_replay_scope='retained_coordinates')
    receipt = admit_exposed_representation_evidence(current, material)
    assert len(receipt['parent_theory_ids']) == 2
    agent = current['agent']
    child = next(q for q in agent.questions if q['id'] == receipt['question_ref']['id'])
    plan = child['native_representation_plan']
    assert plan['retained_research_program_refs'] == [reference]
    assert plan['retained_replay_scope'] == 'retained_coordinates'
    assert agent.theory_index[receipt['parent_theory_ids'][1]]['program'] == parent.to_dict()


def test_registered_retained_driver_uses_original_account_and_real_route(tmp_path, monkeypatch):
    from hegel_machine.cognition import registered_source_scene as source_module
    current, material = paid_scene(tmp_path)
    agent = current['agent']
    before = deepcopy(agent.costs)
    source = tmp_path / 'source_metadata.json'
    source.write_text(json.dumps({'plan': {'cases': [{'id': 'synthetic'}], 'limits': {}},
        'archive': {}, 'costs': before}))
    reference = lambda key: {'file': str(source), 'json_path': [key]}

    def restore(*args, **kwargs):
        assert kwargs['starting_costs'] == before
        assert kwargs['limits'].compute_budget == before['compute_cost'] + 50000
        agent.config = kwargs['limits']
        return current

    monkeypatch.setattr(source_module, 'restore_retained_common_scene', restore)
    registration = {'execution_admitted': True,
        'source_scene': {'plan_ref': reference('plan'), 'initial_archive_ref': reference('archive'),
            'costs_ref': reference('costs'), 'case_id': 'synthetic',
            'training_ref': {'file': str(tmp_path / 'old/training.json')}},
        'material': {**material, 'exposed_terminal_as_development': True},
        'limits': {'compute': 50000, 'python_seconds': 30., 'activity_seconds': 30.,
            'native_host_seconds': 20., 'new_observation_rows': 0,
            'research_units': 1, 'per_unit_compute': 20000, 'max_candidates': 2},
        'source_costs_ref': reference('costs')}
    result = run_registered_representation_research(registration, tmp_path / 'registered')
    assert result['research_unit_count'] == 1
    assert result['comparison_completed'] == result['comparison_requested']
    assert result['complete_retained_paths'] == 1
    assert result['costs']['compute_cost'] == before['compute_cost'] + result['new_stage_costs']['compute_cost']
    assert result['new_stage_costs']['observation_cost'] == result['new_observations'] == 0
    assert result['new_stage_costs']['model_calls'] == 0
    saved = json.loads((tmp_path / 'registered/final_state.json').read_text())
    child = next(q for q in saved['online_questions'] if q.get('kind') == 'representation'
                 and q.get('development_material_ref'))
    assert child['native_representation_split']['validation_records'] == [4]


@pytest.mark.parametrize('mse,expected_status', [(1., 'open'), (0., 'resolved_on_paid_rows')])
def test_collision_free_return_still_needs_whole_model_quality(tmp_path, monkeypatch, mse, expected_status):
    from hegel_machine.cognition import registered_source_control as driver
    current, material = paid_scene(tmp_path)
    agent, lab = current['agent'], current['lab']
    admitted = admit_exposed_representation_evidence(current, material)
    parent = agent.theory_index[admitted['parent_theory_id']]
    monkeypatch.setattr(driver, '_prepare_fresh_representation_material', lambda *args:
        (agent, lab, [parent], ['old.py'], ['old.py'], {}, [4]))

    def execute(action, arguments, laboratory):
        if arguments.get('operator') == 'research_route':
            return {'selected_model': 'old.py'}
        if action == 'evaluate':
            return {'returncode': 0, 'mse': mse,
                    'representation_diagnostic': {'empirical_mse_floor': 0.}}
        return {'models_requested': 1, 'models_completed': 1}

    monkeypatch.setattr(agent, '_execute', execute)
    registration = {'execution_admitted': True,
        'material': {'exposed_terminal_as_development': True},
        'limits': {'research_units': 1, 'per_unit_compute': 20000, 'max_candidates': 2},
        'source_costs_ref': material['panel_ref']}
    driver.run_registered_representation_research(registration, tmp_path / 'quality_return')
    child = next(q for q in agent.questions if q['id'] == admitted['question_ref']['id'])
    assert child['status'] == expected_status
    assert child['representation_return']['input_distinction_recovered']
    assert child['representation_return']['prediction_criterion_satisfied'] == (mse == 0.)
    if mse:
        assert child['representation_return']['next_research'] == 'response_or_composition_revision'
