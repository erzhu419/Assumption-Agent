"""Retained fit-state references preserve the fit prefix independently of scoring."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import (
    extract_episodes, load_common_sources, load_source_values,
)


FIT_FILES = {
    'fitter_ref': ('research_programs.py', '_Fitter.fit'),
    'linear_fit_ref': ('continuous_state.py', '_linear_fit'),
    'solve_ref': ('synthesis.py', '_solve'),
}


def retained_state(tmp_path):
    workspace = tmp_path / 'study' / 'case_000' / 'pr3'
    workspace.mkdir(parents=True)
    term = Program('feature', name='u').to_dict()
    rounds = []
    for index, count in enumerate((2, 3)):
        recipe = {'kind': 'residual_composition', 'fit_rows': count,
            'selection_steps': [{'operation': 'discover_conditional_coordinates',
                                 'parameter_terms': [term]}],
            'term_nodes': ['term:0'], 'graph': {'nodes': {'term:0': {'expression': term}}}}
        rounds.append({'fit_record_indices': list(range(index + 1)),
            'fit_evidence': list(range(count)), 'observed_record_count': index + 1,
            'candidates': [{'model': f'native_{index}.py',
                'program': Program('const', value=float(index)).to_dict(),
                'source': 'conditional_experiment:0', 'fit_evidence': list(range(count)),
                'research_program': recipe}]})
    question = {'id': 'measured', 'meaning_version': 1,
        'context_ref': {'source_id': 'toy_physical_source'}, 'native_condition_research': rounds}
    residual_scope = {'evaluated_rows': 3}
    before = {'path': 'native_0.py', 'returncode': 0, 'loss_kind': 'mse', 'mse': 2.,
              'paid_residuals': residual_scope}
    after = {'path': 'native_1.py', 'returncode': 0, 'loss_kind': 'mse', 'mse': 1.,
        'paid_residuals': residual_scope, 'retention_decision': {
            'status': 'candidate_retained', 'same_paid_rows': True,
            'incumbent': {'source_model': 'native_0.py', 'evaluated_rows': 3, 'mse': 2.},
            'candidate': {'model': 'native_1.py', 'evaluated_rows': 3},
            'scope': 'Toy recorded same-paid comparison; not confirmation.'}}
    observations = [{'rows': [{}, {}], 'row_indices': [0, 1]},
                    {'rows': [{}], 'row_indices': [2]}]
    events = [{'feedback': {'operator': 'research_choice', 'research_question_id': 'measured',
        'paid_rows': 3, 'evaluations': [before, after], 'compute_cost': 6}}]
    agent = SimpleNamespace(workspace=workspace, questions=[question], theory_index={},
        observations=observations, events=events, costs={'compute_cost': 6},
        config=SimpleNamespace(confirmation_mse_threshold=1e-10))
    return agent


def change(intake):
    changes = [item for episode in intake['episodes'] for item in episode['actual_changes']
               if item['kind'] == 'retained_program_replaced']
    result, = changes
    return result


def save_source(agent):
    state = {'questions': agent.questions, 'theory_index': {},
             'limits': {'confirmation_mse_threshold': 1e-10}}
    (agent.workspace / 'research_state.json').write_text(json.dumps(state))
    training = {'online_questions': [], 'workspace': str(agent.workspace),
        'paid_observations': agent.observations, 'events': agent.events,
        'limits': state['limits'], 'costs': agent.costs, 'status': 'method_stopped'}
    path = agent.workspace / 'training.json'
    path.write_text(json.dumps(training))
    return {'file': str(path), 'json_path': [], 'source_theory_ids': []}


def test_extract_retains_exact_before_and_after_fit_references_without_predictions(tmp_path, monkeypatch):
    agent = retained_state(tmp_path)
    original = deepcopy((agent.questions, agent.events, agent.costs))
    monkeypatch.setattr(Program, 'predict', lambda *args, **kwargs: pytest.fail('Metadata only'))
    actual = change(extract_episodes(agent))
    fits = actual['fit_state_refs']
    for side, index in [('before', 0), ('after', 1)]:
        base = ['questions', 0, 'native_condition_research', index]
        candidate = [*base, 'candidates', 0]
        assert fits[side]['fit_evidence_ref']['json_path'] == [*candidate, 'fit_evidence']
        assert fits[side]['fit_record_indices_ref']['json_path'] == [*base, 'fit_record_indices']
        assert fits[side]['fit_rows_ref']['json_path'] == [*candidate, 'research_program', 'fit_rows']
        assert fits[side]['parameter_terms_refs'][0]['json_path'] == [
            *candidate, 'research_program', 'selection_steps', 0, 'parameter_terms']
        assert 'algorithm_ref' not in fits[side]
    assert actual['scope']['evaluated_rows'] == 3
    assert '"op"' not in json.dumps(actual) and '"features"' not in json.dumps(actual)
    assert (agent.questions, agent.events, agent.costs) == original


def test_common_source_anchors_fit_prefix_and_uses_only_its_declared_frozen_algorithm(tmp_path, monkeypatch):
    agent = retained_state(tmp_path)
    reference = save_source(agent)
    study = agent.workspace.parent.parent
    snapshot = []
    for name, _ in FIT_FILES.values():
        relative = f'src/hegel_machine/cognition/{name}'
        snapshot.append(relative)
        path = study / 'source_at_run' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Frozen toy source, never executed by this test.\n')
    (study / 'protocol.json').write_text(json.dumps({'runtime_snapshot': snapshot}))
    monkeypatch.setattr(Program, 'predict', lambda *args, **kwargs: pytest.fail('Metadata only'))
    actual = change(load_common_sources([reference], tmp_path))
    fits = actual['fit_state_refs']
    refs = [fits[side]['fit_rows_ref'] for side in ('before', 'after')]
    assert [item['value'] for item in load_source_values(refs, tmp_path)] == [2, 3]
    assert actual['scope']['evaluated_rows'] == 3
    for side in ('before', 'after'):
        algorithm = fits[side]['algorithm_ref']
        assert algorithm['protocol_ref'] == {
            'file': str(study / 'protocol.json'), 'json_path': ['runtime_snapshot']}
        for key, (name, symbol) in FIT_FILES.items():
            assert algorithm[key] == {'file': str(study / 'source_at_run' /
                'src/hegel_machine/cognition' / name), 'symbol': symbol}
        assert fits[side]['research_program_ref']['file'] == str(agent.workspace / 'research_state.json')
    assert agent.costs == {'compute_cost': 6}


def test_missing_recipe_or_frozen_runtime_does_not_invent_fit_state_or_reject_change(tmp_path):
    agent = retained_state(tmp_path)
    agent.questions[0]['native_condition_research'][0]['candidates'][0].pop('research_program')
    reference = save_source(agent)
    study = agent.workspace.parent.parent
    (study / 'protocol.json').write_text(json.dumps({'runtime_snapshot': [
        f'src/hegel_machine/cognition/{name}' for name, _ in FIT_FILES.values()]}))
    actual = change(load_common_sources([reference], tmp_path))
    assert 'fit_rows_ref' not in actual['fit_state_refs']['before']
    assert 'fit_evidence_ref' in actual['fit_state_refs']['before']
    assert all('algorithm_ref' not in item for item in actual['fit_state_refs'].values())
    assert actual['scope']['same_paid_rows'] and not actual['scope']['independent_confirmation']
