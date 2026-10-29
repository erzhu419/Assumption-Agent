"""A retained scene carries its real case identity across numeric arm paths."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scene
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study


@pytest.mark.parametrize('explicit,relative,expected', [
    (None, 'case_000/archive', 'case_000'),
    ('case_002', 'study/00/workspace', 'case_002'),
    ('case_001', 'study/10/workspace', 'case_001'),
])
def test_real_factory_selects_question_and_paid_index_by_declared_case(
        tmp_path, monkeypatch, explicit, relative, expected):
    plan = {'prepared_public_questions': {f'case_{n:03}': {'id': f'question_{n}'}
                                          for n in range(3)},
            'paid_score_index_refs_by_case': {}}
    for case in plan['prepared_public_questions']:
        source = tmp_path / f'{case}_paid.json'
        source.write_text(json.dumps({'scopes': [], 'assessments': [], 'contexts': []}))
        plan['paid_score_index_refs_by_case'][case] = {'file': str(source), 'json_path': []}
    captured = []

    def construct(provider, workspace, **kwargs):
        captured.append((workspace, kwargs))
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', construct)
    factory, _ = preparation._prepared_callbacks(plan, case_id=explicit)
    knowledge = {'finite_native_author': {}, 'files': {}}
    agent = factory(None, tmp_path / relative, knowledge=knowledge, arm='pr3')
    assert agent.knowledge['finite_native_author']['question'] == plan['prepared_public_questions'][expected]
    index = json.loads(agent.knowledge['files']['_paid_python_scores/_evaluated_model/assessments.json'])
    assert index['historical_score_index_ref'] == plan['paid_score_index_refs_by_case'][expected]
    assert 'case_id' not in captured[0][1]
    assert not (tmp_path / relative).exists()


def test_retained_scene_passes_real_case_to_factory_and_preserves_paid_state(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    workspace = tmp_path / 'study/00/workspace'
    costs = {'compute_cost': 900, 'observation_cost': 2, 'measurement_cost': 0,
             'python_execution_seconds': 3., 'model_calls': 0, 'reported_tokens': 0}
    records = [{'rows': [1, 2]}]
    order = ['new.py', 'old.py']
    comparison = {'operator': 'research_choice', 'paid_rows': 2, 'models_requested': 2,
                  'models_completed': 2, 'arguments': {'models': order},
                  'evaluations': [{'path': path} for path in order]}
    previous = {'costs': deepcopy(costs), 'paid_observations': deepcopy(records)}
    state = {'questions': [{'id': 'case2_response'}], 'operator_results': [comparison],
             'cursor': {'material': 7, 'binding': 3}}
    case = {'id': 'case_002', 'confirmation_seed': 17, 'kind': 'hysteresis', 'seed': 19}
    question = {'id': 'case2_response', 'meaning_version': 4}
    plan = {'prepared_public_questions': {'case_002': question}, 'research_organization': {}}
    knowledge = {'finite_native_author': {'predictor_paths': ['old.py', 'new.py']}, 'files': {}}
    constructed = []

    class Agent:
        def __init__(self, provider, workspace, **kwargs):
            assert kwargs['knowledge']['finite_native_author']['question'] == question
            constructed.append(Path(workspace))
            self.workspace = Path(workspace)
            self.knowledge = kwargs['knowledge']
            self._organization_state = {}
            self._organization_registration = {}

        def restore_research(self, prior, origin, lab, *, confirmation_seed, retained_state):
            assert origin == source and confirmation_seed == 17
            self.costs = deepcopy(prior['costs'])
            self.observations = deepcopy(prior['paid_observations'])
            self.rows = [1, 2]
            self.questions = deepcopy(retained_state['questions'])
            self.operator_results = deepcopy(retained_state['operator_results'])
            self._organization_state['finite_author'] = deepcopy(retained_state['cursor'])
            self.candidate_paths = []
            self.theory_index = {'retained': {'id': 'retained'}}
            return {'new_observations': 0}

    monkeypatch.setattr(scene, 'load_retained_common_checkpoint',
                        lambda path: (deepcopy(previous), deepcopy(state)))
    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', Agent)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(preparation, '_retain_scene_score_index',
                        lambda source, workspace, *, paid_rows: {'available_paid_rows': paid_rows,
                            'retained_assessments': 2, 'retained_scopes': 1})
    monkeypatch.setattr(scene, 'save_theory_library', lambda agent: None)
    restored = scene.restore_retained_common_scene(source, workspace, plan=plan, case=case,
        knowledge=knowledge, limits=SimpleNamespace(), starting_costs=deepcopy(costs))
    assert constructed == [workspace]
    assert restored['models'] == order
    assert restored['agent'].costs == costs
    assert restored['agent'].observations == records
    assert restored['validation']['finite_author_cursor'] == state['cursor']
    assert restored['validation']['retained_score_index']['retained_assessments'] == 2
    assert restored['validation']['new_fits'] == restored['validation']['new_scores'] == 0
    assert restored['validation']['new_predictions'] == restored['validation']['new_observations'] == 0
    assert previous['costs'] == costs and state['cursor'] == {'material': 7, 'binding': 3}
