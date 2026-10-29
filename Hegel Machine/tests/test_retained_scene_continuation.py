"""Continue one literal C checkpoint without projecting away unfinished work."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from hegel_machine.cognition import method_effect_query, theory_consolidation
from test_prepare_pr3_source_successor import save


def fixture(tmp_path):
    source = tmp_path / 'completed/pr3'
    origin = tmp_path / 'original/target/case_001/pr3'
    carry = {'observation_cost': 145, 'measurement_cost': 0, 'compute_cost': 1000,
             'python_execution_seconds': 10., 'model_calls': 0, 'reported_tokens': 0}
    paid = {**carry, 'compute_cost': 1048, 'python_execution_seconds': 10.03}
    library = {'candidates': {'old': {'id': 'old', 'version': 1}, 'new': {'id': 'new', 'version': 1}},
        'executions': {'original_key': {'cursor': {'layer': 1, 'offset': 2},
                                      'request_basis': {'library_path': 'original.json'}}},
        'transfers': [{'theory_id': 'old_failure', 'status': 'refuted'}]}
    save(source / 'shared/theory_library.json', library)
    manifest = {'source_training_ref': {'file': str(origin / 'training.json'), 'json_path': []},
        'library_views': {'old_ids': ['old'], 'new_ids': ['new']},
        'post97_prediction_contract': {'rows': 48}}
    manifest_ref = save(tmp_path / 'scene/scene_manifest.json', manifest)
    save(source.parent / 'protocol.json', {'source_manifest_ref': manifest_ref})
    training_ref = {'file': str(source / 'training.json'), 'json_path': []}
    save(source.parent / 'summary.json', {'status': 'completed', 'trials': [
        {'arm': 'pr3', 'result_ref': training_ref, 'costs': paid, 'training_only_costs': carry}]})
    public = {'id': 'native_public_response', 'question': 'retained original question',
        'meaning_version': 1, 'frozen_library_path': 'original.json',
        'query_execution': {'old_result': True}, 'routing_execution': {'basis': 'old'},
        'native_condition_research': {'full_content_file': 'research_state.json'}}
    old = {'id': 'parent_revision', 'meaning_version': 1, 'frozen_library_path': 'original.json',
           'theory_target': {'parent_theory_id': 'old_parent'},
           'frozen_recall_execution': {'basis': {'library_path': 'original.json'}}}
    parent = {'id': 'parent_revision', 'meaning_version': 2, 'frozen_library_path': 'original.json',
              'meaning_history': [old], 'theory_target': {'parent_theory_id': 'new_parent'}}
    save(source / 'training.json', {'costs': carry,
        'research_organization': {'finite_author': {'completed_steps': 8, 'contract': {'max_steps': 8}}},
        'online_questions': [public, parent],
        'evaluated_model': {'source_model': 'retained.py', 'mse': .01}})
    plan = {'limits': {}, 'author': {'path': 'shared/author.py', 'max_steps': 8, 'confirm': True},
        'prepared_public_questions': {'case_001': {
            'id': 'native_public_response', 'question': 'original published declaration'}},
        'cases': [{'id': 'case_001', 'kind': 'delay', 'seed': 7, 'confirmation_seed': 8}]}
    save(origin.parent.parent / 'protocol.json', plan)
    save(origin.parent.parent / 'initial_archive.json', {})
    return source, training_ref, carry, paid, library


def test_preparation_uses_small_summary_and_preserves_already_paid_prediction(tmp_path, monkeypatch):
    source, training_ref, carry, paid, _ = fixture(tmp_path)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        assert path.name not in {'training.json', 'research_state.json', 'theory_library.json'}
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    registered = preparation.prepare_retained_scene_continuation(training_ref, tmp_path / 'registered')
    assert registered['retained_training_costs'] == carry
    assert registered['starting_costs'] == paid
    assert registered['additional_compute_per_arm'] == 800000
    assert registered['additional_python_seconds_per_arm'] == 1350
    assert registered['additional_method_steps_per_arm'] == 2
    assert not registered['comparison_is_original_B_C_replay']
    assert registered['prior_consumed_candidates_and_scores_retained_in_both_arms']
    assert not Path(registered['run_output']).exists()


def test_literal_restore_preserves_cursors_parent_and_steps_with_two_future_views(tmp_path, monkeypatch):
    source, reference, carry, paid, library = fixture(tmp_path)
    registered_root = tmp_path / 'registered'
    registered = preparation.prepare_retained_scene_continuation(reference, registered_root)
    agents = []

    class Agent:
        def __init__(self, _provider, workspace, **kwargs):
            assert kwargs['arm'] == 'pr3'
            self.workspace = workspace
            self.workspace.mkdir(parents=True)
            self.files = {}
            self.knowledge, self.config = kwargs['knowledge'], kwargs['config']
            agents.append(self)

        def restore_research(self, original, previous_workspace, _lab, **kwargs):
            assert previous_workspace == source and original['costs'] == paid
            self.questions = deepcopy(original['online_questions'])
            self.costs = deepcopy(original['costs'])
            self._organization_state = deepcopy(original['research_organization'])
            self.evaluated_model = {'source_model': 'retained.py', 'current_mse': .01}
            self.events, self.method_results, self.candidate_paths = [{}] * 28, [{}] * 18, ['retained.py']
            self.rows, self.observations = [None] * 97, [None] * 6
            return {'restored_paid_rows': 97, 'new_observations': 0}

        def _prepare_program_state(self):
            pass

        def _save(self, name, content):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', Agent)
    monkeypatch.setattr(study, '_prepare_organization', lambda *_: (
        {'phase': 'target'}, {'files': {}, 'finite_native_author': {'max_compute': 50000}}))
    monkeypatch.setattr(study, 'archived_paid_trace_references', lambda *_a, **_k: [])
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **_kwargs: None)
    monkeypatch.setattr(preparation, '_retain_scene_score_index',
        lambda *_a, paid_rows: {'status': 'reused', 'available_paid_rows': paid_rows})
    monkeypatch.setattr(method_effect_query, 'retained_research_progress', lambda agent, **_kwargs: {
        'current_parent': deepcopy(agent.evaluated_model), 'full_pool_count': 2377})
    result = preparation.accept_prepared_retained_scene(registered_root / 'retained_scene_registration.json')
    assert result['acceptance']['literal_retained_state']
    assert result['plan']['author'] == {'path': 'shared/author.py', 'max_steps': 10, 'confirm': False}
    for arm, agent in zip(('fixed_pr2', 'pr3'), agents):
        assert agent._organization_state['finite_author']['completed_steps'] == 8
        assert agent._organization_state['finite_author']['contract']['max_steps'] == 10
        assert agent.config.compute_budget == paid['compute_cost'] + 800000
        assert agent.config.execution_seconds == paid['python_execution_seconds'] + 1350
        assert agent.theory_consolidation['executions'] == library['executions']
        assert agent.theory_consolidation['transfers'] == library['transfers']
        active = list(agent.theory_consolidation['candidates'])
        assert active == (['old'] if arm == 'fixed_pr2' else ['old', 'new'])
        owner = agent.questions[1]
        assert owner['theory_target']['parent_theory_id'] == 'new_parent'
        old = owner['meaning_history'][0]
        assert old['theory_target']['parent_theory_id'] == 'old_parent'
        assert old['frozen_recall_execution']['basis']['library_path'] == 'original.json'
        assert old['frozen_library_path'] == owner['frozen_library_path']
        assert Path(old['frozen_library_path']).name == arm + '.json'
        declared = agent.knowledge['finite_native_author']['question']
        assert declared['meaning_version'] == 1
        assert declared['frozen_library_path'] == owner['frozen_library_path']
        assert all(field not in declared for field in
                   ('query_execution', 'routing_execution', 'native_condition_research'))
        assert agent.costs == paid
        frozen = json.loads(Path(old['frozen_library_path']).read_text())
        assert frozen['source_ref']['file'] == str(source / 'shared/theory_library.json')
        assert 'executions' not in frozen
        assert json.loads((agent.workspace / 'materialization_validation.json').read_text())['current_parent_retained']


def test_retained_run_reserves_frozen_prediction_within_the_common_budget(tmp_path, monkeypatch):
    from test_saved_scene_report_recovery import training

    @dataclass
    class Config:
        compute_budget: int
        execution_seconds: float
        confirmation_mse_threshold: float = 1e-10

    source_tree = tmp_path / 'checkout'
    for name in ('src/hegel_machine/__init__.py', 'scripts/run_anonymous_revision_inquiry.py',
                 'scripts/prepare_pr3_source_successor.py', 'scripts/run_online_inquiry_llm.py',
                 'scripts/run_online_inquiry_study.py'):
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen common code\n')
    monkeypatch.setattr(preparation, 'ROOT', source_tree)
    output = tmp_path / 'run'
    carry = training()['prior_research_cost']
    arms = {}
    for name in ('fixed_pr2', 'pr3'):
        workspace = output / name
        workspace.mkdir(parents=True)
        agent = SimpleNamespace(workspace=workspace, candidate_paths=['retained.py'],
            config=Config(carry['compute_cost'] + 300000, carry['python_execution_seconds'] + 1350))
        agent._save = lambda *_: None
        agent.result = lambda status: {**deepcopy(training()), 'status': status}
        arms[name] = {'agent': agent, 'lab': None, 'previous': {}, 'starting_costs': carry,
            'event_start': 28, 'method_start': 18, 'validation': {'acceptance': {'new_costs': {}}}}
    registration = {'source_manifest_ref': {'file': 'actual_manifest.json'},
        'additional_compute_per_arm': 300000, 'additional_python_seconds_per_arm': 1350,
        'additional_method_steps_per_arm': 2}
    scene = {'preparation': tmp_path, 'output': output, 'manifest': {'library_views': {'new_ids': ['new']}},
        'source': tmp_path / 'actual_C', 'plan': {'author': {'path': 'shared/author.py', 'max_steps': 10}},
        'case': {'kind': 'delay'}, 'carry': carry,
        'limits': Config(carry['compute_cost'] + 300000, carry['python_execution_seconds'] + 1350),
        'acceptance': {'status': 'both_arms_accepted_before_scientific_execution', 'initial_method_steps': 8},
        'arms': arms, 'retained_scene_registration': registration,
        'retained_scene_registration_ref': {'file': 'registration.json'},
        'retained_frozen_libraries': {'fixed_pr2': {}, 'pr3': {}}}

    def execute(agent, _lab, plan, _checkpoint, *, native_attempt_limit):
        assert agent.config.compute_budget == carry['compute_cost'] + 300000 - 48
        assert native_attempt_limit == 2 and plan['author']['max_steps'] == 10
        return {'status': 'method_step_limit'}

    monkeypatch.setattr(preparation, '_source_research_steps', execute)
    monkeypatch.setattr(study, '_native_process_trials', lambda run, jobs, workers: [run(job) for job in jobs])
    monkeypatch.setattr(method_effect_query, 'retained_research_progress', lambda *_a, **_k: {'small': True})
    monkeypatch.setattr(method_effect_query, 'registered_successor_effect', lambda *_a, **_k: {'cases': []})
    result = preparation.run_prepared_retained_scene(None, accepted_scene=scene)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['frozen_prediction_compute_reserved_per_arm'] == 48
    assert protocol['additional_compute_per_arm'] == 300000
    assert not result['common_scheduler_repair_is_PR3_effect']
    assert all(row['case'] == 'retained_delay_same_C' for row in result['trials'])
    for name, retained in arms.items():
        assert retained['agent'].config.compute_budget == carry['compute_cost'] + 300000
        actual = json.loads((output / name / 'training.json').read_text())
        assert actual['saved_scene_cutoff']['event'] == 27
        assert (output / name / 'research_progress_after.json').is_file()
