"""Native export identities survive inherited directories and resumed banks."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition import finite_native_author, structural_inquiry
from hegel_machine.cognition.language import Program


def saved_result():
    # A bookkeeping fixture, not a world relationship or a simulated label.
    return {'status': 'candidates_require_evaluation', 'finite_prior': 'saved fixture',
        'candidates': [{'program': Program('const', value=7.).to_dict(),
                        'fit_mse': 3., 'source': 'saved_fixture'}],
        'proposed_experiments': [], 'fit_evidence': [], 'evidence': []}


def agent_at(workspace, question):
    return SimpleNamespace(workspace=workspace, questions=[{'id': 'task'}, question],
        costs={'compute_cost': 0, 'observation_cost': 0, 'python_execution_seconds': 0.},
        config=SimpleNamespace(compute_budget=10000, observation_budget=10, confirmation_rows=1),
        rows=[], observations=[], evaluated_model=None, _inquiry_stack=[],
        _return_structural_answer=lambda *args: None)


def test_export_does_not_replace_inherited_or_resumed_native_versions(tmp_path, monkeypatch):
    question = {'id': 'local', 'question': 'Test the stored executable',
                'native_condition_research': [{'candidates': []}]}
    agent = agent_at(tmp_path, question)
    laboratory = SimpleNamespace(public_spec=lambda: {'controls': ['u']})
    monkeypatch.setattr(structural_inquiry, 'research_native_conditions',
                        lambda *a, **k: deepcopy(saved_result()))
    old = []
    for index in [0, 1]:
        path = tmp_path / f'_experiment_models/native_conditions_0001_{index:04d}/_evaluated_model/candidate_0000.py'
        path.parent.mkdir(parents=True)
        path.write_text(f'original_version = {index}\n')
        old.append((path, path.read_bytes()))

    def export(active):
        result = structural_inquiry.execute_native_condition_research(
            active, {'max_compute': 0}, laboratory, active.questions[1])
        entry = active.questions[1]['native_condition_research'][-1]
        return entry['candidates'][0]['model'], entry['round_index'], result

    first, round_, _ = export(agent)
    assert all(path.read_bytes() == text for path, text in old)
    assert first.startswith('_experiment_models/native_conditions_0001_0002/')
    assert round_ == 1
    first_text = (tmp_path / first).read_bytes()
    second, round_, _ = export(agent)
    assert second.startswith('_experiment_models/native_conditions_0001_0003/')
    assert round_ == 2
    assert (tmp_path / first).read_bytes() == first_text

    # A reopened question starts a new bank; occupied execution paths still stay.
    reopened = agent_at(tmp_path, {'id': 'local', 'question': question['question']})
    third, round_, _ = export(reopened)
    assert third.startswith('_experiment_models/native_conditions_0001_0004/')
    assert round_ == 0
    restored = agent_at(tmp_path, deepcopy(question))
    fourth, round_, _ = export(restored)
    assert fourth.startswith('_experiment_models/native_conditions_0001_0005/')
    assert round_ == 3
    assert len({first, second, third, fourth}) == 4
    assert all(path.read_bytes() == text for path, text in old)
    assert (tmp_path / first).read_bytes() == first_text

    # The finite author's real next action still lists inherited and new paths.
    context = {'source_id': 'fixture/local'}
    public_question = {'id': 'local', 'question': question['question'],
        'context_ref': context, 'theory_target': {'context_ref': context, 'output_unit': '1',
            'role_signature': {'u': {'role': 'control', 'unit': '1', 'time': 'current'}}}}
    state = {'questions': [{'id': 'task'}, public_question],
        'operator_results': [{'operator': 'native_condition_research',
            'research_question_id': 'local', 'native_condition_research': {
                'observed_record_count': 1, 'candidates': [{'model': name} for name in [first, second]]}}]}
    monkeypatch.setattr(finite_native_author, '_state', lambda memory: deepcopy(state))
    monkeypatch.chdir(tmp_path)
    inherited = [str(path.relative_to(tmp_path)) for path, _ in old]
    action = finite_native_author.select_actions({'controls': ['u']}, [{'rows': [None]}],
        {'finite_native_author': {'question': public_question, 'predictor_paths': inherited,
            'predictor_catalog_ref': {'file': 'retained.json'}, 'max_compute': 100,
            'max_candidates': 2, 'boundary_resolution': .1}, 'program_results': []})
    assert action['action'] == 'run'
    assert action['arguments']['operator'] == 'research_choice'
    assert action['arguments']['models'] == [*inherited, first, second]
