"""Complete a purchased comparison without restarting its source research."""

from copy import deepcopy

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from scripts import run_learned_structure_successor as runner
from scripts.run_native_decision_pair import save_prepared
from test_native_decision_checkpoint import prepared_scene


def test_comparison_transfer_preserves_total_and_confirmation_reserve(tmp_path, monkeypatch):
    previous, output = tmp_path / 'previous', tmp_path / 'continued'
    previous.mkdir()
    old = {'operations': {'baseline': 'program_replay', 'challenger': 'structural_research'},
        'preparation_compute_limit': 90000, 'common_compute_limit': 90005,
        'method_review_compute_limit': 5, 'per_arm_compute_limit': 89952,
        'per_arm_total_compute_limit': 90000, 'confirmation_rows': 48,
        'confirmation_compute_per_arm': 48,
        'source_task_ref': {'file': 'original/registration.json', 'json_path': ['task']}}
    costs = {'compute_cost': 89992, 'observation_cost': 149, 'measurement_cost': 1416}
    runner.save(previous / 'registration.json', old)
    runner.save(previous / 'receipt.json', {'status': 'preparation_incomplete'})
    runner.save(previous / 'common_receipt.json', {'costs': costs})
    runner.save(previous / 'prepared_state.json', {'uncompared': [{}, {}, {}, {}]})
    old_files = {path.name: path.read_bytes() for path in previous.iterdir()}
    monkeypatch.setattr(runner, 'load_successor_scene', lambda path: {'validation': [None] * 24})
    freezes = []
    monkeypatch.setattr(runner, 'freeze_runtime', lambda target, source: freezes.append((target, source)))

    registered = runner.register_continuation(previous, output)

    assert registered['comparison_resume']['compute_required'] == 96
    assert registered['comparison_resume']['per_arm_transfer_to_common'] == 44
    assert registered['preparation_compute_limit'] == 90088
    assert registered['per_arm_compute_limit'] == 89908
    assert registered['per_arm_total_compute_limit'] == 89956
    assert registered['per_arm_total_compute_limit'] - registered['per_arm_compute_limit'] == 48
    assert registered['confirmation_rows'] == registered['confirmation_compute_per_arm'] == 48
    assert registered['method_review_compute_limit'] == 5
    assert registered['common_compute_limit'] + 2 * registered['per_arm_total_compute_limit'] == (
        old['common_compute_limit'] + 2 * old['per_arm_total_compute_limit'])
    assert registered['prior_paid_costs'] == costs
    assert registered['source_task_ref'] == old['source_task_ref']
    assert registered['comparison_resume']['new_fits'] == registered['comparison_resume']['new_observations'] == 0
    assert freezes == [(output, previous)]
    assert {path.name: path.read_bytes() for path in previous.iterdir()} == old_files


def test_complete_paid_preparation_only_scores_pending_and_restores_full_pool(tmp_path, monkeypatch):
    previous, output = tmp_path / 'previous', tmp_path / 'continued'
    previous.mkdir()
    output.mkdir()
    agent, frame, scene = prepared_scene()
    scene['knowledge_base_ref'] = {'file': 'confirmed_method/knowledge_delta.json'}
    square = Program('mul', children=(Program('feature', name='x'), Program('feature', name='x')))
    good = _sum_terms([square], [1.8], .25)
    bad = Program('const', value=100.)
    pending = [
        {'candidate': Candidate('pending_exact', good, 'paid_fit'), 'kind': 'paid_fit', 'training_loss': 0.},
        {'candidate': Candidate('pending_bad', bad, 'paid_fit'), 'kind': 'paid_fit', 'training_loss': 1000.}]
    frame.update(uncompared=pending, comparison_complete=False, remaining_compute=0)
    saved_pool, saved_knowledge = deepcopy(frame['assessed']), deepcopy(agent.library.to_dict())
    save_prepared(previous, agent, frame, scene, [{'action': 'paid_before_resume'}])
    old_files = {path.name: path.read_bytes() for path in previous.iterdir()}
    registration = {'previous_unit': str(previous), 'source_cost_refs': [{'file': 'source/costs.json'}],
        'prior_paid_costs': {'compute_cost': 1234, 'observation_cost': 149, 'measurement_cost': 1416},
        'comparison_resume': {'pending_candidates': 2, 'compute_required': 6}}
    runner.save(output / 'registration.json', registration)
    restored_paths = []

    def paid_scene(path):
        restored_paths.append(path)
        return deepcopy(scene)

    def no_research(*args, **kwargs):
        raise AssertionError('Continuation must not repeat training, knowledge recovery, fitting or acquisition')

    monkeypatch.setattr(runner, 'load_successor_scene', paid_scene)
    monkeypatch.setattr(runner, 'load_retained_knowledge', no_research)
    monkeypatch.setattr(UnifiedResearchAgent, 'prepare_research_decision', no_research)
    monkeypatch.setattr(UnifiedResearchAgent, '_replay_program_queue', no_research)
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', no_research)
    monkeypatch.setattr('hegel_machine.cognition.anonymous_object_lab.AnonymousObjectLab.acquire', no_research)
    predictions, original_predict = [], Program.predict

    def pending_prediction(program, row, **kwargs):
        assert program in (good, bad) and row in scene['validation']
        predictions.append(program)
        return original_predict(program, row, **kwargs)

    monkeypatch.setattr(Program, 'predict', pending_prediction)
    budget = SearchBudget(1240, spent=1234)
    timing = ExecutionTiming(('preparation',))
    restored, actual, returned_scene = runner.complete_paid_preparation(output, registration, budget, timing)

    assert budget.spent == 1240 and predictions == [good] * 3 + [bad] * 3
    assert actual['assessed'][:2] == saved_pool and len(actual['assessed']) == 4
    assert actual['comparison_complete'] and not actual['uncompared'] and actual['terminal']
    assert actual['assessed'][2]['validation_loss'] < 1e-12
    assert actual['prepare_compute_cost'] == 1240 and actual['remaining_compute'] == 0
    assert actual['training'] == frame['training'] and actual['validation'] == frame['validation']
    assert restored.library.to_dict() == saved_knowledge
    assert returned_scene['knowledge_base_ref'] == scene['knowledge_base_ref']
    assert restored_paths == [previous, output]
    assert {path.name: path.read_bytes() for path in previous.iterdir()} == old_files
    ledger = runner.load(output / 'common_receipt.json')
    assert ledger['incremental_compute'] == 6 and ledger['costs']['compute_cost'] == 1240
    assert ledger['costs']['observation_cost'] == 149 and ledger['costs']['measurement_cost'] == 1416
    saved = runner.load(output / 'prepared_state.json')
    assert saved['knowledge_base_ref'] == scene['knowledge_base_ref']
    assert len(saved['assessed']) == 4 and saved['terminal']
