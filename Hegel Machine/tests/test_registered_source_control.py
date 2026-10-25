"""Registered controls preserve stage limits and never refit frozen parents."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from hegel_machine.cognition import registered_source_control as control
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program


@pytest.mark.parametrize('resume_phase', [None, 'first_comparison', 'last_research_allocation'])
@pytest.mark.parametrize('model_count,unpaid_score_count', [(1310, 0), (2261, 0), (2261, 2261)])
def test_arm_uses_two_bounded_research_allocations_and_complete_comparisons(
        tmp_path, monkeypatch, resume_phase, model_count, unpaid_score_count):
    """A separate final route must not silently fund a third research pass."""
    @dataclass
    class BudgetConfig:
        compute_budget: int
        execution_seconds: float

    config = BudgetConfig(compute_budget=500100, execution_seconds=300.)
    costs = {'compute_cost': 100, 'python_execution_seconds': 0.,
             'observation_cost': 97, 'model_calls': 0}
    question = {'id': 'native_public_response'}
    calls = []
    structural = {'id': 'outer_native', 'program': Program('const', value=.25).to_dict()}
    opaque = {'id': 'old_python', 'model': 'retained_python.py'}
    old_components = [{'id': 'old0_original', 'program': Program('const', value=-.5).to_dict()}]
    development_components = [{'id': 'new_original', 'program': Program('const', value=.75).to_dict()}]
    old_inputs = {'_retained_parent': {'id': 'old0_parent'},
        '_retained_components': old_components, 'component_theory_ids': ['old0_original']}
    retained_inputs = {'_retained_components': [structural, opaque],
        '_retained_execution_inputs': {'old0': old_inputs}}
    agent = SimpleNamespace(config=config, costs=costs, questions=[question],
        rows=[None] * 97, candidate_paths=[], workspace=tmp_path, theory_index={},
        theory_consolidation={'candidates': {'old0': {}, 'old1': {}, 'new': {}}})

    def execute(action, arguments, lab):
        operator = arguments['operator']
        allowance = agent.config.compute_budget - agent.costs['compute_cost']
        calls.append((operator, allowance, deepcopy(arguments)))
        if operator == 'research_route':
            expected = min(150000, config.compute_budget - 2 - 97 * unpaid_score_count
                           - agent.costs['compute_cost'])
            assert allowance == (0 if resume_phase == 'last_research_allocation' else expected)
            agent.costs['compute_cost'] += allowance
        else:
            assert operator == 'research_choice'
            assert len(arguments['models']) == model_count
            assert agent.config.compute_budget == config.compute_budget - 2
            final = sum(item[0] == 'research_choice' for item in calls) == 2
            agent.costs['compute_cost'] += max(10000, 97 * unpaid_score_count) if final else 10000
        return {'operator': operator, 'status': 'completed'}

    agent._execute = execute
    agent._evaluated_summary = lambda: {'model': '_evaluated_model/model.py'}
    scene = {'agent': agent, 'lab': object(), 'models': [f'model_{i}.py' for i in range(model_count)],
        'resume_phase': resume_phase,
        'parent': {'id': 'source_parent'}, 'retained_components': development_components,
        'source_component_theory_ids': ['new_original'], 'transfer_arguments': {
            'candidate_id': 'new', 'library_path': 'shared/theory_library.json',
            'parent_theory_id': 'source_parent', 'fitting': {}, 'evidence_checks': [],
            'allocation': {'compute': 150000, 'comparison_rows': 97}}}

    def recall(restored, owner, arguments):
        assert resume_phase != 'first_comparison', 'completed recall must not execute again'
        assert restored.config.compute_budget - restored.costs['compute_cost'] == 150000
        assert arguments['_retained_components'] == [structural, opaque]
        per_object = arguments['_retained_execution_inputs']
        assert per_object['old0'] == old_inputs
        assert per_object['old1']['_retained_components'] == [structural]
        assert per_object['old1']['component_theory_ids'] == ['outer_native']
        assert per_object['new']['_retained_components'] == development_components
        assert per_object['new']['component_theory_ids'] == ['new_original']
        calls.append(('frozen_recall', 150000, arguments))
        restored.costs['compute_cost'] += 150000
        restored.config = control.replace(restored.config,
            compute_budget=restored.config.compute_budget - 9603)
        return {'status': 'UNKNOWN', 'search_completed': False}

    monkeypatch.setattr(control.consolidation, 'recall_frozen_theories', recall)
    monkeypatch.setattr('hegel_machine.cognition.question_routing._retained_revision_inputs',
                        lambda *args: deepcopy(retained_inputs))
    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.export_revision_candidates',
                        lambda *args: {'new_exports': 0})
    def saved_exports(restored, owner):
        assert restored is agent and owner is question
        calls.append(('export_saved_work', 0, {}))
        return {'new_exports': 0}

    monkeypatch.setattr(control, '_export_saved_work', saved_exports)
    monkeypatch.setattr(control, '_checkpoint', lambda *args: None)
    monkeypatch.setattr(control, 'quote_joint_paid_scores', lambda restored: {
        'models': model_count, 'paid_rows': 97,
        'reused_score_count': model_count - unpaid_score_count,
        'unpaid_score_count': unpaid_score_count,
        'unpaid_scoring_compute': 97 * unpaid_score_count,
        'paid_execution_refs': [{'model': 'must_not_copy_into_actions.py'}]})
    result = control._arm_run(scene, {'budget': {'per_consumption_compute': 150000}})
    assert [item[0] for item in calls] == [
        'export_saved_work' if resume_phase == 'first_comparison' else 'frozen_recall',
        'research_choice', 'research_route', 'research_choice']
    route_allowance = next(item[1] for item in calls if item[0] == 'research_route')
    assert result['continuation_costs']['compute_cost'] == (
        (0 if resume_phase == 'first_comparison' else 150000) + 10000
        + route_allowance + max(10000, 97 * unpaid_score_count))
    assert result['continuation_costs']['compute_cost'] <= 500000 - 2
    assert result['status'] == 'research_completed'
    route = next(item for item in json.loads((tmp_path / 'registered_actions.json').read_text())
                 if item['action'] == 'research_route')
    assert route['comparison_score_reserve'] == 97 * unpaid_score_count
    assert route['comparison_score_quote']['unpaid_score_count'] == unpaid_score_count
    assert 'paid_execution_refs' not in route['comparison_score_quote']
    if model_count == 2261 and unpaid_score_count == 0 and resume_phase is None:
        assert route_allowance == 150000
    if unpaid_score_count == 2261 and resume_phase is None:
        assert route_allowance == 120681
    assert agent.config is config


def _fake_finished_control(tmp_path, monkeypatch, *, fitted_parent=False):
    """Prepare tiny completed scenes; all real scientific work is replaced."""
    from hegel_machine.cognition import experiment_runtime, research_episodes

    question = {'id': 'native_public_response', 'context_ref': {'source_id': 'same'},
                'theory_target': {'output_unit': '1'}}
    scenes = {}
    for arm in ('remove', 'keep'):
        workspace = tmp_path / arm
        frozen = workspace / '_evaluated_model'
        frozen.mkdir(parents=True)
        source = ('def fit(rows):\n    raise RuntimeError("frozen refit is forbidden")\n'
                  'def predict(row):\n    return 0.25\n' if fitted_parent else
                  export_temporal_program(Program('const', value=.25)))
        (frozen / 'model.py').write_text(source)
        scenes[arm] = {'agent': SimpleNamespace(questions=[deepcopy(question)]),
                       'workspace': workspace}
    monkeypatch.setattr(control, 'prepare_registered_control',
                        lambda *args: (scenes, {'status': 'accepted'}))
    def snapshot(source, destination):
        assert Path(destination).parent.is_dir()

    monkeypatch.setattr(control, 'clone_or_copy', snapshot)

    def arm_result(scene, registration):
        return {'workspace': str(scene['workspace']), 'status': 'research_completed',
            'retained_parent': {'model': '_evaluated_model/model.py'},
            'continuation_costs': {'compute_cost': 0, 'python_execution_seconds': 0.}}

    monkeypatch.setattr(control, '_arm_run', arm_result)
    monkeypatch.setattr(experiment_runtime, 'physical_cpu_groups', lambda *args: [(0,), (2,)])
    def in_process_trials(callback, jobs, workers, *, release_parent):
        # Each real fork owns a copy of this mapping. Restore that tiny map
        # between mocked workers, then exercise the real parent-release hook.
        inherited = dict(scenes)
        results = []
        for job in jobs:
            scenes.update(inherited)
            results.append(callback(job))
        release_parent()
        return results

    monkeypatch.setattr(experiment_runtime, 'run_process_trials', in_process_trials)
    experiments = [{'id': str(i), 'query': {'op': 'Readout', 'record_index': 0, 'step': 0,
        'binding': {'action': {'kind': 'probe'}, 'features': {'u': float(i)}, 'history': []}}}
        for i in range(2)]
    monkeypatch.setattr(research_episodes, 'load_source_values',
                        lambda *args: [{'value': experiments}])
    registration = {'scope': 'same_source', 'measurement': {'public_prediction_bindings_ref': {}},
        'budget': {'additional_compute_per_arm': 500000,
                   'additional_Python_seconds_per_arm': 300.},
        'costs': {'two_source_account_carry_once': {'known_source_compute_total': 1075176},
                  'new_induction_and_development_compute_once': 204329}}
    return registration


def test_frozen_literal_winner_uses_two_public_readouts_without_training(tmp_path, monkeypatch):
    """Actual retained-model paths include _evaluated_model and remain runnable."""
    registration = _fake_finished_control(tmp_path, monkeypatch)
    result = control.run_registered_control(registration, tmp_path, workers=8)
    assert result['status'] == 'completed'
    assert result['common_source_and_development_compute_once'] == 1279505
    assert all(row['continuation_costs']['compute_cost'] == 2 for row in result['arms'])
    assert (tmp_path / 'frozen_parents.json').is_file()


@pytest.mark.parametrize('workers', [1, 2])
def test_control_workers_release_other_admitted_scene_without_restoring_again(
        tmp_path, monkeypatch, workers):
    from hegel_machine.cognition import experiment_runtime

    runtime = experiment_runtime.run_process_trials
    physical_groups = experiment_runtime.physical_cpu_groups
    if workers == 2 and len(experiment_runtime.physical_worker_cpus()) < 2:
        pytest.skip('two physical cores are required for two native workers')
    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, acceptance = control.prepare_registered_control(registration, tmp_path)

    class Agent:
        def __init__(self, questions):
            self.questions = questions

    class Previous(dict):
        pass

    for arm in scenes:
        scenes[arm]['agent'] = Agent(scenes[arm]['agent'].questions)
        scenes[arm]['previous'] = Previous(recovered=True)
    references = {arm: weakref.ref(scene['agent']) for arm, scene in scenes.items()}
    previous = {arm: weakref.ref(scene['previous']) for arm, scene in scenes.items()}

    def arm_result(scene, registration):
        arm = scene['registered_arm']
        other = 'keep' if arm == 'remove' else 'remove'
        assert references[arm]() is scene['agent']
        if workers == 2 or arm == 'keep':
            assert references[other]() is None, 'fork retained the unrelated arm agent'
        else:
            assert references[other]() is not None, 'serial worker still owes its other arm'
        assert previous[arm]() is None, 'runner retained its unused pre-admission result view'
        return {'workspace': str(scene['workspace']), 'status': 'research_completed',
            'retained_parent': {'model': '_evaluated_model/model.py'},
            'continuation_costs': {'compute_cost': 0, 'python_execution_seconds': 0.}}

    monkeypatch.setattr(control, '_arm_run', arm_result)
    monkeypatch.setattr(experiment_runtime, 'run_process_trials', runtime)
    monkeypatch.setattr(experiment_runtime, 'physical_cpu_groups', physical_groups)
    monkeypatch.setattr(control, 'prepare_registered_control',
        lambda *args: pytest.fail('the admitted RAM scenes must not be restored again'))
    result = control.run_registered_control(registration, tmp_path, workers=workers,
        accepted_scene=(scenes, acceptance))
    assert result['status'] == 'completed'
    assert scenes == {} and all(reference() is None for reference in references.values())


def test_frozen_fittable_winner_is_not_loaded_or_refitted(tmp_path, monkeypatch):
    """The strong old Python pool can win; its fit hook must not run after freeze."""
    registration = _fake_finished_control(tmp_path, monkeypatch, fitted_parent=True)
    monkeypatch.setattr('runpy.run_path',
                        lambda *args: pytest.fail('freezing must not invoke a fitted winner again'))
    result = control.run_registered_control(registration, tmp_path, workers=8)
    assert result['status'] == 'completed'
    assert all(row['continuation_costs']['compute_cost'] == 0 for row in result['arms'])


def test_closed_scene_registration_uses_retained_readouts_and_preserves_old_account(
        tmp_path, monkeypatch):
    """The completed-scene control must neither require old source keys nor erase fees."""
    from hegel_machine.cognition import research_episodes

    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, acceptance = control.prepare_registered_control(registration, tmp_path)
    experiments = [{'id': str(i), 'query': {'op': 'Readout', 'record_index': 0, 'step': 0,
        'binding': {'action': {'kind': 'probe'}, 'features': {'u': float(i)}, 'history': []}}}
        for i in range(2)]
    for scene in scenes.values():
        scene['agent'].questions[0]['resolution_experiments'] = deepcopy(experiments)
    original_agents = [scene['agent'] for scene in scenes.values()]
    original_questions = deepcopy([agent.questions for agent in original_agents])
    monkeypatch.setattr(control, 'prepare_registered_control',
        lambda *args: pytest.fail('accepted completed scenes must not be restored again'))
    registration.pop('measurement')
    prior = {'known_source_and_development_compute_once': 1279505,
        'known_scientific_compute_including_sources_once': 1886200,
        'known_compute_including_engineering_acceptance': 2022706,
        'known_billed_Python_seconds': 714.3728449614136,
        'other_D3_costs': 'retained_in_original_ledgers_unknown_not_zero',
        'prior_fault_compute_preserved': 44642}
    prior_path = tmp_path / 'old_costs.json'
    prior_path.write_text(json.dumps(prior))
    original_prior = prior_path.read_bytes()
    induction = {'compute_cost': 4439, 'python_execution_seconds': 0.,
                 'observation_cost': 0, 'model_calls': 0}
    registration['costs'] = {'old_sources_and_all_failures_ref': str(prior_path),
                             'new_mainline_induction_once': induction}
    monkeypatch.setattr(research_episodes, 'load_source_values',
        lambda refs, workspace: [{'value': json.loads(Path(refs[0]['file']).read_text())}])

    result = control.run_registered_control(registration, tmp_path, workers=8,
        accepted_scene=(scenes, acceptance))

    assert result['status'] == 'completed'
    predictions = json.loads((tmp_path / 'frozen_predictions.json').read_text())
    assert all([answer['value'] for answer in arm['answers']] == [.25, .25]
               for arm in predictions.values())
    assert all(row['continuation_costs']['compute_cost'] == 2 for row in result['arms'])
    assert result['common_source_and_development_compute_once'] == 1283944
    assert result['prior_costs_ref'] == str(prior_path)
    assert result['prior_known_scientific_compute_once'] == 1886200
    assert result['prior_known_compute_including_engineering_once'] == 2022706
    assert result['known_scientific_compute_including_sources_once'] == 1890643
    assert result['known_compute_including_engineering_acceptance'] == 2027149
    assert result['known_billed_Python_seconds'] >= prior['known_billed_Python_seconds']
    assert result['new_mainline_induction_once'] == induction
    assert result['other_D3_costs'] == prior['other_D3_costs']
    assert prior_path.read_bytes() == original_prior
    assert [agent.questions for agent in original_agents] == original_questions
    assert scenes == {}, 'the parent must release both admitted live scenes'


def test_closed_scene_public_readouts_must_match_before_research(tmp_path, monkeypatch):
    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, acceptance = control.prepare_registered_control(registration, tmp_path)
    registration.pop('measurement')
    for scene in scenes.values():
        scene['agent'].questions[0]['resolution_experiments'] = [{'id': 'one'}, {'id': 'two'}]
    scenes['keep']['agent'].questions[0]['resolution_experiments'][1]['id'] = 'different'
    monkeypatch.setattr(control, '_arm_run',
        lambda *args: pytest.fail('mismatched public material must not start research'))
    with pytest.raises(ValueError, match='same two public readouts'):
        control.run_registered_control(registration, tmp_path, workers=8,
            accepted_scene=(scenes, acceptance))


def test_completed_control_metadata_acceptance_is_carried_once_outside_scientific_costs(
        tmp_path):
    prior = {'known_source_and_development_compute_once': 1279505,
        'known_scientific_compute_including_sources_once': 1886200,
        'known_compute_including_engineering_acceptance': 2022706,
        'known_billed_Python_seconds': 714.3728449614136,
        'other_D3_costs': 'retained_in_original_ledgers_unknown_not_zero'}
    path = tmp_path / 'retained_account.json'
    path.write_text(json.dumps(prior))
    original = path.read_bytes()
    registration = {'costs': {'old_sources_and_all_failures_ref': str(path),
        'new_mainline_induction_once': {'compute_cost': 4439, 'python_execution_seconds': 0.},
        'restoration_acceptance_compute_once': 15287}}
    results = [{'continuation_costs': {'compute_cost': 146801,
                                     'python_execution_seconds': 24.309401598060504}},
               {'continuation_costs': {'compute_cost': 149032,
                                     'python_execution_seconds': 24.096358267124742}}]
    actual = control._registered_cost_summary(registration, tmp_path, results)
    assert actual['restoration_acceptance_compute_once'] == 15287
    assert actual['known_scientific_compute_including_sources_once'] == 2186472
    assert actual['known_compute_including_engineering_acceptance'] == 2338265
    assert actual['prior_known_compute_including_engineering_once'] == 2022706
    assert actual['known_billed_Python_seconds'] == pytest.approx(762.7786048265989)
    assert actual['other_D3_costs'] == prior['other_D3_costs']
    assert path.read_bytes() == original
    registration['costs'].pop('restoration_acceptance_compute_once')
    without_acceptance = control._registered_cost_summary(registration, tmp_path, results)
    assert without_acceptance['restoration_acceptance_compute_once'] == 0
    assert without_acceptance['known_compute_including_engineering_acceptance'] == 2322978
    assert without_acceptance['known_scientific_compute_including_sources_once'] == 2186472


def test_retained_diagnostics_are_counted_once_without_changing_scientific_costs(tmp_path):
    prior = {'known_source_and_development_compute_once': 1279505,
        'known_scientific_compute_including_sources_once': 1886200,
        'known_compute_including_engineering_acceptance': 2022706,
        'known_billed_Python_seconds': 714.3728449614136,
        'other_D3_costs': 'retained_in_original_ledgers_unknown_not_zero'}
    prior_path = tmp_path / 'retained_account.json'
    prior_path.write_text(json.dumps(prior))
    original = prior_path.read_bytes()
    engineering = [{'compute_cost': 389, 'ref': {'file': 'source_contract_audit/summary.json'}},
        {'compute_cost': 5937, 'ref': {'file': 'parent_reference/static_acceptance.json'}},
        {'compute_cost': 3767, 'ref': {'file': 'parent_reference/first_static_failure.json'}}]
    registration = {'costs': {'old_sources_and_all_failures_ref': str(prior_path),
        'new_mainline_induction_once': {'compute_cost': 4439, 'python_execution_seconds': 0.},
        'restoration_acceptance_compute_once': 15287,
        'retained_engineering_diagnostics_once': deepcopy(engineering)}}
    results = [{'continuation_costs': {'compute_cost': 292147,
                                     'python_execution_seconds': 58.623020461178385}},
               {'continuation_costs': {'compute_cost': 297191,
                                     'python_execution_seconds': 48.92923155822791}}]
    expected_registration = deepcopy(registration)
    actual = control._registered_cost_summary(registration, tmp_path, results)
    assert actual['known_scientific_compute_including_sources_once'] == 2479977
    assert actual['known_compute_including_engineering_acceptance'] == 2641863
    assert actual['restoration_acceptance_compute_once'] == 15287
    assert actual['retained_engineering_diagnostics_compute_once'] == 10093
    assert actual['retained_engineering_diagnostics_once'] == engineering
    assert actual['known_billed_Python_seconds'] == pytest.approx(821.92509698082)
    assert actual['other_D3_costs'] == prior['other_D3_costs']
    assert registration == expected_registration
    assert prior_path.read_bytes() == original


@pytest.mark.parametrize('resume_compute', [199998, 199999])
def test_resume_costs_include_prior_failure_once_within_original_500k(
        tmp_path, monkeypatch, resume_compute):
    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, _ = control.prepare_registered_control(registration, tmp_path)
    prior = {'compute_cost': 300000, 'python_execution_seconds': 31.}
    original = {'compute_cost': 763743, 'python_execution_seconds': 254.96256205474492}
    for scene in scenes.values():
        scene['prior_continuation_costs'] = deepcopy(prior)
        scene['original_starting_costs'] = deepcopy(original)

    def arm_result(scene, registration):
        return {'workspace': str(scene['workspace']), 'status': 'research_completed',
            'retained_parent': {'model': '_evaluated_model/model.py'},
            'starting_costs': {'compute_cost': original['compute_cost'] + prior['compute_cost']},
            'continuation_costs': {'compute_cost': resume_compute, 'python_execution_seconds': 2.}}

    monkeypatch.setattr(control, '_arm_run', arm_result)
    if resume_compute == 199999:
        with pytest.raises(ValueError, match='exceeded its declared compute allowance'):
            control.run_registered_control(registration, tmp_path, workers=8)
        return
    result = control.run_registered_control(registration, tmp_path, workers=8)
    assert result['status'] == 'completed'
    for arm in result['arms']:
        assert arm['continuation_costs']['compute_cost'] == 500000
        assert arm['starting_costs'] == original
        assert arm['prior_failed_continuation_costs'] == prior
        assert arm['this_resume_costs']['compute_cost'] == resume_compute + 2
        assert arm['continuation_costs']['python_execution_seconds'] >= 33.


def test_resumed_control_readout_fees_belong_to_the_current_delta(tmp_path, monkeypatch):
    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, acceptance = control.prepare_registered_control(registration, tmp_path)
    prior = {'compute_cost': 1000, 'python_execution_seconds': 30.}
    original = {'compute_cost': 100, 'python_execution_seconds': 4.}
    for scene in scenes.values():
        scene.update(prior_continuation_costs=deepcopy(prior),
                     original_starting_costs=deepcopy(original))

    def result(scene, registration):
        return {'workspace': str(scene['workspace']), 'status': 'research_completed',
            'retained_parent': {'model': '_evaluated_model/model.py'},
            'continuation_costs': {'compute_cost': 50, 'python_execution_seconds': 2.}}

    monkeypatch.setattr(control, '_arm_run', result)
    summary = control.run_registered_control(registration, tmp_path,
        accepted_scene=(scenes, acceptance))
    for arm in summary['arms']:
        delta = arm['this_resume_costs']
        assert delta['compute_cost'] == 52
        assert delta['python_execution_seconds'] > 2.
        assert arm['continuation_costs']['compute_cost'] == prior['compute_cost'] + delta['compute_cost']
        assert arm['continuation_costs']['python_execution_seconds'] == pytest.approx(
            prior['python_execution_seconds'] + delta['python_execution_seconds'])
        assert arm['starting_costs'] == original
        assert arm['prior_failed_continuation_costs'] == prior


def test_saved_export_view_includes_committed_outputs_and_preserves_original_cursor(monkeypatch):
    previous = {'basis': {'target': {'id': 'same_parent'}},
        'cursor': {'next_candidate_id': 'old1'},
        'result': {'work': [{'candidate_id': 'old1', 'theory_ids': ['existing']} ]}}
    question = {'id': 'native_public_response', 'meaning_version': 1,
                'frozen_recall_execution': previous}
    saved = deepcopy(previous)
    prefix = 'relation_transfer:frozen_recall:native_public_response@1:'
    agent = SimpleNamespace(theory_consolidation={'executions': {
        prefix + 'old0': {'result': {'generated_theories': [{'theory_id': 'committed_before_failure'}]}},
        'relation_transfer:frozen_recall:other@1:old0': {
            'result': {'generated_theories': [{'theory_id': 'unrelated'}]}}}})

    def export(restored, owner):
        assert restored is agent and owner is question
        identifiers = [identifier for work in owner['frozen_recall_execution']['result']['work']
                       for identifier in work['theory_ids']]
        assert identifiers == ['existing', 'committed_before_failure']
        assert owner['frozen_recall_execution']['cursor'] == saved['cursor']
        return {'new_exports': 1}

    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.export_revision_candidates', export)
    assert control._export_saved_work(agent, question) == {'new_exports': 1}
    assert question['frozen_recall_execution'] is previous
    assert previous == saved
