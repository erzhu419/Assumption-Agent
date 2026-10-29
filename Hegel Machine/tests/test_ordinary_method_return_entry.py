"""Paid-return replay uses actual feedback identities and no injected revision."""

from copy import deepcopy
import pytest

from scripts import run_method_condition_cohort as cohort


@pytest.mark.parametrize('different_dependency', [False, True])
def test_ordinary_entry_accumulates_original_returns_and_routes_only_published_questions(
        tmp_path, monkeypatch, different_dependency):
    from scripts import run_state_method_pairing as pairing
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.method_feedback import _read
    from hegel_machine.cognition.research_programs import export_research_selection_program
    from test_theory_consolidation import agent_scene

    def saved(relative, value):
        return cohort.save(tmp_path / relative, value)

    plan = {'ordinary_method_path': 'shared/guard_ordinary.py',
        'condition': Program('ge', children=(Program('feature', name='public_0'),
            Program('const', value=.5))).to_dict(),
        'condition_feature_paths': [{'path': ['observation_count'], 'kind': 'number'}],
        'when_true': {'first_segment': {'current_option': {'id': 'native_representation:whole_state', 'family': 'N'}}},
        'when_false': {}}
    methods = []
    for name in ('P0', 'guard'):
        method = {'id': name, 'version': 1, 'kind': 'python_research_method',
            'execution': {'kind': 'python_select_actions', 'path': f'shared/{name}.py', 'max_steps': 8},
            'selection_plan': (deepcopy(plan) if name == 'guard'
                else {'ordinary_method_path': 'shared/P0_ordinary.py'})}
        artifact = saved(f'{name}/method_artifact.json', method)
        original = tmp_path / name / 'workspace'
        (original / 'shared').mkdir(parents=True)
        (original / method['execution']['path']).write_text(
            export_research_selection_program(method['selection_plan']))
        (original / method['selection_plan']['ordinary_method_path']).write_text(
            'def select_actions(public, observations, memory):\n    return {"stop": True}\n')
        methods.append(artifact)
    sources = []
    for index in range(2):
        profile_ref = saved(f'source_{index}/profile.json', {'observation_count': 3 + index})
        source_ref = saved(f'source_{index}/request.json', {'source_id': f'actual_source_{index}'})
        source_ref['json_path'] = ['source_id']
        outcome = {'method_ref': {'id': 'P0', 'version': 1}, 'method_artifact_ref': methods[0],
            'method_invocation_binding': {'candidate_id': 'P0', 'candidate_version': 1,
                'origin_ref': {'file': f'actual_source_{index}_invocation.json', 'json_path': []}},
            'status': 'completed', 'execution_complete': True, 'profile_ref': profile_ref,
            'start_state_ref': {'file': f'actual_source_{index}_state.json', 'json_path': []},
            'native_representation_returns': [{'record_ref': {'file': f'actual_native_{index}.json'}}],
            'comparison': {'models_requested': 48, 'models_completed': 48},
            'costs': {'compute_cost': 10000 + index, 'observation_cost': 213}}
        sources.append({'source_task_ref': source_ref,
            'outcome_ref': saved(f'source_{index}/outcome.json', outcome)})
    originals = {item['outcome_ref']['file']: cohort.read(item['outcome_ref']) for item in sources}
    observed = []
    restored = []

    def scene(registration, workspace):
        agent, lab = agent_scene(workspace)
        agent._open_question(deepcopy(registration['question']))
        restored.append(agent)
        analyzed = set()

        def execute(action, arguments, actual_lab):
            assert action == 'run' and arguments['operator'] == 'research_route'
            assert set(arguments) == {'operator', 'research_question', 'allocation'}
            question = arguments['research_question']
            owner = next(q for q in agent.questions
                if q['id'] == question['origin_method_question_ref']['question_id'])
            assert owner['candidate_method_refs'] == [
                {'id': 'P0', 'version': 1}, {'id': 'guard', 'version': 1}]
            returns = question['method_use_plan']['ordinary_return_refs']
            records = [_read(agent, ref) for ref in returns]
            assert all(record['invoked_method_ref'] == {'id': 'P0', 'version': 1} for record in records)
            assert [record['method_content_source'] for record in records] == sources[:len(records)]
            observed.append(len(records))
            if len(records) not in analyzed:
                agent.costs['compute_cost'] += len(records)
                analyzed.add(len(records))
            proposed = [] if len(records) < 2 else [{
                'producer': 'finite_paid_guard_mature_startup_relation_induction',
                'research_selection_plan': deepcopy(plan), 'source_refs': deepcopy(returns)}]
            return {'method_action_research': {
                'status': 'method_content_proposed_pending_validation' if proposed else 'unresolved',
                'proposals': {'candidates': proposed}}}

        monkeypatch.setattr(agent, '_execute', execute)
        return agent, lab, []

    monkeypatch.setattr(pairing, '_scene', scene)
    registration = saved('registration.json', {'question': {'id': 'parent',
        'question': 'Explain the paid observations', 'parent_id': 'task'}})
    selection = saved('selection.json', {'general_guard_supported': True})
    verified = methods[1]
    if different_dependency:
        old = cohort.read(methods[1])
        verified = saved('verified/method_artifact.json', old)
        dependency = tmp_path / 'verified/workspace' / old['selection_plan']['ordinary_method_path']
        dependency.parent.mkdir(parents=True)
        dependency.write_text('def select_actions(public, observations, memory):\n'
                              '    return {"stop": True, "reason": "different retained implementation"}\n')
    result = cohort.form_method_from_ordinary_returns({'registration_ref': registration,
        'existing_method_refs': methods, 'ordinary_return_sources': sources,
        'analysis_compute_cap': 100, 'verified_method_ref': verified,
        'condition_selection_ref': selection, 'selection_checks': []}, tmp_path / 'acceptance')
    assert observed == [1, 2, 2]
    assert result['same_finite_selection_body']
    assert result['same_dependency'] is not different_dependency
    assert result['status'] == ('automatic_method_content_requires_separate_execution_evidence'
        if different_dependency else 'automatic_method_content_matches_verified_body')
    assert result['duplicate_return_reused'] and result['duplicate_return']['new_compute_cost'] == 0
    assert result['new_compute_cost'] == 5  # Two analyses and the finite-body comparison.
    assert result['remaining_analysis_compute'] == 95
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    assert not result['new_method_installed'] and not result['source_costs_recharged']
    assert len(restored[0].theory_consolidation['method_feedback']['P0']['outcome_refs']) == 2
    for source in sources:
        assert cohort.read(source['outcome_ref']) == originals[source['outcome_ref']['file']]


def test_selection_comparison_executes_each_retained_dependency_not_shared_memory_path(tmp_path):
    paths = []
    for name in ('old', 'current'):
        path = tmp_path / f'{name}.py'
        path.write_text('def select_actions(public, observations, memory):\n'
                        f'    return {{"action": "run", "arguments": {{"operator": "{name}"}}}}\n'
                        'def select_research_segment(public, observations, memory, *, plan, '
                        'ordinary_select_actions, condition=None):\n'
                        '    return ordinary_select_actions(public, observations, memory)\n')
        paths.append(path)
    shared = {'ordinary_method_path': str(paths[1])}
    plan = {'ordinary_method_path': 'ordinary.py'}
    actual = [cohort._static_method_action(plan, path, {}, [], shared) for path in paths]
    assert [choice['arguments']['operator'] for choice in actual] == ['old', 'current']
    assert shared['ordinary_method_path'] == str(paths[1])


def test_successor_keeps_retained_guard_support_separate_from_new_method_analysis(tmp_path):
    old_support = cohort.save(tmp_path / 'old_support.json', {
        'general_guard_supported': True, 'coverage': ['old_validated_profile']})
    latest_analysis = cohort.save(tmp_path / 'new_analysis.json', {
        'local_program_supported': True, 'general_guard_supported': True,
        'coverage': ['new_whole_method_profile']})
    body = {'ordinary_method_path': 'shared/guard_ordinary.py', 'condition': {'op': 'feature'},
        'when_true': {'first_segment': {'current_option': {'id': 'whole'}}},
        'when_false': {'first_segment_bindings': {'parent_rule': 'paid_startup_witnesses'}}}
    candidate = {'id': 'guard', 'version': 1, 'selection_plan': body,
        'selection_ref': old_support, 'source_evidence_refs': []}
    route = cohort.save(tmp_path / 'route.json', {'method_action_research': {
        'installed_candidate': candidate, 'guard_reconstruction': {'changed': False}}})
    bundle = cohort.save(tmp_path / 'pair.json', {'cells': {'00': {'id': 'P0'}, '10': {'id': 'old'}}})
    result = cohort.freeze_successor_policy(bundle, {
        'selection_ref': latest_analysis, 'research_route_ref': route}, tmp_path / 'frozen')
    cell = cohort.read(result['method_pair_ref'])['cells']['10']
    assert cell['selection_plan']['when_false'] == body['when_false']
    assert cell['selection_ref'] == result['selection_ref'] == old_support
    assert result['analysis_selection_ref'] == latest_analysis
    assert cohort.read(cell['selection_ref'])['coverage'] == ['old_validated_profile']


def test_successor_resolves_retained_guard_support_in_its_original_workspace(tmp_path):
    old_support = cohort.save(tmp_path / 'source/workspace/method_feedback/support.json', {
        'general_guard_supported': True, 'coverage': ['original_guard_profile']})
    latest_analysis = cohort.save(tmp_path / 'new_analysis.json', {
        'local_program_supported': True, 'general_guard_supported': True})
    candidate = {'id': 'guard', 'version': 1,
        'selection_plan': {'ordinary_method_path': 'shared/guard_ordinary.py',
            'condition': {'op': 'feature'}, 'when_true': {}, 'when_false': {}},
        'selection_ref': {'file': 'method_feedback/support.json', 'json_path': []},
        'source_evidence_refs': []}
    route = cohort.save(tmp_path / 'source/route.json', {'method_action_research': {
        'installed_candidate': candidate, 'policy_selection_ref': old_support}})
    bundle = cohort.save(tmp_path / 'pair.json', {'cells': {'00': {'id': 'P0'}, '10': {'id': 'old'}}})
    result = cohort.freeze_successor_policy(bundle, {
        'selection_ref': latest_analysis, 'research_route_ref': route}, tmp_path / 'successor')
    cell = cohort.read(result['method_pair_ref'])['cells']['10']
    assert cell['selection_ref'] == result['selection_ref'] == old_support
    assert cohort.read(cell['selection_ref'])['coverage'] == ['original_guard_profile']
    assert cohort.read(route)['method_action_research']['installed_candidate'] == candidate

