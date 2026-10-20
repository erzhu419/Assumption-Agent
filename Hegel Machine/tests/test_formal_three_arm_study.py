"""Registered ABC mechanisms on local fixtures, never a scientific trial."""

from copy import deepcopy
from dataclasses import asdict
import json
from shutil import copytree

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.theory_consolidation import _save as save_theory_library
from scripts.run_anonymous_revision_inquiry import (
    ORGANIZATION_ARMS, ORGANIZATION_ORIGIN, RegisteredOrganizationAgent, protocol, run_study,
)
from test_inquiry_revision import PublicLab, action
from test_question_routing import X, SQUARE, future_question, probe, theory
from test_theory_consolidation import affine, sources


class NoAcquisitionLab(PublicLab):
    costs = {'observation_cost': 0}

    def execute(self, action):
        raise AssertionError('Organization must not purchase even fixture rows')


def registration(*, phase='source', pr3_library_path=None, allowance=300):
    record = {'phase': phase, 'theories': [*sources(), theory('linear', X), theory('square', SQUARE)],
        'consolidation': {'theme': 'registered_fixture_sources',
            'source_theory_ids': ['source_a', 'source_b'],
            'mission': {'purpose': 'theory_learning', 'mission_id': 'registered_source_budget',
                        'theory_learning_allocation': {'compute': allowance}},
            'allocation': {'compute': allowance, 'max_groups': 4}}}
    if pr3_library_path is not None:
        record['pr3_library_path'] = str(pr3_library_path)
    return record


def study(tmp_path, *, registered=None, author_contract=True):
    agents, initial = {}, {}
    limits = InquiryLLMConfig(max_model_calls=0, token_budget=0, observation_budget=40,
                             confirmation_rows=2, compute_budget=3000, execution_seconds=30.)
    plan = protocol(kinds=('local_fixture',), arms=ORGANIZATION_ARMS, config=limits,
                    research_organization=registered or registration())
    knowledge = {'files': {'shared/old_predictor.py': '# Complete old archive remains available.\n'},
                 'lessons': ['Old failed attempt remains unresolved.'], 'manifest': {}}

    def factory(*args, **kwargs):
        agent = RegisteredOrganizationAgent(*args, **kwargs)
        agents[agent.organization_group] = agent
        initial[agent.organization_group] = {'arm': agent.arm, 'knowledge': deepcopy(agent.knowledge),
            'questions': deepcopy(agent.questions), 'index': deepcopy(agent.theory_index),
            'config': asdict(agent.config), 'interface': agent.common_interface,
            'costs': deepcopy(agent.costs)}
        return agent

    def prepare(agent, laboratory, case):
        if author_contract:
            question = future_question(agent)
            question['resolution_experiments'] = [probe(agent)]
            if callable(author_contract):
                author_contract(question)
            agent._execute('write', {'path': 'shared/note.txt', 'code': 'Authored fixture contract',
                                    'research_question': question}, laboratory)
        return {}

    result = run_study(plan, tmp_path, provider_factory=lambda _: None, knowledge=knowledge,
        prepare_prefix=prepare, laboratory_factory=lambda _: NoAcquisitionLab(), agent_factory=factory)
    return agents, initial, result, plan, knowledge


def automatic(agent, operator):
    return [event for event in agent.events if event.get('origin') == ORGANIZATION_ORIGIN
            and event['arguments']['operator'] == operator]


def test_registered_groups_share_the_strong_initial_state_and_execute_the_actual_mechanisms(tmp_path):
    agents, initial, result, plan, _ = study(tmp_path)
    assert initial['archive'] == initial['fixed_pr2'] == initial['pr3']
    assert json.loads((tmp_path / 'initial_archive.json').read_text()) == initial['archive']['knowledge']
    assert initial['archive']['arm'] == 'code_react'
    assert [q['id'] for q in initial['archive']['questions']] == ['task']
    assert 'compare_experiments' in initial['archive']['interface']
    assert RegisteredOrganizationAgent._execute is CommonArchiveInquiryAgent._execute
    assert not automatic(agents['archive'], 'research_route')
    assert not automatic(agents['archive'], 'theory_consolidation')
    for group in ('fixed_pr2', 'pr3'):
        event, = automatic(agents[group], 'research_route')
        assert event['arguments']['research_question'] == {'id': 'future_response'}
        assert event['arguments']['mission']['purpose'] == 'task'
        assert event['feedback']['selection'] is not None
        assert event['feedback']['matching']['status']
    assert not automatic(agents['fixed_pr2'], 'theory_consolidation')
    event, = automatic(agents['pr3'], 'theory_consolidation')
    assert event['feedback']['frozen_candidates']
    actual = event['feedback']['new_compute_cost']
    assert 0 < actual <= plan['research_organization']['consolidation']['allocation']['compute']
    assert agents['pr3']._organization_state['theory_mission']['resource_usage']['compute'] == actual
    for group, agent in agents.items():
        assert agent.costs['compute_cost'] == sum(e['feedback']['new_compute_cost'] for e in agent.events)
        assert agent.costs['compute_cost'] <= agent.config.compute_budget
        assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
        assert agent.observations == [] and not agent.confirmations
        assert agent.knowledge['files']['shared/old_predictor.py']
        assert agent.knowledge['lessons'] == ['Old failed attempt remains unresolved.']
    assert result['evidence_kind'] == 'scripted_interface_check'
    assert {row['arm'] for row in result['trials']} == set(ORGANIZATION_ARMS)
    assert all(row['research_organization']['inner_arm'] == 'code_react' for row in result['trials'])


def test_archive_still_runs_the_original_comparison_and_task_value_policy(tmp_path):
    agents, _, _, _, _ = study(tmp_path)
    agent, laboratory = agents['archive'], NoAcquisitionLab()
    agent.rows = [Observation({'u': 0.}, 0., ())]
    for name, source in [('linear.py', "def predict(row): return row['features']['u']\n"),
                         ('square.py', "def predict(row): return row['features']['u'] ** 2\n")]:
        agent._execute('write', {'path': name, 'code': source}, laboratory)
    result = agent._execute('run', {'operator': 'compare_experiments',
        'models': ['linear.py', 'square.py'], 'experiments': [{'id': 'public', 'actions': [action(2.)]}],
        'reference_actions': [action(3.)]}, laboratory)
    comparison = result['comparison']
    assert comparison['status'] == 'compared'
    assert comparison['selection_policy'] == 'expected_reference_risk_reduction'
    assert comparison['selected_experiment']['id'] == 'public'
    assert comparison['task_value']
    assert not automatic(agent, 'research_route') and agent.observations == []


def test_fixed_automatic_library_does_not_consume_later_working_objects_or_drop_ordinary_candidates(tmp_path):
    agents, _, _, _, _ = study(tmp_path)
    fixed, learned = agents['fixed_pr2'], agents['pr3']
    fixed.theory_consolidation = deepcopy(learned.theory_consolidation)
    save_theory_library(fixed)
    register_theories(fixed, [theory('ordinary_authored_candidate', affine(9., 2.))])
    fixed._before_model_turn(NoAcquisitionLab())
    latest = automatic(fixed, 'research_route')[-1]
    assert latest['feedback']['frozen_recall']['status'] == 'no_frozen_candidates'
    assert fixed.theory_consolidation['candidates']
    assert 'ordinary_authored_candidate' in {item['id'] for item in resolved_theories(fixed)}
    assert json.loads((tmp_path / 'initial_theory_library.json').read_text()) == {'candidates': {}}


def test_source_theory_allowance_survives_task_switch_and_recovery(tmp_path):
    agents, _, _, _, _ = study(tmp_path / 'source', registered=registration(allowance=80))
    agent = agents['pr3']
    before = agent._organization_state['theory_mission']['resource_usage']['compute']
    register_theories(agent, [sources()[0] | {'program': affine(3., 5.).to_dict()}])
    agent._before_model_turn(NoAcquisitionLab())
    assert before <= agent._organization_state['theory_mission']['resource_usage']['compute'] <= 80
    assert agent.research_mission['purpose'] == 'task'
    agent._request(NoAcquisitionLab())
    previous = agent.result('researching')
    previous['research_organization'].pop('source_theory_scopes')
    restored = RegisteredOrganizationAgent(None, tmp_path / 'restored', arm='pr3',
        research_organization=agent._organization_registration, knowledge=deepcopy(agent.knowledge),
        source_task=False, config=agent.config)
    restored.restore_research(previous, agent.workspace, NoAcquisitionLab(), confirmation_seed=17)
    assert restored._organization_state['theory_mission'] == previous['research_organization']['theory_mission']
    assert set(restored._organization_state['source_theory_scopes']) == {'source_a', 'source_b'}
    restored._before_model_turn(NoAcquisitionLab())
    assert restored._organization_state['theory_mission']['resource_usage']['compute'] <= 80
    assert restored.costs['compute_cost'] <= restored.config.compute_budget


def test_target_only_uses_the_frozen_library_and_missing_author_contract_remains_unresolved(tmp_path):
    source_agents, _, _, _, _ = study(tmp_path / 'source')
    frozen = source_agents['pr3'].workspace / 'shared/theory_library.json'
    agents, _, _, _, _ = study(tmp_path / 'target', registered=registration(
        phase='target', pr3_library_path=frozen), author_contract=False)
    for group, agent in agents.items():
        assert not automatic(agent, 'theory_consolidation')
        assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
        assert not any('pending_query' in q for q in agent.questions)
        if group != 'archive':
            event, = automatic(agent, 'research_route')
            assert event['feedback']['status'] == 'author_work_request'
            assert event['feedback']['selection'] is None
            assert agent._request(NoAcquisitionLab())['online_research']['questions'][0]['pending_author_work']
    assert agents['pr3'].theory_consolidation['candidates']


def test_resume_rejects_changed_registered_organization_before_using_saved_results(tmp_path):
    _, _, _, plan, knowledge = study(tmp_path / 'original')
    changed = deepcopy(plan)
    changed['research_organization']['phase'] = 'target'
    with pytest.raises(ValueError, match='preserve registered research_organization'):
        run_study(changed, tmp_path / 'changed', provider_factory=lambda _: pytest.fail('no provider'),
                  knowledge=knowledge, previous_study=tmp_path / 'original')


def test_actual_automatic_input_failure_keeps_each_registered_source_and_its_costs(tmp_path):
    def incomplete_panel(question):
        question['resolution_experiments'][0].pop('tolerance')

    agents, _, summary, _, _ = study(tmp_path, author_contract=incomplete_panel)
    assert len(summary['trials']) == 3
    for group in ('fixed_pr2', 'pr3'):
        event, = automatic(agents[group], 'research_route')
        assert event['feedback']['status'] == 'automatic_research_failed'
        assert event['feedback']['error'] == 'ValueError: research plans need their PR1 query and declared tolerance'
        assert agents[group]._request(NoAcquisitionLab())['research_organization']['routing']['event_index'] == agents[group].events.index(event)
        assert agents[group].costs['compute_cost'] == sum(
            entry['feedback']['new_compute_cost'] for entry in agents[group].events)
        assert agents[group].observations == [] and not agents[group].confirmations


def test_source_consolidation_retains_its_owner_while_an_author_child_is_active(tmp_path):
    from dataclasses import replace

    agents, _, _, _, _ = study(tmp_path)
    agent, laboratory = agents['pr3'], NoAcquisitionLab()
    agent.provider = object()  # Starting a session calls no provider method.
    agent.config = replace(agent.config, max_model_calls=6)
    arguments = {'operator': 'inquire', 'max_model_calls': 2,
        'question': {'id': 'author_child', 'question': 'The author supplied this unresolved task.',
                     'parent_id': 'task', 'depends_on': [],
                     'mission': {'purpose': 'task', 'mission_id': 'author_task'}}}
    feedback = agent._execute('run', arguments, laboratory)
    agent.events.append({'action': 'run', 'arguments': arguments, 'feedback': feedback})
    assert agent._inquiry_stack[-1]['question_id'] == 'author_child'
    agent._before_model_turn(laboratory)
    event = automatic(agent, 'theory_consolidation')[-1]
    owner_id = agent._organization_state['theory_question_id']
    assert event['arguments']['research_question'] == {'id': owner_id}
    owner = next(q for q in agent.questions if q['id'] == owner_id)
    assert owner['mission']['purpose'] == 'theory_learning'
    assert owner['mission'] == agent._organization_state['theory_mission']
    assert owner['mission']['resource_usage']['compute'] <= 300
    assert event['feedback'].get('reason') != 'theory_allocation_missing'
    assert automatic(agent, 'research_route')[-1]['arguments']['research_question'] == {'id': 'author_child'}
    assert agent._active_question_id == 'author_child'
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0


def test_run_study_restores_an_active_registered_group_without_a_paid_reply(tmp_path, monkeypatch):
    original, previous, resumed = (tmp_path / name for name in ('original', 'previous', 'resumed'))
    _, _, original_summary, plan, knowledge = study(original)
    copytree(original, previous)
    case_id = plan['cases'][0]['id']
    old_group = original / case_id / 'fixed_pr2'
    checkpoint_bytes = (old_group / 'active.json').read_bytes()
    training_bytes = (old_group / 'training.json').read_bytes()
    saved = json.loads(checkpoint_bytes)
    (previous / case_id / 'fixed_pr2' / 'training.json').unlink()
    assert (previous / case_id / 'fixed_pr2' / 'active.json').is_file()
    assert not (previous / case_id / 'fixed_pr2' / 'research_context.json').exists()
    assert not list((previous / case_id / 'fixed_pr2').glob('code_action_*'))
    monkeypatch.setattr('scripts.run_anonymous_revision_inquiry.restore_pending_question_error',
                        lambda *args, **kwargs: pytest.fail('No historical paid reply belongs to this checkpoint'))
    provider_workspaces = []

    def no_provider(workspace):
        provider_workspaces.append(workspace)
        return None

    summary = run_study(plan, resumed, previous_study=previous, knowledge=knowledge,
        provider_factory=no_provider, laboratory_factory=lambda _: NoAcquisitionLab(),
        prepare_prefix=lambda *args: pytest.fail('The retained prefix must not be purchased again'))
    assert provider_workspaces == [resumed / case_id / 'fixed_pr2']
    assert summary['source_count'] == original_summary['source_count'] == 1
    rows = {row['arm']: row for row in summary['trials']}
    original_rows = {row['arm']: row for row in original_summary['trials']}
    assert set(rows) == set(ORGANIZATION_ARMS)
    for group in ('archive', 'pr3'):
        assert rows[group] == original_rows[group]
    restored = json.loads((resumed / case_id / 'fixed_pr2' / 'training.json').read_text())
    assert restored['prior_research_cost'] == saved['costs']
    assert restored['costs'] == saved['costs']
    assert all(value == 0 for value in restored['continuation_cost'].values())
    assert 'recovered_paid_reply' not in restored
    assert restored['resume_origin'] == str(previous / case_id / 'fixed_pr2' / 'active.json')
    assert restored['research_organization'] == saved['research_organization']
    assert restored['theory_index'] == saved['theory_index']
    assert restored['research_mission'] == saved['research_mission']
    assert restored['events'] == saved['events']
    assert restored['costs']['model_calls'] == restored['costs']['observation_cost'] == 0
    assert restored['research_organization']['automatic_library_path'] == str(original / 'initial_theory_library.json')
    assert rows['pr3']['research_organization']['automatic_library_path'] == 'shared/theory_library.json'
    assert rows['pr3']['research_organization']['theory_mission']['resource_usage']['compute'] <= 300
    prepared = json.loads((resumed / 'initial_archive.json').read_text())
    original_input = json.loads((original / 'initial_archive.json').read_text())
    assert prepared == original_input
    assert json.loads((resumed / 'initial_theory_library.json').read_text()) == {
        'source_ref': original_input['theory_consolidation']}
    assert (old_group / 'active.json').read_bytes() == checkpoint_bytes
    assert (old_group / 'training.json').read_bytes() == training_bytes


def final_author_source(tmp_path, allowance=500):
    registered = registration(allowance=allowance)
    initial = registered['theories'][:2]
    for source in initial:
        source['episode_ref'] = {'question_id': source['id'], 'meaning_version': 1}
    authored = []
    for source in initial:
        authored.append({**deepcopy(source), 'id': 'new_' + source['id'],
                         'program': affine(7., 8.).to_dict()})
    authored.append({**deepcopy(authored[0]), 'id': 'another_meaning',
                     'episode_ref': {'question_id': 'source_a', 'meaning_version': 2}})

    class FinalActionProvider:
        calls = 0

        def next_action(self, request):
            self.calls += 1
            return {'action': 'run', 'arguments': {'operator': 'research_route',
                'research_question': {'id': 'task'}, 'theories': authored},
                # Actual supported termination after a valid last action with
                # unavailable usage: no next before-turn hook runs.
                'usage': {'backend': 'scripted_interface_test', 'total_tokens': None}}

    provider = FinalActionProvider()
    agent = RegisteredOrganizationAgent(provider, tmp_path, arm='pr3',
        research_organization={**registered, 'initial_library_path': str(tmp_path / 'initial.json')},
        config=InquiryLLMConfig(max_model_calls=1, token_budget=10, observation_budget=40,
                               confirmation_rows=2, compute_budget=3000), source_task=False)
    (tmp_path / 'initial.json').write_text('{"candidates":{}}')
    for source in initial:
        agent._open_question({'id': source['id'], 'question': 'Retained source fixture.',
            'parent_id': None, 'depends_on': [], 'context_ref': source['context_ref']})
    result = agent.run(NoAcquisitionLab())
    return agent, result, provider, registered


@pytest.mark.parametrize('allowance', [500, 1])
def test_final_author_action_consolidates_new_ids_only_in_registered_episodes(tmp_path, allowance):
    agent, result, provider, _ = final_author_source(tmp_path, allowance)
    event = automatic(agent, 'theory_consolidation')[-1]
    assert event['arguments']['source_theory_ids'] == [
        'source_a', 'source_b', 'new_source_a', 'new_source_b']
    assert result['status'] == 'token_usage_unavailable' and provider.calls == 1
    assert agent.costs['model_calls'] == 1 and agent.costs['observation_cost'] == 0
    actual = sum(item['feedback']['new_compute_cost'] for item in automatic(agent, 'theory_consolidation'))
    assert agent._organization_state['theory_mission']['resource_usage']['compute'] == actual <= allowance
    if allowance == 1:
        assert event['feedback']['status'] == 'UNKNOWN'
        assert not event['feedback']['frozen_candidates']
    else:
        frozen = [agent.theory_consolidation['candidates'][item['candidate_id']]
                  for item in event['feedback']['frozen_candidates']]
        instances = [entry for candidate in frozen for entry in candidate.get('instances', [])]
        assert {'new_source_a', 'new_source_b'} <= {entry['episode_id'].rsplit(':', 1)[1] for entry in instances}
        assert all(candidate['independent_source_count'] == 2 for candidate in frozen)


def test_final_authored_source_exports_to_formal_target_consumption(tmp_path):
    source, source_result, _, registered = final_author_source(tmp_path / 'source')
    latest = automatic(source, 'theory_consolidation')[-1]['feedback']
    frozen_ids = {item['candidate_id'] for item in latest['frozen_candidates']}
    knowledge = source_result['outgoing_knowledge']
    frozen_ref = knowledge['theory_consolidation']
    frozen_bytes = (source.workspace / latest['library_path']).read_bytes()
    target_registration = {**registered, 'phase': 'target',
        'initial_library_path': source._organization_state['initial_library_path'],
        'pr3_library_path': frozen_ref}
    agents, initial = {}, {}

    def factory(*args, **kwargs):
        agent = RegisteredOrganizationAgent(*args, **kwargs)
        agents[agent.organization_group] = agent
        initial[agent.organization_group] = {'knowledge': deepcopy(agent.knowledge),
            'index': deepcopy(agent.theory_index), 'interface': agent.common_interface,
            'limits': asdict(agent.config), 'arm': agent.arm}
        return agent

    def prepare(agent, laboratory, case):
        question = future_question(agent)
        question['resolution_experiments'] = [probe(agent)]
        agent._execute('write', {'path': 'shared/note.txt', 'code': 'Same authored target contract',
                                'research_question': question}, laboratory)
        return {}

    plan = protocol(kinds=('local_fixture',), arms=ORGANIZATION_ARMS,
        config=InquiryLLMConfig(max_model_calls=0, token_budget=0, observation_budget=40,
                               confirmation_rows=2, compute_budget=3000, execution_seconds=30.),
        research_organization=target_registration)
    summary = run_study(plan, tmp_path / 'target', knowledge=knowledge,
        provider_factory=lambda _: None, laboratory_factory=lambda _: NoAcquisitionLab(),
        prepare_prefix=prepare, agent_factory=factory)
    assert initial['archive'] == initial['fixed_pr2'] == initial['pr3']
    assert initial['archive']['arm'] == 'code_react'
    assert {'source_a', 'source_b', 'new_source_a', 'new_source_b', 'linear', 'square'} <= set(initial['archive']['index'])
    assert not agents['archive'].theory_consolidation['candidates']
    assert not agents['fixed_pr2'].theory_consolidation['candidates']
    assert frozen_ids <= set(agents['pr3'].theory_consolidation['candidates'])
    assert not automatic(agents['archive'], 'research_route')
    fixed, = automatic(agents['fixed_pr2'], 'research_route')
    applied, = automatic(agents['pr3'], 'research_route')
    assert fixed['feedback']['frozen_recall']['status'] == 'no_frozen_candidates'
    recalled = applied['feedback']['frozen_recall']['generated_theory_ids']
    assert recalled and all(any(identifier in generated for generated in recalled) for identifier in frozen_ids)
    resolved = {item['id']: item for item in resolved_theories(agents['pr3'])}
    assert all(resolved[identifier]['support']['status'] == 'candidate' for identifier in recalled)
    predicted = applied['feedback']['selection']['groups']
    assert set(recalled) <= {identifier for group in predicted for identifier in group['theory_ids']}
    assert sorted(group['task_answer'] for group in predicted) == [3., 7., 9.]
    # Both source abstractions currently yield the same target Program; the
    # final authored object's recall does not add another prediction vote.
    shared = next(group for group in predicted if group['task_answer'] == 7.)
    assert set(shared['theory_ids']) == set(recalled)
    assert sorted(group['task_answer'] for group in fixed['feedback']['selection']['groups']) == [3., 9.]
    for group in ('fixed_pr2', 'pr3'):
        agent = agents[group]
        event, = automatic(agent, 'research_route')
        selection = event['feedback']['selection']
        assert selection['groups'] and all(item['task_answer'] is not None for item in selection['groups'])
        assert selection['support_updated'] is False
        parent = next(item for item in agent.questions if item['id'] == 'task')
        chosen = selection['selected_experiment']
        assert chosen['id'] == 'distinguishing_probe'
        assert parent['pending_query']['actions'] == chosen['actions']
        assert event['feedback']['parent_use']['status'] == 'parent_query_prepared'
    for agent in agents.values():
        assert not automatic(agent, 'theory_consolidation')
        assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
        assert agent.costs['compute_cost'] <= plan['limits']['compute_budget']
        assert agent.costs['compute_cost'] == sum(event['feedback']['new_compute_cost']
            for event in agent.events if event.get('origin') == ORGANIZATION_ORIGIN)
        assert agent.observations == [] and not agent.confirmations
    assert summary['evidence_kind'] == 'scripted_interface_check' and len(summary['trials']) == 3
    assert (source.workspace / latest['library_path']).read_bytes() == frozen_bytes


def test_registered_episode_cutoff_requires_the_actual_program_address(tmp_path):
    registered = registration()
    source = registered['theories'][0]
    source['episode_ref'] = {'question_id': 'source_a', 'meaning_version': 1}
    registered['consolidation']['source_theory_ids'] = ['source_a']
    document = tmp_path / 'retained.json'
    actions = [{'kind': 'revise', 'details': {'before_program': affine().to_dict(),
        'after_program': affine(slope).to_dict()}} for slope in (3., 4.)]
    document.write_text(json.dumps({'episodes': [{'task_id': 'source_a', 'program': affine().to_dict(),
        'actions': actions, 'questions': [], 'status': 'inconclusive'}]}))
    episode_ref = {'file': str(document), 'json_path': ['episodes', 0], 'action_cutoff': 0}
    registered['consolidation']['episode_sources'] = [episode_ref]
    agent = RegisteredOrganizationAgent(None, tmp_path / 'workspace', arm='pr3',
        research_organization={**registered, 'initial_library_path': str(tmp_path / 'initial.json')},
        source_task=False, config=InquiryLLMConfig(max_model_calls=0, compute_budget=3000))
    authored = []
    for name, index in [('missing_address', None), ('within_cutoff', 0), ('after_cutoff', 1)]:
        declaration = {**deepcopy(source), 'id': name,
            'episode_ref': {**source['episode_ref'], 'source_ref': episode_ref}}
        if index is not None:
            declaration['program'] = actions[index]['details']['after_program']
            declaration['program_source_ref'] = {**episode_ref,
                'json_path': ['episodes', 0, 'actions', index, 'details', 'after_program']}
        authored.append(declaration)
    authored.append({**deepcopy(authored[1]), 'id': 'derived_transfer',
                     'consolidation_ref': {'candidate_id': 'retained_object'}})
    authored.append({**deepcopy(authored[1]), 'id': 'foreign_file',
        'program_source_ref': {'file': str(tmp_path / 'other.json'), 'json_path': ['episodes', 0, 'program']}})
    register_theories(agent, authored)
    agent._before_model_turn(NoAcquisitionLab())
    event, = automatic(agent, 'theory_consolidation')
    assert event['arguments']['source_theory_ids'] == ['source_a', 'within_cutoff']
    assert event['feedback']['source_count'] == 1
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert agent._organization_state['theory_mission']['resource_usage']['compute'] == event['feedback']['new_compute_cost'] <= 300
    # Real retained sequences repeat task IDs: the explicit original index,
    # not the short task name, keeps the same source qualified after reload.
    payload = json.loads(document.read_text())
    payload['episodes'].append(deepcopy(payload['episodes'][0]))
    document.write_text(json.dumps(payload))
    qualified = deepcopy(registered)
    qualified['theories'][0]['episode_ref']['source_ref'] = episode_ref
    other_source = {**deepcopy(authored[1]), 'id': 'same_task_other_episode',
        'episode_ref': {**source['episode_ref'], 'source_ref': {**episode_ref, 'json_path': ['episodes', 1]}},
        'program_source_ref': {**episode_ref,
            'json_path': ['episodes', 1, 'actions', 0, 'details', 'after_program']}}
    second = RegisteredOrganizationAgent(None, tmp_path / 'duplicate', arm='pr3',
        research_organization={**qualified, 'initial_library_path': str(tmp_path / 'initial.json')},
        source_task=False, config=agent.config)
    register_theories(second, [*authored, other_source])
    assert second._registered_source_ids() == ['source_a', 'within_cutoff']
    assert second._organization_state['source_theory_scopes']['source_a']['episode_id'].endswith('@source:episodes[0]')
