"""Finite formal dispatch on supplied toy programs, never discovery evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.structural_inquiry import _register_native_theories
from scripts import run_anonymous_revision_inquiry as study
from test_common_archive_online_research import ConfirmationLab, NEW_MODEL, OLD_MODEL
from test_inquiry_revision import action


DRIVER = 'shared/finite_driver.py'
SOURCE = '''import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    own = [r for r in state['method_results'] if r['path'] == 'shared/finite_driver.py']
    if not own:
        return {'action':'evaluate','arguments':{'model':'old.py'}}
    if len(own) == 1:
        return {'actions':[{'kind':'trajectory','sequence':[{'u':2.}]}],
            'revision':{'id':'fixture_contrast','affected_object':'representation',
                        'models':['old.py','new.py']}}
    return {'model':'new.py'}
'''


class LocalLab(ConfirmationLab):
    @property
    def costs(self):
        return {'observation_cost': self.paid}

    def evaluation_rows(self, length, seed):
        record = self.execute(action(*[.25 + i for i in range(length)]))
        return self.observation_rows(record)


@pytest.fixture
def runtime_root(tmp_path, monkeypatch):
    root = tmp_path / 'current_runtime'
    for name in ('src/hegel_machine/current_module.py', 'scripts/run_anonymous_revision_inquiry.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# current runtime fixture\n')
    monkeypatch.setattr(study, 'ROOT', root)
    return root


def registration():
    return {'phase': 'source', 'theories': [],
        'exploration_policy': {'kind': 'seeded_legal', 'seed': 11},
        'consolidation': {'source_theory_ids': [], 'source_question_refs': [],
            'mission': {'purpose': 'theory_learning', 'mission_id': 'finite_fixture',
                        'theory_learning_allocation': {'compute': 1000}},
            'allocation': {'compute': 1000, 'max_groups': 4}}}


def plan(*, steps=3, confirm=True, registered=None):
    result = study.protocol(kinds=('supplied_fixture',), arms=study.ORGANIZATION_ARMS,
        research_organization=registered or registration(),
        author={'kind': 'finite_native', 'path': DRIVER, 'max_steps': steps, 'confirm': confirm},
        config=InquiryLLMConfig(observation_budget=20, confirmation_rows=2,
            compute_budget=10000, execution_seconds=30., max_model_calls=0, token_budget=0))
    result['evaluation']['rows'] = 3
    return result


def knowledge(source=SOURCE):
    return {'files': {DRIVER: source, 'old.py': OLD_MODEL, 'new.py': NEW_MODEL},
        'lessons': ['Retain the old failed draft and its unresolved scope.'],
        'manifest': {'hypotheses': [{'id': 'old', 'path': 'old.py'}],
                     'research_programs': [], 'support_questions': []}}


def paid_prefix(agent, lab, case):
    agent._execute('query', {'actions': [action(1.)]}, lab)
    return {}


def test_finite_formal_dispatch_uses_common_archive_hooks_and_original_confirmation(tmp_path, runtime_root):
    agents = {}

    def factory(*args, **kwargs):
        agent = study.RegisteredOrganizationAgent(*args, **kwargs)
        agents[agent.organization_group] = agent
        assert agent.provider is None and agent.arm == 'code_react'
        return agent

    output = tmp_path / 'formal'
    registered = plan()
    result = study.run_study(registered, output,
        provider_factory=lambda _: pytest.fail('finite author must not construct a provider'),
        knowledge=knowledge(), prepare_prefix=paid_prefix,
        laboratory_factory=lambda _: LocalLab(), agent_factory=factory)
    assert result['evidence_kind'] == 'finite_native_program_execution'
    assert len(result['trials']) == 3
    for record in result['trials']:
        agent = agents[record['arm']]
        assert record['status'] == 'completed' and record['terminal']['passed']
        assert record['fresh_confirmation_passed']
        assert record['costs']['model_calls'] == record['costs']['reported_tokens'] == 0
        assert record['costs']['observation_cost'] == 4
        assert record['costs']['compute_cost'] <= registered['limits']['compute_budget']
        assert record['research_organization']['finite_author'] == {
            'contract': registered['author'], 'completed_steps': 3}
        assert agent.knowledge['files']['old.py'] == OLD_MODEL
        assert agent.knowledge['lessons'] == knowledge()['lessons']
        revision, = agent.revisions
        assert [f['prospective_mse'] for f in revision['forecasts']] == [4., 0.]
        own = [e for e in agent.events if e.get('usage', {}).get('backend') == 'frozen_program_execution']
        assert len(own) == 3
        automatic = [e for e in agent.events if e.get('origin') == study.ORGANIZATION_ORIGIN]
        assert bool(automatic) == (record['arm'] != 'archive')
        for event in automatic:
            if event['arguments']['operator'] == 'research_route':
                assert event['arguments']['exploration_policy'] == registered['research_organization']['exploration_policy']
    frozen = json.loads((output / 'protocol.json').read_text())
    assert frozen['runtime_snapshot'] == ['scripts/run_anonymous_revision_inquiry.py',
                                           'src/hegel_machine/current_module.py']
    for name in frozen['runtime_snapshot']:
        assert (output / 'source_at_run' / name).read_text() == (runtime_root / name).read_text()
    source_ref = frozen['author_source_ref']
    assert source_ref == {'file': 'initial_archive.json', 'json_path': ['files', DRIVER]}
    assert json.loads((output / source_ref['file']).read_text())['files'][DRIVER] == SOURCE
    assert not list(output.glob('case_*/*/source_at_run'))


def test_finite_confirmation_requires_the_explicit_original_permission(tmp_path, runtime_root):
    result = study.run_study(plan(confirm=False), tmp_path / 'no_confirm', knowledge=knowledge(),
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
    for record in result['trials']:
        assert record['status'] == 'confirmation_not_authorized'
        assert record['model'] is None and not record['fresh_confirmation_passed']
        assert record['costs']['observation_cost'] == 2
        assert record['costs']['model_calls'] == 0


@pytest.mark.parametrize('incoming', ['omitted', 'changed'])
def test_recovery_runs_only_the_remaining_finite_steps_and_keeps_paid_evidence(tmp_path, runtime_root, monkeypatch, incoming):
    driver = '''import json
def select_actions(spec, observations, memory):
    own = [r for r in json.load(open('research_state.json'))['method_results']
           if r['path'] == 'shared/finite_driver.py']
    if not own:
        return {'actions':[{'kind':'trajectory','sequence':[{'u':2.}]}]}
    return {'stop':'Finite supplied queue is complete.'}
'''
    registered = plan(steps=2, confirm=False)
    actual = study.execute_procedure

    def interrupted(*args, **kwargs):
        kwargs['max_steps'] = 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(study, 'execute_procedure', interrupted)
    previous = tmp_path / 'previous'
    study.run_study(registered, previous, knowledge=knowledge(driver),
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
    case = registered['cases'][0]['id']
    workspace = previous / case / 'archive'
    before = json.loads((workspace / 'active.json').read_text())
    assert before['research_organization']['finite_author']['completed_steps'] == 1
    assert before['costs']['observation_cost'] == 2
    (workspace / 'training.json').unlink()
    monkeypatch.setattr(study, 'execute_procedure', actual)
    result = study.run_study(registered, tmp_path / 'resumed', previous_study=previous,
        knowledge=(None if incoming == 'omitted' else knowledge(
            "def select_actions(*args): raise AssertionError('Changed incoming driver must not run')\n")),
        laboratory_factory=lambda _: LocalLab(),
        prepare_prefix=lambda *args: pytest.fail('Recovery must not repurchase the prefix'),
        provider_factory=lambda _: pytest.fail('Recovery must not construct a provider'))
    resumed = next(row for row in result['trials'] if row['arm'] == 'archive')
    assert resumed['status'] == 'method_stopped'
    assert resumed['research_organization']['finite_author']['completed_steps'] == 2
    assert resumed['costs']['observation_cost'] == 2 and resumed['costs']['model_calls'] == 0
    saved = json.loads((tmp_path / 'resumed' / case / 'archive' / 'training.json').read_text())
    assert len(saved['paid_observations']) == len(before['paid_observations']) == 2
    assert len(saved['method_results']) == 2
    assert saved['continuation_cost']['observation_cost'] == 0
    assert (tmp_path / 'resumed' / case / 'archive' / DRIVER).read_text() == driver
    assert saved['knowledge_protocol'] == 'method_practice_workspace'
    assert 'files' not in saved['outgoing_knowledge']
    changed = deepcopy(registered)
    changed['author']['max_steps'] = 3
    with pytest.raises(ValueError, match='preserve registered author'):
        study.run_study(changed, tmp_path / 'changed', previous_study=previous, knowledge=knowledge(driver))


def test_missing_finite_driver_stops_before_any_prefix_or_provider(tmp_path, runtime_root):
    with pytest.raises(ValueError, match='common executable archive'):
        study.run_study(plan(), tmp_path / 'missing', knowledge={'files': {'old.py': OLD_MODEL}},
            prepare_prefix=lambda *args: pytest.fail('Invalid author must not acquire a prefix'),
            provider_factory=lambda *args: pytest.fail('No finite provider'))


def test_registered_question_scope_admits_actual_native_bank_without_a_fake_seed(tmp_path, runtime_root):
    registered = registration()
    registered['consolidation']['source_question_refs'] = [{'question_id': 'native_source', 'meaning_version': 1}]
    prepared, memory = study._prepare_organization(plan(registered=registered), tmp_path, knowledge())
    agent = study.RegisteredOrganizationAgent(None, tmp_path / 'source', arm='pr3',
        research_organization=prepared, knowledge=memory, source_task=False,
        config=InquiryLLMConfig(compute_budget=10000, max_model_calls=0))
    assert not agent.theory_index and agent._registered_source_ids() == []
    target = {'role_signature': {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}},
        'output_unit': '1', 'context_ref': {'source_id': 'fixture_source'}}
    registered_ids = []
    for identifier in ('native_source', 'other_source'):
        question = agent._open_question({'id': identifier, 'question': 'Supplied fixture source',
            'context_ref': {'source_id': 'fixture_source'}, 'theory_target': target})
        question['native_condition_research'] = [{'status': 'search_incomplete', 'compute_cost': 17,
            'candidates': [{'program': Program('feature', name='u').to_dict(),
                'status': 'refuted', 'fit_mse': 3., 'counterexample_refs': [
                    {'file': 'laboratory_observations.json', 'json_path': [0]}]}]}]
        receipt = _register_native_theories(agent, question, agent.questions.index(question), 0)
        registered_ids.append(receipt['registered'][0]['theory_id'])
    original, other = registered_ids
    assert agent._registered_source_ids() == [original]
    assert agent.theory_index[original]['support']['status'] == 'refuted'
    source = agent.theory_index[original]
    foreign = deepcopy(source) | {'id': 'foreign'}
    foreign['program_source_ref']['file'] = str(tmp_path / 'foreign/research_state.json')
    derived = deepcopy(source) | {'id': 'derived', 'consolidation_ref': {'candidate_id': 'not_a_source'}}
    register_theories(agent, [foreign, derived])
    assert agent._registered_source_ids() == [original]
    changed = agent._open_question({'id': 'native_source', 'context_ref': {'source_id': 'new_meaning'}})
    assert changed['meaning_version'] == 2
    changed['theory_target'] = target
    changed['native_condition_research'] = [{'status': 'candidate', 'candidates': [
        {'program': Program('const', value=7.).to_dict()}]}]
    later = _register_native_theories(agent, changed, agent.questions.index(changed), 0)
    assert later['registered'][0]['theory_id'] not in (original, other)
    assert agent._registered_source_ids() == [original]
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0


def test_finite_cli_bypasses_subscription_validation(tmp_path, runtime_root, monkeypatch, capsys):
    organization = tmp_path / 'organization.json'
    author = tmp_path / 'author.json'
    organization.write_text(json.dumps(registration()))
    author.write_text(json.dumps(plan()['author']))
    captured = {}

    def dispatch(registered, output, **kwargs):
        captured.update(plan=registered, **kwargs)
        return {'status': 'not_executed_test'}

    monkeypatch.setattr(study, 'run_study', dispatch)
    monkeypatch.setattr(study, 'verify_codex_subscription', lambda: pytest.fail('No subscription is used'))
    monkeypatch.setattr('sys.argv', ['runner', '--arms', *study.ORGANIZATION_ARMS,
        '--research-organization', str(organization), '--finite-native-author', str(author),
        '--output', str(tmp_path / 'cli')])
    study.main()
    assert captured['provider_factory'] is None
    assert captured['plan']['provider'] == {'backend': 'finite_native_procedure', 'model_calls': 0}
    assert captured['plan']['author'] == plan()['author']
    assert 'not_executed_test' in capsys.readouterr().out


def test_formal_constructor_shares_original_paid_trace_before_writing_arm_files(tmp_path, runtime_root, monkeypatch):
    trace = 'archive/source_000/paid_trace.json'
    repeated = 'archive/source_001/_evaluated_model/archive/source_000/paid_trace.json'
    content = '{"events":["retained paid evidence"]}'
    original = tmp_path / 'original'
    source = original / trace
    source.parent.mkdir(parents=True)
    source.write_text(content)
    memory = knowledge("def select_actions(spec, observations, memory): return {'stop':'Fixture done.'}\n")
    memory['files'].update({trace: content, repeated: content, 'parameter.json': '{"value": 3}'})
    incoming = original / 'archive.json'
    incoming.write_text(json.dumps(memory))
    registered = plan(steps=1, confirm=False)
    registered['archive'] = str(incoming)
    write = Path.write_text

    def no_repeated_trace_write(path, *args, **kwargs):
        assert path.name != 'paid_trace.json', 'The complete archive trace must be referenced before construction'
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', no_repeated_trace_write)
    output = tmp_path / 'shared'
    study.run_study(registered, output, knowledge=memory, prepare_prefix=lambda *args: {},
        laboratory_factory=lambda _: LocalLab())
    case = registered['cases'][0]['id']
    for group in study.ORGANIZATION_ARMS:
        workspace = output / case / group
        assert (workspace / trace).is_symlink()
        assert (workspace / trace).resolve() == (workspace / repeated).resolve() == source.resolve()
        assert not (workspace / 'old.py').is_symlink()
        assert not (workspace / 'parameter.json').is_symlink()
    (output / case / 'archive/parameter.json').write_text('{"value": 99}')
    assert (output / case / 'fixed_pr2/parameter.json').read_text() == '{"value": 3}'
    assert source.read_text() == content


def test_last_finite_native_action_registers_and_consolidates_only_its_declared_source(tmp_path, runtime_root):
    from test_native_condition_inquiry import COMPOSITION, PaidConditionalLab, buy_fixture
    from test_native_theory_registration import CONTEXT, TARGET

    registered = registration()
    registered['consolidation']['source_question_refs'] = [
        {'question_id': COMPOSITION['id'], 'meaning_version': 1}]
    question = {**COMPOSITION, 'context_ref': CONTEXT, 'theory_target': TARGET}
    arguments = {'operator': 'native_condition_research', 'research_question': question,
        'max_compute': 500_000, 'max_candidates': 2,
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}]}
    source = 'def select_actions(spec, observations, memory):\n    return ' + repr(
        {'action': 'run', 'arguments': arguments}) + '\n'
    configured = plan(steps=1, confirm=False, registered=registered)
    configured['limits'].update(observation_budget=300, confirmation_rows=16,
                                compute_budget=2_000_000, execution_seconds=60.)
    agents = {}

    def factory(*args, **kwargs):
        agent = study.RegisteredOrganizationAgent(*args, **kwargs)
        agents[agent.organization_group] = agent
        return agent

    def prepare(agent, lab, case):
        buy_fixture(agent, lab)
        return {}

    result = study.run_study(configured, tmp_path / 'native', knowledge=knowledge(source),
        prepare_prefix=prepare, laboratory_factory=lambda _: PaidConditionalLab(), agent_factory=factory)
    for record in result['trials']:
        agent = agents[record['arm']]
        assert record['status'] == 'method_step_limit'
        assert record['costs']['model_calls'] == 0 and record['costs']['observation_cost'] == 80
        assert not agent.confirmations and record['research_organization']['finite_author']['completed_steps'] == 1
        question = next(q for q in agent.questions if q['id'] == COMPOSITION['id'])
        assert question['meaning_version'] == 1
        assert question['native_condition_research'][0]['candidates']
        assert agent.theory_index and all(t['source_ref']['kind'] == 'native_candidate'
                                          for t in agent.theory_index.values())
        assert agent.knowledge['files']['old.py'] == OLD_MODEL
    source_agent = agents['pr3']
    event = [e for e in source_agent.events if e.get('origin') == study.ORGANIZATION_ORIGIN
             and e['arguments']['operator'] == 'theory_consolidation'][-1]
    assert set(event['arguments']['source_theory_ids']) == set(source_agent.theory_index)
    assert source_agent.events.index(event) > next(i for i, e in enumerate(source_agent.events)
        if e.get('usage', {}).get('backend') == 'frozen_program_execution')
    assert event['arguments']['source_question_refs'] == registered['consolidation']['source_question_refs']
    # An executable source family is retained, while one real source Episode
    # does not establish an active library or independent transfer support.
    frozen, = event['feedback']['frozen_candidates']
    assert frozen['source_count'] == 1 and frozen['source_reconstruction'] == 'exact'
    assert frozen['active_library'] is False and frozen['transfer_status'] == 'not_established'
    assert event['feedback']['support_updated'] is False
    assert not [e for e in agents['fixed_pr2'].events if e.get('origin') == study.ORGANIZATION_ORIGIN
                and e['arguments']['operator'] == 'theory_consolidation']


@pytest.mark.parametrize('saved_training', [False, True])
def test_single_compact_source_outgoing_remains_complete_executable_target_input(tmp_path, runtime_root, saved_training):
    source_output = tmp_path / 'source'
    source_output.mkdir()
    memory = knowledge("def select_actions(spec, observations, memory): return {'stop':'Supplied queue complete.'}\n")
    prepared, common = study._prepare_organization(plan(), source_output, memory)
    source = study.RegisteredOrganizationAgent(None, source_output / 'workspace', arm='pr3',
        research_organization=prepared, knowledge=common, source_task=False,
        config=InquiryLLMConfig(compute_budget=10000, max_model_calls=0, confirmation_rows=2))
    source._before_model_turn(LocalLab())
    source._prepare_program_state()
    original = source.result('method_stopped')
    outgoing = original['outgoing_knowledge']
    assert 'files' not in outgoing and outgoing['workspace'] == str(source.workspace)
    if saved_training:
        (source.workspace / 'training.json').write_text(json.dumps(original))
    else:
        assert not (source.workspace / 'training.json').exists()
    target_registration = registration() | {'phase': 'target',
        'initial_library_path': prepared['initial_library_path'],
        'pr3_library_path': outgoing['theory_consolidation']}
    agents = {}

    def factory(*args, **kwargs):
        agent = study.RegisteredOrganizationAgent(*args, **kwargs)
        agents[agent.organization_group] = agent
        return agent

    configured = plan(steps=1, confirm=False, registered=target_registration)
    result = study.run_study(configured, tmp_path / 'target', knowledge=outgoing,
        prepare_prefix=lambda *args: {}, laboratory_factory=lambda _: LocalLab(), agent_factory=factory)
    assert all(row['status'] == 'method_stopped' for row in result['trials'])
    assert all(row['costs']['model_calls'] == row['costs']['observation_cost'] == 0
               for row in result['trials'])
    for agent in agents.values():
        assert agent.knowledge['lessons'] == memory['lessons']
        for name, content in memory['files'].items():
            assert (agent.workspace / name).read_text() == content
        assert (agent.workspace / 'new.py').read_text() == NEW_MODEL  # Unlisted draft remains in the full archive.
        assert agent.method_results[0]['feedback']['status'] == 'method_stopped'
    assert (source.workspace / DRIVER).read_text() == memory['files'][DRIVER]


def test_direct_cold_constructor_resolves_workspace_and_shares_trace_before_writes(tmp_path, runtime_root, monkeypatch):
    source_output = tmp_path / 'source'
    source_output.mkdir()
    memory = knowledge("def select_actions(spec, observations, memory): return {'stop':'Supplied queue complete.'}\n")
    trace = 'archive/source_000/paid_trace.json'
    memory['files'][trace] = '{"events":["retained fixture failure"]}'
    memory['files']['parameter.json'] = '{"value": 3}'
    prepared, common = study._prepare_organization(plan(), source_output, memory)
    source = study.RegisteredOrganizationAgent(None, source_output / 'workspace', arm='pr3',
        research_organization=prepared, knowledge=common, source_task=False,
        config=InquiryLLMConfig(compute_budget=10000, max_model_calls=0, confirmation_rows=2))
    question = source._open_question({'id': 'native_source', 'question': 'Retained fixture candidate',
        'theory_target': {'role_signature': {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}},
            'output_unit': '1', 'context_ref': {'source_id': 'fixture_source'}}})
    question['native_condition_research'] = [{'status': 'search_incomplete', 'compute_cost': 17,
        'round_index': 0, 'fit_evidence': [], 'exported_models': [],
        'candidates': [{'program': Program('feature', name='u').to_dict(), 'status': 'candidate'}]}]
    registered = _register_native_theories(source, question, source.questions.index(question), 0)
    native = registered['registered'][0]
    source._before_model_turn(LocalLab())
    source._prepare_program_state()
    result = source.result('method_stopped')
    outgoing = result['outgoing_knowledge']
    assert 'files' not in outgoing and not (source.workspace / 'training.json').exists()
    native_ref = next(q['native_condition_research'] for q in outgoing['question_traces']
                      if q['id'] == 'native_source')
    assert native_ref['full_content_file'] == 'research_state.json'
    write = Path.write_text

    def no_paid_trace_rewrite(path, *args, **kwargs):
        assert path.name != 'paid_trace.json', 'Cold construction must install references before file writes'
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', no_paid_trace_rewrite)
    target = study.RegisteredOrganizationAgent(None, tmp_path / 'cold_target', arm='pr3',
        research_organization=registration() | {'phase': 'target',
            'initial_library_path': prepared['initial_library_path'],
            'pr3_library_path': str(source.workspace / 'shared/theory_library.json')},
        knowledge=outgoing, source_task=False,
        config=InquiryLLMConfig(compute_budget=10000, max_model_calls=0, confirmation_rows=2))
    assert (target.workspace / trace).is_symlink()
    assert (target.workspace / trace).resolve() == (source.workspace / trace).resolve()
    assert (target.workspace / DRIVER).read_text() == memory['files'][DRIVER]
    assert (target.workspace / 'old.py').read_text() == OLD_MODEL
    assert (target.workspace / 'new.py').read_text() == NEW_MODEL
    assert target.knowledge['lessons'] == memory['lessons']
    assert target.knowledge['theory_index'] == outgoing['theory_index']
    retained = next(q['native_condition_research'] for q in target.knowledge['question_traces']
                    if q['id'] == 'native_source')
    assert retained == native_ref
    imported = target.theory_index[native['theory_id']]
    assert imported['source_snapshot']['source_ref'] == native['source_ref']
    assert imported['support']['status'] == 'candidate'
    assert Path(imported['program_source_ref']['file']).is_absolute()
    assert (target.workspace / 'parameter.json').is_symlink() is False
    (target.workspace / 'parameter.json').write_text('{"value": 99}')
    assert (source.workspace / 'parameter.json').read_text() == '{"value": 3}'
    execution = study.execute_procedure(target, LocalLab(), DRIVER, max_steps=1,
        allow_confirmation=False, record_events=True)
    assert execution['status'] == 'method_stopped'
    assert target.costs['observation_cost'] == target.costs['model_calls'] == 0
