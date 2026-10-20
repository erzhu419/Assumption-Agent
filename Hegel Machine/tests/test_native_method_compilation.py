"""Compilation phase wiring; scripted publications are not discovery evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.structural_inquiry import NATIVE_CONDITION_INTERFACE
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, register_published_method
from scripts import run_anonymous_method_revision as revision
from scripts import run_native_method_compilation as compilation
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from test_anonymous_method_learning import memoryless_source, read_json, write_json
from test_conditional_design_study import prospective_source
from test_prospective_design_study import acquisition_source
from test_recurrent_method_revision import SOURCE, revision_sources


def test_paid_source_repair_accepts_unconfirmed_scene_without_publication(tmp_path):
    study = tmp_path / 'scene'
    workspace = study / 'source_006' / 'hegel'
    write_json(workspace / 'training.json', {
        'status': 'model_call_budget_exhausted', 'model': None,
        'costs': {'observation_cost': 1941, 'model_calls': 272},
        'fresh_confirmation': {'passed': False, 'mse': 4.47, 'rows': 96}})
    write_json(workspace / 'research_context.json', {})
    (workspace / METHOD_PATH).parent.mkdir(parents=True)
    (workspace / METHOD_PATH).write_text('def select_actions(*args): return {"stop": "review"}\n')
    write_json(study / 'initial_archive.json', {'files': {}})
    write_json(study / 'protocol.json', {'cases': [{'id': 'source_006'}],
        'model': 'gpt-6-luna', 'provider': {'backend': 'codex_cli'}})

    plan = compilation.paid_source_repair_protocol(workspace / 'training.json')

    assert plan['source_status'] == 'model_call_budget_exhausted'
    assert plan['source_costs']['observation_cost'] == 1941
    assert plan['author_limits']['observation_budget'] == 4096
    assert plan['author_limits']['confirmation_rows'] == 0
    assert plan['author_limits']['max_model_calls'] == 24
    assert plan['limits']['confirmation_rows'] == 96
    assert plan['limits']['max_model_calls'] == 0
    assert list(plan['practice_cases']) == ['source_006']
    assert 'transfer_cases' not in plan

    old = read_json(workspace / 'training.json')
    old.update(model='model.py', fresh_confirmation={'passed': True})
    write_json(workspace / 'training.json', old)
    with pytest.raises(ValueError, match='already confirmed'):
        compilation.paid_source_repair_protocol(workspace / 'training.json')


@pytest.mark.parametrize('additional_calls', [0, 8])
def test_paid_source_author_resume_keeps_trials_and_only_remaining_account(tmp_path, monkeypatch, additional_calls):
    from hegel_machine.cognition.anonymous_method_learning import SourceProcedureRepairLearner
    from test_paid_scene_method_repair import make_repair
    from test_temporal_python_refit import LawLab

    author, source_checkpoint, _ = make_repair(tmp_path / 'old')
    baseline = author._practice('same_scene')
    author.events.extend([
        {'step': -1, 'action': 'run', 'arguments': {'operator': 'practice', 'case': 'same_scene'},
         'feedback': baseline, 'usage': {'total_tokens': 0}},
        {'step': 0, 'action': 'run', 'arguments': {}, 'feedback': {},
         'usage': {'total_tokens': 100}},
        {'step': 1, 'action': 'provider_error', 'error': 'interrupted', 'usage': {}}])
    author.costs.update(model_calls=2, reported_tokens=None)
    previous = author.result('provider_failed')
    saved = author.workspace / 'training.json'
    write_json(saved, previous)
    limits = dict(observation_budget=4096, compute_budget=2_000_000,
        execution_seconds=7200., run_timeout_seconds=300., confirmation_rows=12,
        confirmation_mse_threshold=1e-6, max_model_calls=0, token_budget=0)
    archive = author.workspace.parent / 'initial_archive.json'
    write_json(archive, author.knowledge)
    plan = {'study': 'luna_paid_source_procedure_repair', 'source_checkpoint': str(source_checkpoint),
        'initial_archive': str(archive), 'model': 'fixture', 'provider': {},
        'practice_cases': {'same_scene': {'confirmation_seed': 173}}, 'limits': limits,
        'author_limits': {**limits, 'confirmation_rows': 0, 'max_model_calls': 24,
                          'token_budget': 12_000_000}, 'runtime_snapshot': [], 'scope': 'fixture'}
    write_json(author.workspace.parent / 'protocol.json', plan)
    continued = (compilation.continuation_protocol(saved, additional_author_calls=additional_calls)
                 if additional_calls else compilation.continuation_protocol(saved))
    assert continued['author_limits']['max_model_calls'] == 22 + additional_calls
    assert continued['author_total_limits']['max_model_calls'] == 24 + additional_calls
    assert continued['additional_author_calls'] == additional_calls
    assert continued['author_limits']['token_budget'] == 11_999_900
    assert continued['author_limits']['observation_budget'] == 4096 - author.costs['observation_cost']
    assert continued['author_limits']['compute_budget'] == 2_000_000 - author.costs['compute_cost']
    assert continued['author_limits']['execution_seconds'] == 7200. - author.costs['python_execution_seconds']
    assert continued['prior_unknown_usage_calls'] == 1
    assert continued['prior_author_phases'][0]['costs']['reported_tokens'] is None
    brief = 'Compare retained candidates with one saved trajectory partition; report every fold.'
    continued['research_brief'] = {'path': 'operator_brief.md', 'text': brief}
    trial = author.workspace / baseline['directory']
    original_bytes = (trial / 'result.json').read_bytes()
    monkeypatch.setattr(compilation, 'laboratory', lambda _: LawLab(slope=-2., offset=.4))

    def no_baseline(*args, **kwargs):
        pytest.fail('resuming an author must not replay its paid baseline')

    monkeypatch.setattr(SourceProcedureRepairLearner, '_practice', no_baseline)

    class Provider:
        def next_action(self, request):
            assert request['limits']['max_model_calls'] == 22 + additional_calls
            assert request['files'][METHOD_PATH] == (author.workspace / METHOD_PATH).read_text()
            assert request['paid_scene']['working_directory'] == baseline['directory']
            assert request['paid_scene']['training'] == baseline['directory'] + '/result.json'
            assert request['paid_scene']['initial_scene']['training'] == str(source_checkpoint)
            assert json.loads(request['files']['scene_source.json']) == request['paid_scene']
            assert 'CURRENT OPERATOR RESEARCH BRIEF\n' + brief in request['task_interface']
            return {'action': 'provider_error', 'error': 'fixture stop', 'usage': {'total_tokens': 0}}

    output = tmp_path / 'resumed'
    result = compilation.run_paid_source_repair(continued, output, provider_factory=lambda _: Provider())
    inherited_trial = output / 'author' / baseline['directory']
    assert inherited_trial.is_symlink() and inherited_trial.resolve() == trial.resolve()
    assert (output / 'author/scene_source').resolve() == trial.resolve()
    assert (trial / 'result.json').read_bytes() == original_bytes
    assert result['practice_trials'] == 1 and result['new_practice_trials'] == 0
    assert result['author_costs']['observation_cost'] == result['author_costs']['compute_cost'] == 0
    assert result['source_checkpoint'] == str(trial / 'result.json')
    assert result['method_published'] is False and result['target_episodes_run'] == 0
    assert read_json(output / 'protocol.json')['research_brief'] == continued['research_brief']
    again = compilation.continuation_protocol(output / 'author/training.json')
    assert again['author_limits']['max_model_calls'] == 21 + additional_calls
    assert again['author_total_limits']['max_model_calls'] == 24 + additional_calls
    assert again['additional_author_calls'] == 0
    assert again['author_limits']['token_budget'] == 11_999_900
    assert again['prior_unknown_usage_calls'] == 1


@pytest.fixture
def confirmed_compilation_source(tmp_path, prospective_source):
    method = revision.protocol(prospective_source, conditional_design=True)
    previous = read_json(prospective_source / 'author/training.json')
    # Existing fixture trials have source files; materialize their retained paid
    # result references as the real continuation driver requires.
    for trial in previous['practice_trials']:
        write_json(Path(previous['workspace']) / trial['directory'] / 'result.json', trial)
    method['continued_author'] = str(prospective_source / 'author/training.json')
    method['runtime_snapshot'] = []
    old_phase = {'result': str(prospective_source / 'author/training.json'),
        'status': previous['status'], 'costs': deepcopy(previous['costs'])}
    method['prior_author_phases'] = [old_phase]
    retained = [{'case': name, 'terminal': {'passed': True}, 'historical_only': True}
                for name in method['old_target_ids']]
    write_json(prospective_source / 'summary.json', {'frozen': {'revised': retained}})
    study, workspace = tmp_path / 'online', tmp_path / 'online/source/hegel'
    source_costs = {'observation_cost': 7901, 'measurement_cost': 0,
        'compute_cost': 123456, 'python_execution_seconds': 123.,
        'model_calls': 1027, 'reported_tokens': 1234567}
    confirmed = {'status': 'completed', 'model': 'online_only.py', 'costs': source_costs,
        'fresh_confirmation': {'passed': True, 'rows': 96, 'mse': 1e-23}}
    write_json(workspace / 'training.json', confirmed)
    (workspace / 'online_only.py').write_text('def predict(row): return 0.\n')
    write_json(workspace / 'laboratory_observations.json', [{'source_only_evidence': 93071}])
    write_json(workspace / 'archive.json', {'files': {'online_only.py': 'source-only fit'}})
    write_json(study / 'protocol.json', {'method': method})
    write_json(study / 'summary.json', {'trials': [{
        'status': 'completed', 'model': 'online_only.py', 'costs': source_costs,
        'fresh_confirmation_passed': True, 'terminal': {'passed': True},
        'archive': str(workspace / 'archive.json')}]})
    return study


@pytest.mark.parametrize('failed_part', ['status', 'confirmation', 'terminal'])
def test_source_requires_completed_independent_confirmation_and_terminal(
        confirmed_compilation_source, failed_part):
    study = confirmed_compilation_source
    summary = read_json(study / 'summary.json')
    trial = summary['trials'][0]
    if failed_part == 'status':
        trial['status'] = 'model_call_budget_exhausted'
    elif failed_part == 'confirmation':
        trial['fresh_confirmation_passed'] = False
    else:
        trial['terminal']['passed'] = False
    write_json(study / 'summary.json', summary)
    with pytest.raises(ValueError, match='source confirmation and terminal success'):
        compilation.protocol(study)


def test_bounded_protocol_preserves_retention_and_paid_history(confirmed_compilation_source):
    study = confirmed_compilation_source
    old = read_json(study / 'protocol.json')['method']
    plan = compilation.protocol(study)
    old_ids = set(old['old_target_ids'])
    assert plan['old_target_ids'] == old['old_target_ids']
    assert [case for case in plan['transfer_cases'] if case['id'] in old_ids] == [
        case for case in old['transfer_cases'] if case['id'] in old_ids]
    assert len(old_ids) == 10
    new = [case for case in plan['transfer_cases'] if case['id'] not in old_ids]
    assert {(case['parameter_pair'], case['stage']) for case in new} == {
        (0, 1), (0, 2), (1, 1), (1, 2)}
    assert all(case in old['transfer_cases'] for case in new)
    assert len(plan['practice_cases']) == 8
    assert all(case == old['practice_cases'][name] for name, case in plan['practice_cases'].items())
    assert plan['prior_author_phases'] == old['prior_author_phases']
    assert plan['continued_author'] == old['continued_author']
    assert plan['source_research_costs'] == read_json(study / 'summary.json')['trials'][0]['costs']
    assert not any(plan.get(name) for name in
                   ('design_experiments', 'prospective_design', 'conditional_design'))
    assert read_json(study / 'protocol.json')['method'] == old


def test_author_continuation_keeps_cohorts_drafts_and_paid_phase(
        tmp_path, confirmed_compilation_source):
    plan = compilation.protocol(confirmed_compilation_source)
    plan['restore_accepted'] = str(tmp_path / 'accepted_baseline')
    study = tmp_path / 'exhausted'
    previous = read_json(Path(plan['continued_author']))
    previous.update(workspace=str(study / 'author'),
                    status='model_call_budget_exhausted', method_source=None)
    previous['costs']['model_calls'] = 32
    # A compact restore projection points back to the complete paid record.
    previous['full_result'] = str(study / 'author/training.json')
    path = tmp_path / 'next/author_resume.json'
    write_json(path, previous)
    write_json(study / 'protocol.json', plan)
    usage = {'input_tokens': 2413264, 'output_tokens': 43704}
    write_json(study / 'summary.json', {
        'method_published': False, 'author_usage': usage})

    continued = compilation.continuation_protocol(path)
    assert 'restore_accepted' not in continued
    assert continued['continued_author'] == str(path.resolve())
    for key in ('practice_cases', 'transfer_cases', 'author_limits', 'limits'):
        assert continued[key] == plan[key]
    assert continued['prior_author_phases'][:-1] == plan['prior_author_phases']
    assert continued['prior_author_phases'][-1] == {
        'phase': 'native_method_compilation', 'result': previous['full_result'],
        'status': previous['status'], 'costs': previous['costs'], 'usage': usage}
    assert read_json(path) == previous
    assert read_json(study / 'protocol.json') == plan
    write_json(study / 'summary.json', {
        'method_published': True, 'author_usage': usage})
    with pytest.raises(ValueError, match='unpublished compilation author'):
        compilation.continuation_protocol(path)


@pytest.mark.parametrize('unpublished', [False, True])
def test_completion_request_replaces_revision_assignment_without_changing_source_or_support(
        tmp_path, confirmed_compilation_source, monkeypatch, unpublished):
    from hegel_machine.cognition.code_provider_context import prepare_code_context

    plan = compilation.protocol(confirmed_compilation_source)
    plan.update(runtime_snapshot=[], author_only=True, transfer_cases=[], complete_practices=True,
                published_revision={'frozen_study': 'retained_frozen_study'})
    restored = read_json(Path(plan['continued_author']))
    if unpublished:
        restored.update(status='model_call_budget_exhausted', method_source=None)
        plan['continued_author'] = str(tmp_path / 'continued.json')
        write_json(Path(plan['continued_author']), restored)

    def author_run(author, *, on_checkpoint):
        source = (author.workspace / METHOD_PATH).read_text()
        coverage = deepcopy(author._current_source_coverage())
        request = author._request(author.interface)
        context = prepare_code_context(author.workspace, request, schema={}, instructions='test')
        task = context['task_interface']
        assert 'CURRENT DELIVERABLE: complete the restored source' in task
        assert 'retain confirmed cases and practice only missing cases' in task
        assert 'independently decide whether to publish' in task
        assert 'CURRENT DELIVERABLE: use the retained paid comparisons' not in task
        assert author._current_source_coverage() == coverage
        assert author.practice_trials == restored['practice_trials']
        assert (author.workspace / METHOD_PATH).read_text() == source
        assert author.model is None and author.published_source is None
        assert author.costs['model_calls'] == author.costs['observation_cost'] == 0
        return author.result('model_call_budget_exhausted')

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    result = compilation.run_compilation(plan, tmp_path / 'completion', provider_factory=lambda _: None)
    assert result['method_published'] is False and result['target_episodes_run'] == 0
    assert result['published_revision']['source_changed'] is False
    assert read_json(Path(plan['continued_author'])) == restored


def test_user_accepted_draft_transfers_without_fabricating_author_publication(
        tmp_path, confirmed_compilation_source, monkeypatch):
    plan = compilation.protocol(confirmed_compilation_source)
    for i in range(8, 12):
        plan['practice_cases'][f'design_case_{i:03d}'] = deepcopy(plan['practice_cases']['design_case_007'])
    study, output = tmp_path / 'partial_author', tmp_path / 'accepted_targets'
    prior = read_json(Path(plan['continued_author']))
    source = SOURCE + '# frozen partial development draft\n'
    prior.update(status='model_call_budget_exhausted', method_source=None,
        outgoing_knowledge={'files': {METHOD_PATH: source, 'author_only.py': 'must not transfer'}},
        practice_trials=[{'case': name, 'directory': f'trials/trial_{i:03d}',
            'method_source': source, 'status': 'completed' if i < 11 else 'confirmation_failed',
            'fresh_confirmation': {'passed': i < 11, 'mse': 0. if i < 11 else 2e-4}}
            for i, name in enumerate(plan['practice_cases'])])
    write_json(study / 'protocol.json', plan)
    write_json(study / 'author/training.json', prior)
    original_summary = {'method_published': False, 'author_usage': {'known_tokens': 123}}
    write_json(study / 'summary.json', original_summary)
    frozen_plan, selected = compilation.prepare_accepted_draft(study, output)
    accepted = frozen_plan['accepted_draft']
    assert len(accepted['confirmed_cases']) == 11
    assert accepted['unconfirmed_cases'] == ['design_case_011']
    assert selected['status'] == 'user_accepted_draft'
    assert (output / 'accepted_method.py').read_text() == source
    assert frozen_plan['transfer_cases'] == plan['transfer_cases']
    assert frozen_plan['limits'] == plan['limits']
    original = read_json(Path(plan['source']) / 'initial_archive.json')
    before = deepcopy(original)
    order = []

    def frozen(part, branch, memory, sources, method_source, *, agent_type):
        assert method_source == source and memory['files'][METHOD_PATH] == source
        assert 'author_only.py' not in memory['files']
        assert agent_type is ExperimentalDesignMethodAgent
        entry, = [item for item in memory['manifest']['research_programs'] if item['path'] == METHOD_PATH]
        assert entry['admission'] == 'user_accepted_development_draft'
        assert entry['prior_execution']['cases'][-1]['fresh_confirmation']['passed'] is False
        order.append('all_research_stopped')
        return []

    def score(part, results):
        assert order == ['all_research_stopped']
        return [{'case': case['id'], 'terminal': {'passed': True}} for case in part['transfer_cases']]

    monkeypatch.setattr(compilation, 'frozen_transfers', frozen)
    monkeypatch.setattr(compilation, 'score_frozen', score)
    monkeypatch.setattr(compilation, 'evaluate_python_model', lambda *a, **kw: {'returncode': 0, 'mse': 1.})
    result = compilation.run_targets(frozen_plan, output, selected, original, [])
    assert result['method_published'] is False
    assert result['accepted_draft']['author_publication_passed'] is False
    assert result['live_consumer_model_calls'] == 0
    assert len(result['retention']) == 10 and len(result['new_parameters']) == 4
    assert result['new_practice_trials'] == 0
    assert original == before
    assert read_json(study / 'author/training.json') == prior
    assert read_json(study / 'summary.json') == original_summary
    assert read_json(study / 'protocol.json') == plan


@pytest.fixture
def published_compilation(tmp_path, confirmed_compilation_source):
    plan = compilation.protocol(confirmed_compilation_source)
    plan['runtime_snapshot'] = []
    for i in range(8, 12):
        plan['practice_cases'][f'design_case_{i:03d}'] = deepcopy(plan['practice_cases']['design_case_007'])
    study = tmp_path / 'published_author'
    source = SOURCE + '# published adaptive-boundary fixture\n'
    trials = [{'case': case, 'directory': f'trials/trial_{i:03d}',
        'status': 'completed', 'fresh_confirmation': {'passed': True, 'mse': 1e-20, 'rows': 96}}
        for i, case in enumerate(plan['practice_cases'])]
    support = {'current': True, 'source_path': f'{trials[0]["directory"]}/{METHOD_PATH}',
               'trials': trials}
    prior = {'current': False, 'trials': [
        {'case': 'old_failure', 'directory': 'trials/trial_099', 'status': 'confirmation_failed'}]}
    write_json(study / 'protocol.json', plan)
    write_json(study / 'published_method.json', {'published': True, 'source': source,
                                               'lesson': 'fixture publication'})
    write_json(study / 'summary.json', {'method_published': True,
        'author_costs': {'model_calls': 27, 'observation_cost': 3341},
        'author_usage': {'known_tokens': 2635259}})
    write_json(study / 'author/method_candidates.json', [prior, support])
    supported_source = study / 'author' / support['source_path']
    supported_source.parent.mkdir(parents=True, exist_ok=True)
    supported_source.write_text(source)
    # A fresh freeze must use the indexed support, not this large author export.
    (study / 'author/training.json').write_text('not a JSON input to frozen evaluation')
    (study / 'reference_archive').mkdir()
    return study


def test_freeze_published_uses_small_support_and_registers_unopened_parameter_groups(
        tmp_path, published_compilation):
    study, output = published_compilation, tmp_path / 'frozen_publication'
    old = read_json(study / 'protocol.json')
    published = read_json(study / 'published_method.json')
    plan, learned = compilation.prepare_published_targets(study, output)
    old_ids = set(old['old_target_ids'])
    assert [case for case in plan['transfer_cases'] if case['id'] in old_ids] == [
        case for case in old['transfer_cases'] if case['id'] in old_ids]
    new = [case for case in plan['transfer_cases'] if case['id'] not in old_ids]
    assert len(plan['transfer_cases']) == 18 and len(new) == 8
    assert {(case['parameter_pair'], case['stage']) for case in new} == {
        (pair, stage) for pair in range(4) for stage in (1, 2)}
    old_seeds = {case['apparatus_seed'] for case in [
        *old['practice_cases'].values(), *old['transfer_cases']]}
    assert not old_seeds.intersection(case['apparatus_seed'] for case in new)
    assert plan['limits'] == old['limits'] and plan['method_steps'] == old['method_steps']
    assert all(case['limits']['confirmation_mse_threshold'] == 1e-6 for case in new)
    assert plan['parameter_registration']['previously_opened_cases'] == [
        case for case in old['transfer_cases'] if case['id'] not in old_ids]
    assert learned['retained_paid_practices'] == 13
    assert len(learned['practice_trials']) == 12
    assert all(trial['method_source'] == published['source'] for trial in learned['practice_trials'])
    assert (output / 'published_method.py').read_text() == published['source']
    assert read_json(output / 'published_result.json') == learned
    assert (output / 'reference_archive').resolve() == study / 'reference_archive'
    assert not (output / 'author').exists()
    assert read_json(study / 'published_method.json') == published
    assert read_json(study / 'protocol.json') == old
    with pytest.raises(ValueError, match='preserve earlier attempts'):
        compilation.prepare_published_targets(study, output)


@pytest.mark.parametrize('pair', ['none', 'published', 'accepted'])
def test_published_freeze_saves_current_runtime_once_for_both_plans(
        tmp_path, published_compilation, monkeypatch, pair):
    study, output = published_compilation, tmp_path / 'new_frozen_targets'
    old = read_json(study / 'protocol.json')
    old['runtime_snapshot'] = ['scripts/old_main.py']
    write_json(study / 'protocol.json', old)
    publication = read_json(study / 'published_method.json')
    root = tmp_path / 'current_checkout'
    current = {
        'src/hegel_machine/cognition/question_routing.py': '# current route\n',
        'src/hegel_machine/cognition/new_question_module.py': '# newly added module\n',
        'scripts/run_native_method_compilation.py': '# current preparation entry\n',
        'scripts/old_main.py': '# retained main entry\n',
        'scripts/old_baseline.py': '# retained baseline entry\n',
    }
    for name, content in current.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(compilation, 'ROOT', root)
    options = {}
    if pair != 'none':
        baseline = tmp_path / 'old_baseline'
        if pair == 'published':
            shutil.copytree(study, baseline)
            baseline_old = deepcopy(old)
            baseline_old['runtime_snapshot'] = ['scripts/old_baseline.py']
            write_json(baseline / 'protocol.json', baseline_old)
            registered = {
                'required_practice_count': len(old['practice_cases']),
                'regression_protocol': str(study / 'protocol.json'),
                'regression_case_ids': [old['old_target_ids'][0]],
                'regression_groups': {'original': [old['old_target_ids'][0]]},
                'new_transfer_cases': [deepcopy(old['transfer_cases'][-1])],
                'new_transfer_parameter_groups': 1, 'paired_baseline': str(baseline),
                'limits': deepcopy(old['limits']), 'method_steps': old['method_steps'],
                'evaluation': deepcopy(old['evaluation']),
            }
            registration = tmp_path / 'registered_targets.json'
            write_json(registration, {'frozen_evaluation_registration': registered})
            options['evaluation_registration'] = registration
        else:
            baseline_old = {'runtime_snapshot': ['scripts/old_baseline.py'],
                'accepted_draft': {'basis': 'user_accepted_development_draft', 'practice_cases': []}}
            write_json(baseline / 'protocol.json', baseline_old)
            write_json(baseline / 'accepted_draft.json', {
                'status': 'user_accepted_draft', 'method_source': '# old accepted method\n'})
            options['baseline_study'] = baseline
    plan, learned = compilation.prepare_published_targets(study, output, **options)
    expected = set(current) - ({'scripts/old_baseline.py'} if pair == 'none' else set())
    assert plan['runtime_snapshot'] == sorted(expected)
    assert read_json(output / 'protocol.json')['runtime_snapshot'] == sorted(expected)
    for name in expected:
        assert (output / 'source_at_run' / name).read_text() == current[name]
    if pair != 'none':
        paired = output / plan['paired_baseline']['directory']
        paired_plan = read_json(paired / 'protocol.json')
        assert paired_plan['runtime_snapshot'] == plan['runtime_snapshot']
        assert not (paired / 'source_at_run').exists()  # One shared source copy.
        assert read_json(baseline / 'protocol.json') == baseline_old
        assert paired_plan['limits'] == plan['limits']
        assert paired_plan['transfer_cases'] == [
            case for case in plan['transfer_cases'] if case['id'] not in plan['old_target_ids']]
    assert learned['method_source'] == publication['source']
    assert plan['limits'] == old['limits']
    assert read_json(study / 'protocol.json') == old
    assert read_json(study / 'published_method.json') == publication


def test_published_freeze_requires_actual_same_source_confirmation(
        tmp_path, published_compilation):
    study = published_compilation
    candidates = read_json(study / 'author/method_candidates.json')
    candidates[-1]['trials'][-1]['fresh_confirmation']['passed'] = False
    write_json(study / 'author/method_candidates.json', candidates)
    with pytest.raises(ValueError, match='cover every listed practice'):
        compilation.prepare_published_targets(study, tmp_path / 'unconfirmed')
    candidates[-1]['trials'][-1]['fresh_confirmation']['passed'] = True
    write_json(study / 'author/method_candidates.json', candidates)
    path = study / 'author' / candidates[-1]['source_path']
    path.write_text(SOURCE + '# a different draft\n')
    with pytest.raises(ValueError, match='actual published source'):
        compilation.prepare_published_targets(study, tmp_path / 'different_source')


@pytest.fixture
def registered_structural_publication(tmp_path, published_compilation):
    baseline = tmp_path / 'published_twelve'
    plan, _ = compilation.prepare_published_targets(published_compilation, baseline)
    registered = read_json(compilation.ROOT /
        'docs/evidence/conditional_online_failure_20261013/structural_method_revision_20261017.json')[
            'frozen_evaluation_registration']
    registered.update(regression_protocol=str(baseline / 'protocol.json'), paired_baseline=str(baseline))
    registration_path = tmp_path / 'registered_structural_transfer.json'
    write_json(registration_path, {'frozen_evaluation_registration': registered})
    study = tmp_path / 'published_thirteen'
    plan['practice_cases']['design_case_012'] = {
        **deepcopy(registered['new_transfer_cases'][0]), 'apparatus_seed': 9912901}
    source = SOURCE + '# structural revision fixture\n'
    trials = [{'case': case, 'directory': f'trials/trial_{i:03d}', 'status': 'completed',
        'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}}
        for i, case in enumerate(plan['practice_cases'])]
    write_json(study / 'protocol.json', plan)
    write_json(study / 'published_method.json', {'published': True, 'source': source, 'lesson': 'new basis'})
    write_json(study / 'summary.json', {'method_published': True,
        'author_costs': {'model_calls': 10}, 'author_usage': {'known_tokens': 111}})
    write_json(study / 'author/method_candidates.json', [{
        'current': True, 'source_path': f'{trials[0]["directory"]}/{METHOD_PATH}', 'trials': trials}])
    path = study / 'author' / trials[0]['directory'] / METHOD_PATH
    path.parent.mkdir(parents=True)
    path.write_text(source)
    (study / 'author/training.json').write_text('do not load a full author export')
    (study / 'reference_archive').symlink_to(baseline / 'reference_archive', target_is_directory=True)
    return study, baseline, registration_path, registered


@pytest.mark.parametrize('failure', ['unpublished', 'missing_confirmation', 'only_twelve'])
def test_registered_structural_transfer_requires_thirteen_case_publication(
        tmp_path, registered_structural_publication, failure):
    study, _, registration, _ = registered_structural_publication
    if failure == 'unpublished':
        saved = read_json(study / 'summary.json')
        saved['method_published'] = False
        write_json(study / 'summary.json', saved)
    else:
        candidates = read_json(study / 'author/method_candidates.json')
        if failure == 'missing_confirmation':
            candidates[0]['trials'][-1]['fresh_confirmation']['passed'] = False
        else:
            candidates[0]['trials'].pop()
            plan = read_json(study / 'protocol.json')
            plan['practice_cases'].pop('design_case_012')
            write_json(study / 'protocol.json', plan)
        write_json(study / 'author/method_candidates.json', candidates)
    output = tmp_path / 'not_published'
    with pytest.raises(ValueError, match='published source|every listed practice|13-case publication'):
        compilation.prepare_published_targets(study, output, evaluation_registration=registration)
    assert not output.exists()


def test_registered_structural_transfer_keeps_eighteen_regressions_and_published_pair(
        tmp_path, registered_structural_publication, monkeypatch):
    study, baseline_study, registration_path, registration = registered_structural_publication
    output = tmp_path / 'structural_transfer'
    old_read = compilation.read_json
    def no_author_export(path):
        assert Path(path).name != 'training.json'
        return old_read(path)
    monkeypatch.setattr(compilation, 'read_json', no_author_export)
    plan, learned = compilation.prepare_published_targets(
        study, output, evaluation_registration=registration_path)
    opened = read_json(baseline_study / 'protocol.json')['transfer_cases']
    assert len(plan['transfer_cases']) == 26 and plan['transfer_cases'][:18] == opened
    assert plan['transfer_cases'][18:] == registration['new_transfer_cases']
    assert plan['old_target_ids'] == registration['regression_case_ids']
    assert plan['regression_groups'] == registration['regression_groups']
    assert plan['limits'] == registration['limits'] and plan['method_steps'] == registration['method_steps']
    assert len(plan['frozen_publication']['confirmed_cases']) == 13
    baseline = output / 'baseline_12of12'
    baseline_plan = read_json(baseline / 'protocol.json')
    assert baseline_plan['transfer_cases'] == registration['new_transfer_cases']
    assert 'accepted_draft' not in baseline_plan and not (baseline / 'accepted_draft.json').exists()
    assert 'paired_baseline' not in baseline_plan
    assert len(baseline_plan['frozen_publication']['confirmed_cases']) == 12
    assert read_json(baseline / 'published_result.json')['status'] == 'completed'
    assert baseline_plan['limits'] == plan['limits']
    assert (baseline / 'reference_archive').resolve() == (study / 'reference_archive').resolve()
    order = []
    def frozen(part, branch, memory, sources, method_source, *, agent_type):
        assert agent_type is ExperimentalDesignMethodAgent
        assert 'accepted_draft' not in part
        assert memory['files'][METHOD_PATH] == method_source
        entry, = [row for row in memory['manifest']['research_programs'] if row['path'] == METHOD_PATH]
        assert 'admission' not in entry
        if branch == output:
            assert len(part['transfer_cases']) == 26 and order == []
            order.append('new_research_stopped')
        else:
            assert branch == baseline and len(part['transfer_cases']) == 8
            assert order == ['new_research_stopped']
            order.append('baseline_research_stopped')
        return [(case, {}) for case in part['transfer_cases']]
    def score(part, results):
        assert order[:2] == ['new_research_stopped', 'baseline_research_stopped']
        order.append('terminal_scored')
        return [{'case': case['id'], 'terminal': {'passed': True}} for case, _ in results]
    monkeypatch.setattr(compilation, 'frozen_transfers', frozen)
    monkeypatch.setattr(compilation, 'score_frozen', score)
    # This is a protocol test; never generate the registered terminal labels.
    monkeypatch.setattr(compilation, 'laboratory', lambda _: SimpleNamespace(evaluation_rows=lambda **_: []))
    monkeypatch.setattr(compilation, 'evaluate_python_model', lambda *a, **k: {'returncode': 0, 'mse': 9.})
    original = read_json(Path(plan['source']) / 'initial_archive.json')
    result = compilation.run_targets(plan, output, learned, original, [])
    assert len(result['retention']) == 18 and len(result['new_parameters']) == 8
    assert {key: len(rows) for key, rows in result['retention_groups'].items()} == {
        'original_ten': 10, 'previously_successful_eight': 8}
    assert len(result['paired_baseline']['new_parameters']) == 8
    baseline_summary = read_json(baseline / 'summary.json')
    assert baseline_summary['method_published'] is True and 'accepted_draft' not in baseline_summary
    assert baseline_summary['author_usage'] == read_json(baseline / 'published_result.json')['author_usage']
    assert result['new_practice_trials'] == 0 and result['live_consumer_model_calls'] == 0


def test_registered_window_publication_preserves_twenty_six_regression_budgets_and_thirteen_case_pair(
        tmp_path, registered_structural_publication):
    prior, _, registration_path, registered = registered_structural_publication
    baseline = tmp_path / 'published_thirteen_frozen'
    baseline_plan, _ = compilation.prepare_published_targets(
        prior, baseline, evaluation_registration=registration_path)
    original_baseline = read_json(baseline / 'protocol.json')
    study, output = tmp_path / 'published_fourteen', tmp_path / 'window_transfer'
    plan = deepcopy(baseline_plan)
    plan['practice_cases']['design_case_013'] = {
        **deepcopy(plan['practice_cases']['design_case_012']), 'window_width': 3}
    source = SOURCE + '# published window fixture\n'
    trials = [{'case': case, 'directory': f'trials/trial_{i:03d}', 'status': 'completed',
        'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}}
        for i, case in enumerate(plan['practice_cases'])]
    write_json(study / 'protocol.json', plan)
    write_json(study / 'published_method.json', {'published': True, 'source': source})
    write_json(study / 'summary.json', {'method_published': True,
        'author_costs': {'model_calls': 8}, 'author_usage': {'known_tokens': 123}})
    write_json(study / 'author/method_candidates.json', [{
        'current': True, 'source_path': f'{trials[0]["directory"]}/{METHOD_PATH}', 'trials': trials}])
    source_path = study / 'author' / trials[0]['directory'] / METHOD_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source)
    (study / 'author/training.json').write_text('full paid history must not be loaded')
    (study / 'author/window_fit.py').write_text('source-fitted parameters must not transfer')
    (study / 'author/laboratory_observations.json').write_text('["private paid evidence"]')
    (study / 'reference_archive').symlink_to(baseline / 'reference_archive', target_is_directory=True)

    new_cases = deepcopy(registered['new_transfer_cases'])
    for index, case in enumerate(new_cases):
        case.update(id=f'window_target_{index:03d}', window_width=3)
        for key in ('seed', 'apparatus_seed', 'prefix_seed', 'confirmation_seed', 'evaluation_seed'):
            case[key] += 1_000_000
    registration = {**deepcopy(registered), 'required_practice_count': 14,
        'regression_protocol': str(baseline / 'protocol.json'),
        'regression_case_ids': [case['id'] for case in baseline_plan['transfer_cases']],
        'paired_baseline': str(baseline), 'new_transfer_cases': new_cases,
        'regression_groups': {**deepcopy(registered['regression_groups']),
            'previously_successful_nested_eight': [case['id'] for case in registered['new_transfer_cases']]},
        'limits': {**registered['limits'], 'observation_budget': 4096, 'compute_budget': 2_000_000}}
    registered_path = tmp_path / 'window_registration.json'
    write_json(registered_path, {'frozen_evaluation_registration': registration})
    frozen, learned = compilation.prepare_published_targets(
        study, output, evaluation_registration=registered_path)

    assert len(frozen['frozen_publication']['confirmed_cases']) == 14
    assert len(frozen['transfer_cases']) == 34 and frozen['transfer_cases'][26:] == new_cases
    assert {name: len(cases) for name, cases in frozen['regression_groups'].items()} == {
        'original_ten': 10, 'previously_successful_eight': 8, 'previously_successful_nested_eight': 8}
    for before, after in zip(baseline_plan['transfer_cases'], frozen['transfer_cases'][:26]):
        assert {key: value for key, value in before.items() if key != 'limits'} == {
            key: value for key, value in after.items() if key != 'limits'}
        for key in ('observation_budget', 'compute_budget', 'execution_seconds'):
            old_limit = min(baseline_plan['limits'][key], before.get('limits', {}).get(
                key, baseline_plan['limits'][key]))
            assert min(frozen['limits'][key], after['limits'].get(key, frozen['limits'][key])) == old_limit
        assert after['limits']['confirmation_mse_threshold'] == before['limits']['confirmation_mse_threshold']
    paired = output / 'baseline_13of13'
    pair_plan = read_json(paired / 'protocol.json')
    assert frozen['paired_baseline']['directory'] == paired.name
    assert len(pair_plan['frozen_publication']['confirmed_cases']) == 13
    assert pair_plan['transfer_cases'] == new_cases and pair_plan['limits'] == frozen['limits']
    assert '14-case' in frozen['scope'] and '26 previously opened' in frozen['scope']
    assert '13-case' in pair_plan['scope'] and '8 registered' in pair_plan['scope']
    assert (output / 'published_method.py').read_text() == source
    assert not (output / 'author').exists() and not (paired / 'author').exists()
    original = read_json(Path(frozen['source']) / 'initial_archive.json')
    memory = compilation.transfer_memory(frozen, learned, original)
    assert set(memory['files']) == set(original['files']) | {METHOD_PATH}
    assert memory['files'][METHOD_PATH] == source
    assert all('source-fitted parameters' not in text and 'private paid evidence' not in text
               for text in memory['files'].values())
    assert read_json(baseline / 'protocol.json') == original_baseline
    assert read_json(study / 'protocol.json') == plan
    assert (study / 'author/laboratory_observations.json').read_text() == '["private paid evidence"]'


@pytest.fixture
def published_revision_inputs(tmp_path, published_compilation, confirmed_compilation_source):
    study, frozen = published_compilation, tmp_path / 'frozen_for_revision'
    compilation.prepare_published_targets(study, frozen)
    source = read_json(frozen / 'published_result.json')['method_source']
    cases = list(read_json(study / 'protocol.json')['practice_cases'])
    drafts = []
    for index in range(108):
        method = source if index == 107 else SOURCE + f'# earlier draft {index}\n'
        indices = (range(243, 255) if index == 107 else
                   [number for number in range(243) if number % 107 == index])
        trials = []
        for number in indices:
            directory = f'trials/trial_{number:03d}'
            path = study / 'author' / directory
            write_json(path / 'result.json', {'paid_record': number})
            (path / METHOD_PATH).parent.mkdir(parents=True, exist_ok=True)
            (path / METHOD_PATH).write_text(method)
            trials.append({'case': cases[number - 243] if index == 107 else 'old_case',
                'directory': directory, 'status': 'completed',
                'fresh_confirmation': {'passed': True, 'rows': 96, 'mse': 0.},
                'selected_prior_predictor': number == 243})
        drafts.append({'current': index == 107,
            'source_path': f'{trials[0]["directory"]}/{METHOD_PATH}', 'trials': trials})
    write_json(study / 'author/method_candidates.json', drafts)
    recent = {'case': cases[-1], 'directory': 'trials/trial_254',
        'source_path': f'trials/trial_254/{METHOD_PATH}', 'costs': {'compute_cost': 61736},
        'execution_summary': {'research_steps': [], 'submission_count': 0,
            'latest_evaluation': {'model': 'final_adaptive.py', 'mse': 4e-20}}}
    prior_view = {'directory': 'trials/trial_000',
        'source_path': f'trials/trial_000/{METHOD_PATH}',
        'execution_summary': {'research_steps': [], 'submission_count': 0,
                              'latest_evaluation': {'model': 'older.py', 'mse': 7.}}}
    write_json(study / 'author/method_decision.json', {
        'latest_practice': recent, 'latest_evaluated_practice': recent,
        'current_source_latest_evaluation': recent,
        'failed_case_paid_comparison': {'previous_supported_practice': prior_view}})
    # A recent result may be huge; its saved small summary is sufficient for
    # startup. Fail immediately if the driver tries to parse the full record.
    (study / 'author/trials/trial_254/result.json').write_text('full paid result must not be loaded')
    (study / 'author/research_notes.md').write_text('Current published notes; retain adaptive boundary updates.')
    online = confirmed_compilation_source
    case = {'id': 'nested_source', 'kind': 'nested_response', 'seed': 181031,
        'apparatus_seed': 181031, 'prefix_seed': 281034, 'confirmation_seed': 381034,
        'evaluation_seed': 481038, 'response': 'nested', 'window_width': 1,
        'limits': {'confirmation_mse_threshold': 1e-6}}
    write_json(online / 'protocol.json', {'cases': [case]})
    return online, frozen, study, case


@pytest.mark.parametrize('failure', ['status', 'confirmation', 'terminal'])
def test_structural_compilation_waits_for_actual_source_success(
        tmp_path, published_revision_inputs, failure):
    online, frozen, _, _ = published_revision_inputs
    summary = read_json(online / 'summary.json')
    trial = summary['trials'][0]
    if failure == 'status':
        trial['status'] = 'method_stopped'
    elif failure == 'confirmation':
        trial['fresh_confirmation_passed'] = False
    else:
        trial['terminal']['passed'] = False
    write_json(online / 'summary.json', summary)
    output = tmp_path / 'not_ready'
    with pytest.raises(ValueError, match='source confirmation and terminal success'):
        compilation.prepare_revision_compilation(online, frozen, output)
    assert not output.exists()


def test_structural_compilation_restores_current_source_all_paid_drafts_and_notes(
        tmp_path, published_revision_inputs, monkeypatch):
    online, frozen, study, source_case = published_revision_inputs
    output = tmp_path / 'structural_author'
    baseline = read_json(frozen / 'published_result.json')['method_source']
    original = read_json(frozen / 'protocol.json')
    old_read = compilation.read_json

    def no_full_author_export(path):
        assert Path(path).name != 'training.json'
        return old_read(path)

    monkeypatch.setattr(compilation, 'read_json', no_full_author_export)
    plan = compilation.prepare_revision_compilation(online, frozen, output)
    restored = read_json(output / 'author_resume.json')
    assert restored['method_source'] == baseline
    assert restored['workspace'] == str(study / 'author')
    assert len(restored['practice_trials']) == 255
    assert len({trial['method_source'] for trial in restored['practice_trials']}) == 108
    assert [trial['directory'] for trial in restored['practice_trials']] == [
        f'trials/trial_{i:03d}' for i in range(255)]
    assert restored['author_research_notes'] == (study / 'author/research_notes.md').read_text()
    assert restored['practice_trials'][-1]['costs'] == {'compute_cost': 61736}
    assert restored['practice_trials'][-1]['execution_summary']['latest_evaluation']['model'] == 'final_adaptive.py'
    assert restored['practice_trials'][0]['execution_summary']['latest_evaluation']['model'] == 'older.py'
    assert plan['practice_cases'] == {**original['practice_cases'],
        'design_case_012': {key: value for key, value in source_case.items()
                            if key not in {'id', 'parameter_pair', 'evaluation_seed'}}}
    assert plan['author_only'] and plan['transfer_cases'] == original['transfer_cases']
    assert plan['restored_publication']['retained_paid_practices'] == 255
    assert plan['restored_publication']['retained_drafts'] == 108
    assert 'repair_study' not in plan and 'restore_accepted' not in plan
    assert read_json(frozen / 'protocol.json') == original
    assert (study / 'author/training.json').read_text().startswith('not a JSON')

    def author_run(author, *, on_checkpoint):
        assert (author.workspace / METHOD_PATH).read_text() == baseline
        assert author.files['research_notes.md'] == restored['author_research_notes']
        assert author._current_source_coverage()['missing_cases'] == ['design_case_012']
        assert len(author.practice_trials) == 255
        author._request(author.interface)
        decision = read_json(author.workspace / 'method_decision.json')
        assert decision['latest_practice']['directory'] == 'trials/trial_254'
        assert decision['current_source_latest_evaluation']['execution_summary']['latest_evaluation']['model'] == 'final_adaptive.py'
        for trial in author.practice_trials:
            assert (author.workspace / trial['directory']).resolve() == (
                study / 'author' / trial['directory']).resolve()
        candidates = read_json(author.workspace / 'method_candidates.json')
        assert len(candidates) == 108 and sum(item['current'] for item in candidates) == 1
        assert 'online_only.py' not in author.knowledge['files']
        assert not any('online_research_experience' in name for name in author.knowledge['files'])
        # The actual trial preparation uses only the original archive and method;
        # the source fit and its already paid observations are author-only.
        case = plan['practice_cases']['design_case_000']
        payload = author.prepare_trial(case, tmp_path / 'isolated_trial', baseline, author.practice_limits)
        assert len(payload['agent'].rows) == 72
        assert not (payload['agent'].workspace / 'online_only.py').exists()
        assert not (payload['agent'].workspace / 'online_research_experience').exists()
        linked = read_json(author.workspace / 'online_research_experience.json')
        assert (author.workspace / linked['model']).is_file()
        return author.result('model_call_budget_exhausted')

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    plan['runtime_snapshot'] = []
    result = compilation.run_compilation(plan, output, provider_factory=lambda _: None)
    assert result['method_published'] is False and result['target_episodes_run'] == 0
    assert not (output / 'transfer_archive.json').exists()


@pytest.fixture
def matched_compilation_inputs(published_revision_inputs):
    online, frozen, study, case = published_revision_inputs
    baseline = read_json(frozen / 'protocol.json')
    case = {**case, 'compilation_confirmation_seed': case['confirmation_seed'] + 1000003}
    write_json(online / 'protocol.json', {'cases': [case], 'model': 'gpt-6-luna',
        'provider': {'backend': 'codex_cli', 'reasoning_effort': 'high', 'timeout_seconds': 1800.},
        'limits': {**baseline['limits'], 'observation_budget': 4096,
                   'compute_budget': 2000000, 'execution_seconds': 7200.}})
    template = read_json(online / 'source/hegel/training.json')
    trials = []
    for arm in ('hegel', 'code_react'):
        workspace = online / case['id'] / arm
        training = {**deepcopy(template), 'agent_arm': arm,
                    'model': f'{arm}_source.py', 'events': []}
        write_json(workspace / 'training.json', training)
        (workspace / training['model']).write_text('def predict(row): return 0.\n')
        write_json(workspace / 'laboratory_observations.json', [{'own_arm': arm}])
        write_json(workspace / 'archive.json', {'files': {training['model']: arm}})
        write_json(workspace / 'evaluation.json', {'terminal_private': arm})
        trials.append({'case': case['id'], 'arm': arm, 'status': 'completed',
            'model': training['model'], 'costs': training['costs'],
            'fresh_confirmation_passed': True, 'terminal': {'passed': True},
            'archive': str(workspace / 'archive.json')})
    write_json(online / 'summary.json', {'trials': trials})
    return online, frozen, study, case


@pytest.mark.parametrize('failure', ['status', 'confirmation', 'terminal', 'confirmation_seed'])
def test_matched_compilation_requires_selected_arm_success_and_fresh_stream(
        tmp_path, matched_compilation_inputs, failure):
    online, frozen, _, case = matched_compilation_inputs
    summary = read_json(online / 'summary.json')
    trial = summary['trials'][1]
    if failure == 'status':
        trial['status'] = 'model_call_budget_exhausted'
    elif failure == 'confirmation':
        trial['fresh_confirmation_passed'] = False
    elif failure == 'terminal':
        trial['terminal']['passed'] = False
    else:
        plan = read_json(online / 'protocol.json')
        plan['cases'][0]['compilation_confirmation_seed'] = case['confirmation_seed']
        write_json(online / 'protocol.json', plan)
    write_json(online / 'summary.json', summary)
    output = tmp_path / 'not_started'
    with pytest.raises(ValueError, match='source confirmation|separately registered'):
        compilation.prepare_source_comparison_compilation(
            online, frozen, output, case_id=case['id'], arm='code_react')
    assert not output.exists()


def test_matched_compilation_shares_author_gates_and_paid_history_but_isolates_source(
        tmp_path, matched_compilation_inputs, monkeypatch):
    online, frozen, study, case = matched_compilation_inputs
    original = read_json(frozen / 'protocol.json')
    baseline = read_json(frozen / 'published_result.json')['method_source']
    plans, archives, gate_errors = [], [], []
    for arm in ('hegel', 'code_react'):
        output = tmp_path / f'formation_{arm}'
        plan = compilation.prepare_source_comparison_compilation(
            online, frozen, output, case_id=case['id'], arm=arm)
        plans.append(plan)
        assert plan['author_only'] and plan['transfer_cases'] == []
        assert 'published_revision' not in plan
        assert plan['author_limits']['max_model_calls'] == 32
        assert plan['author_limits']['token_budget'] == 12000000
        assert plan['author_limits']['execution_seconds'] == 36000.
        assert len(plan['practice_cases']) == len(original['practice_cases']) + 1
        new_case = plan['practice_cases']['design_case_012']
        assert new_case['confirmation_seed'] == case['compilation_confirmation_seed']
        assert new_case['prefix_seed'] == case['prefix_seed']
        assert 'evaluation_seed' not in new_case and 'compilation_confirmation_seed' not in new_case
        for name, old in original['practice_cases'].items():
            for key in ('observation_budget', 'compute_budget', 'execution_seconds'):
                assert min(plan['limits'][key], plan['practice_cases'][name]['limits'][key]) == min(
                    original['limits'][key], old.get('limits', {}).get(key, original['limits'][key]))

        def author_run(author, *, on_checkpoint):
            assert type(author) is compilation.SourceComparisonMethodLearner and author.arm == arm
            assert (author.workspace / METHOD_PATH).read_text() == baseline
            assert len(author.practice_trials) == 255
            assert author._current_source_coverage()['missing_cases'] == ['design_case_012']
            gate_errors.append(author._prior_selection_error(
                {'prior_reuse_cases': [], 'revised_cases': ['design_case_012']}))
            assert gate_errors[-1] is not None
            assert '32 program invocations' in author.common_interface
            assert 'all 13 listed practices' in author.common_interface
            assert 'CURRENT DELIVERABLE: use the retained paid comparisons' not in author.common_interface
            assert 'A trial receives 4096 observations including' not in author.common_interface
            linked = read_json(author.workspace / 'online_research_experience.json')
            assert linked['model'].endswith(f'{arm}_source.py')
            assert not (author.workspace / 'online_research_experience/evaluation.json').exists()
            other = 'code_react' if arm == 'hegel' else 'hegel'
            assert not (author.workspace / f'online_research_experience/{other}_source.py').exists()
            assert (author.workspace / 'trials/trial_000').resolve() == (
                study / 'author/trials/trial_000').resolve()
            payload = author.prepare_trial(plan['practice_cases']['design_case_000'],
                tmp_path / f'practice_{arm}', baseline, author.practice_limits)
            assert payload['agent'].arm == arm
            assert not (payload['agent'].workspace / 'online_research_experience').exists()
            assert not (payload['agent'].workspace / f'{arm}_source.py').exists()
            # Both arms preserve the actual edit -> paid-practice behavior;
            # this scripted practice is an interface check, not a science result.
            practices = []
            author._practice = lambda name: practices.append(name) or {'status': 'scripted'}
            feedback = author._execute('write', {'path': METHOD_PATH,
                'code': baseline + '\n# same scripted edit\n'}, author.interface)
            assert len(practices) == 1 and feedback['edit_practice']['status'] == 'scripted'
            return author.result('model_call_budget_exhausted')

        monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
        plan['runtime_snapshot'] = []
        result = compilation.run_compilation(plan, output, provider_factory=lambda _: None)
        assert not result['method_published'] and result['target_episodes_run'] == 0
        assert result['author_arm'] == arm and result['source_comparison']['source_changed']
        assert result['source_comparison']['outcome'] == 'method_unpublished'
        assert not (output / 'transfer_archive.json').exists()
        training = read_json(output / 'author/training.json')
        assert training['outgoing_knowledge']['files'][METHOD_PATH] == baseline + '\n# same scripted edit\n'
        # Code's ordinary result omits files; matched author checkpoints must
        # retain the draft required by the existing paid-practice restoration.
        restored_author = compilation.SourceComparisonMethodLearner(
            None, tmp_path / f'restored_{arm}', knowledge=archives[0] if archives else
            read_json(output / 'initial_archive.json'), cases=plan['practice_cases'],
            lab_factory=compilation.laboratory, config=InquiryLLMConfig(**plan['author_limits']),
            practice_limits=InquiryLLMConfig(**plan['limits']), prepare_trial=lambda *args: None)
        restored_author.arm = arm
        restored_author.restore_practice(training)
        assert (restored_author.workspace / METHOD_PATH).read_text() == baseline + '\n# same scripted edit\n'
        assert len(restored_author.practice_trials) == 255
        archives.append(read_json(output / 'initial_archive.json'))
    assert archives[0] == archives[1]
    assert gate_errors[0] == gate_errors[1]
    for key in ('model', 'provider', 'author_limits', 'limits', 'practice_cases'):
        assert plans[0][key] == plans[1][key]
    assert read_json(frozen / 'protocol.json') == original


@pytest.mark.parametrize('old_source_already_listed', [False, True])
def test_window_compilation_uses_fresh_allowance_without_raising_thirteen_old_case_caps(
        tmp_path, published_revision_inputs, old_source_already_listed):
    online, frozen, study, source_case = published_revision_inputs
    old = read_json(frozen / 'protocol.json')
    old_case = {key: deepcopy(value) for key, value in source_case.items()
                if key not in {'id', 'parameter_pair', 'evaluation_seed'}}
    old['practice_cases']['design_case_012'] = old_case
    # An existing lower cap must survive raising the shared practice allowance.
    old['practice_cases']['design_case_000']['limits'].update(
        observation_budget=180, compute_budget=12000, execution_seconds=90.)
    source_plan = read_json(study / 'protocol.json')
    source_plan['practice_cases'] = deepcopy(old['practice_cases'])
    write_json(study / 'protocol.json', source_plan)
    publication = read_json(frozen / 'published_result.json')
    candidates = read_json(study / 'author/method_candidates.json')
    retained = {trial['directory'] for candidate in candidates for trial in candidate['trials']}
    next_trial = max(int(Path(directory).name.removeprefix('trial_')) for directory in retained) + 1
    trial = {'case': 'design_case_012', 'directory': f'trials/trial_{next_trial:03d}',
        'status': 'completed', 'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96},
        'selected_prior_predictor': False}
    current, = [candidate for candidate in candidates if candidate['current']]
    current['trials'].append(trial)
    write_json(study / 'author/method_candidates.json', candidates)
    paid = study / 'author' / trial['directory']
    write_json(paid / 'result.json', {'paid_record': next_trial})
    (paid / METHOD_PATH).parent.mkdir(parents=True)
    (paid / METHOD_PATH).write_text(publication['method_source'])
    publication['practice_trials'].append({**trial, 'method_source': publication['method_source']})
    publication['retained_paid_practices'] = len(retained) + 1
    write_json(frozen / 'published_result.json', publication)
    old['frozen_publication']['confirmed_cases'].append('design_case_012')
    old['frozen_publication']['retained_paid_practices'] = len(retained) + 1
    previous_source = tmp_path / 'earlier_confirmed_nested_source/hegel'
    previous_costs = {'model_calls': 9, 'observation_cost': 216, 'compute_cost': 567,
        'python_execution_seconds': 12.5, 'reported_tokens': 12345}
    old.update(online_research_experience=str(previous_source),
        source_research_costs=previous_costs,
        source_cost_scope='Earlier nested-source research; retain its separately paid account.')
    previous_result = previous_source / 'training.json'
    write_json(previous_result, {'status': 'completed', 'costs': previous_costs})
    if old_source_already_listed:
        old['prior_author_phases'].append({'phase': 'online_source_research',
            'result': str(previous_result), 'costs': previous_costs,
            'cost_scope': old['source_cost_scope']})
    write_json(frozen / 'protocol.json', old)
    allowance = {**old['limits'], 'observation_budget': 4096,
        'compute_budget': 2000000, 'execution_seconds': 7200.}
    online_plan = read_json(online / 'protocol.json')
    online_plan['cases'][0]['window_width'] = 3
    online_plan.update(online_allowance=allowance,
        limits={**allowance, 'observation_budget': 6380,
                'compute_budget': 5000000, 'execution_seconds': 15000.})
    write_json(online / 'protocol.json', online_plan)
    before = {path: path.read_bytes() for path in (
        frozen / 'protocol.json', frozen / 'published_result.json', study / 'protocol.json',
        study / 'author/method_candidates.json',
        study / 'author' / current['source_path'], online / 'protocol.json', previous_result)}

    plan = compilation.prepare_revision_compilation(online, frozen, tmp_path / 'window_author')

    assert len(plan['practice_cases']) == 14
    assert '13' in plan['scope'] and '14' in plan['scope']
    resources = ('observation_budget', 'compute_budget', 'execution_seconds')
    assert {key: plan['limits'][key] for key in resources} == {
        key: max(old['limits'][key], allowance[key]) for key in resources}
    new_case = plan['practice_cases']['design_case_013']
    assert {key: new_case['limits'][key] for key in resources} == {
        key: allowance[key] for key in resources}
    assert new_case['limits']['max_model_calls'] == new_case['limits']['token_budget'] == 0
    prior_source, = [phase for phase in plan['prior_author_phases']
                      if phase['result'] == str(previous_result)]
    assert prior_source['costs'] == read_json(previous_result)['costs'] == previous_costs
    assert prior_source['cost_scope'] == old['source_cost_scope']
    assert plan['source_research_costs'] == read_json(online / 'summary.json')['trials'][0]['costs']
    knowledge = read_json(Path(plan['source']) / 'initial_archive.json')
    prepare = compilation.trial_preparer(knowledge, [], frozen / 'reference_archive',
        agent_type=ExperimentalDesignMethodAgent)
    remaining = replace(InquiryLLMConfig(**plan['limits']), max_model_calls=0, token_budget=0)
    # Exercise the real min(shared, per-case) preparation; no fit or method runs.
    for name, case in plan['practice_cases'].items():
        payload = prepare(case, tmp_path / 'prefixes' / name,
                          publication['method_source'], remaining)
        agent = payload['agent']
        expected = (allowance if name == 'design_case_013' else {
            key: min(old['limits'][key], old['practice_cases'][name].get('limits', {}).get(
                key, old['limits'][key])) for key in resources})
        assert {key: getattr(agent.config, key) for key in resources} == {
            key: expected[key] for key in resources}
        assert len(agent.rows) == agent.costs['observation_cost'] == 72
        assert agent.costs['model_calls'] == agent.config.max_model_calls == agent.config.token_budget == 0
        assert not agent.method_results and not agent.confirmations
    assert {path: path.read_bytes() for path in before} == before


def test_unpublished_structural_continuation_keeps_latest_edits_and_published_baseline(
        tmp_path, published_revision_inputs, monkeypatch):
    online, frozen, original_study, _ = published_revision_inputs
    phase = tmp_path / 'bounded_unpublished_author'
    plan = compilation.prepare_revision_compilation(online, frozen, phase)
    previous = read_json(phase / 'author_resume.json')
    baseline = previous['method_source']
    current = baseline + '# current unpublished author edits\n'
    workspace = phase / 'author'
    workspace.mkdir()
    for trial in previous['practice_trials']:
        target = workspace / trial['directory']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(original_study / 'author' / trial['directory'], target_is_directory=True)
    for number in range(255, 261):
        directory = f'trials/trial_{number:03d}'
        source = baseline if number == 255 else current
        write_json(workspace / directory / 'result.json', {})
        path = workspace / directory / METHOD_PATH
        path.parent.mkdir(parents=True)
        path.write_text(source)
        previous['practice_trials'].append({'case': 'design_case_012', 'directory': directory,
            'method_source': source, 'status': 'completed' if number == 260 else 'confirmation_failed',
            'fresh_confirmation': {'passed': number == 260, 'mse': 1e-15 if number == 260 else 0.1, 'rows': 96},
            'selected_prior_predictor': False,
            'execution_summary': {'research_steps': [], 'submission_count': 0}})
    previous.update(workspace=str(workspace), status='model_call_budget_exhausted',
        method_source=None, method_entry=None,
        outgoing_knowledge={'files': {METHOD_PATH: current}},
        author_research_notes='Latest author notes and observed failures remain available.',
        costs={'model_calls': 32, 'observation_cost': 1840, 'compute_cost': 242752})
    write_json(workspace / 'training.json', previous)
    write_json(phase / 'protocol.json', plan)
    usage = {'known_tokens': 6193644}
    write_json(phase / 'summary.json', {'method_published': False, 'author_usage': usage})
    continued = compilation.continuation_protocol(workspace / 'training.json')
    continued['runtime_snapshot'] = []
    assert continued['prior_author_phases'][-1]['costs'] == previous['costs']
    assert continued['prior_author_phases'][-1]['usage'] == usage
    assert len(continued['practice_cases']) == 13 and continued['author_only']

    def author_run(author, *, on_checkpoint):
        assert (author.workspace / METHOD_PATH).read_text() == current
        assert author.files['research_notes.md'] == previous['author_research_notes']
        assert len(author.practice_trials) == 261
        assert author.knowledge['files'][METHOD_PATH] == baseline
        entry, = [item for item in author.knowledge['manifest']['research_programs']
                  if item['path'] == METHOD_PATH]
        assert len(entry['prior_execution']['cases']) == 12
        author._request(author.interface)
        decision = read_json(author.workspace / 'method_decision.json')
        supported, = [item for item in decision['candidates'] if item['trial'] == 'trials/trial_243']
        assert len(supported['supported_cases']) == 12
        assert supported['failed_cases'] == ['design_case_012']
        assert supported['restore_action'] == {'action': 'run', 'arguments': {
            'operator': 'restore_method', 'trial': 'trials/trial_243'}}
        assert decision['current_source_coverage']['confirmed_cases'] == ['design_case_012']
        assert not any('online_research_experience' in path for path in author.knowledge['files'])
        for trial in author.practice_trials:
            assert (author.workspace / trial['directory']).resolve() == (workspace / trial['directory']).resolve()
        return author.result('model_call_budget_exhausted')

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    result = compilation.run_compilation(continued, tmp_path / 'continued_author', provider_factory=lambda _: None)
    assert result['method_published'] is False and result['target_episodes_run'] == 0
    assert read_json(workspace / 'training.json') == previous
    assert read_json(frozen / 'published_result.json')['method_source'] == baseline


def test_frozen_publication_and_accepted_baseline_finish_research_before_terminal_labels(
        tmp_path, published_compilation, monkeypatch):
    baseline, output = tmp_path / 'accepted_baseline', tmp_path / 'paired_targets'
    source = SOURCE + '# retained 11/12 source\n'
    acceptance = {'basis': 'user_accepted_development_draft', 'practice_cases': [],
        'retained_paid_practices': 236, 'author_usage': {'known_tokens': 123},
        'author_publication_passed': False}
    write_json(baseline / 'protocol.json', {'accepted_draft': acceptance})
    write_json(baseline / 'accepted_draft.json', {'status': 'user_accepted_draft',
        'method_source': source, 'costs': {'model_calls': 32}})
    plan, learned = compilation.prepare_published_targets(
        published_compilation, output, baseline_study=baseline)
    original = read_json(Path(plan['source']) / 'initial_archive.json')
    before = deepcopy(original)
    baseline_before = read_json(baseline / 'accepted_draft.json')
    order = []

    def frozen(part, branch, memory, sources, method_source, *, agent_type):
        assert agent_type is ExperimentalDesignMethodAgent
        assert memory['files'][METHOD_PATH] == method_source
        assert not any('trials/' in path for path in memory['files'])
        entry, = [item for item in memory['manifest']['research_programs']
                   if item['path'] == METHOD_PATH]
        if part.get('accepted_draft'):
            assert order == ['published_research']
            assert method_source == source
            assert len(part['transfer_cases']) == 8
            assert part['transfer_cases'] == plan['transfer_cases'][10:]
            assert part['limits'] == plan['limits']
            assert entry['admission'] == acceptance['basis']
            order.append('baseline_research')
        else:
            assert order == [] and len(part['transfer_cases']) == 18
            assert len(entry['prior_execution']['cases']) == 12
            order.append('published_research')
        return [(case, {'status': 'completed'}) for case in part['transfer_cases']]

    def score(part, results):
        assert order[:2] == ['published_research', 'baseline_research']
        order.append('terminal_scoring')
        return [{'case': case['id'], 'terminal': {'passed': True}}
                for case, _ in results]

    original_read = compilation.read_json
    def no_author_history(path):
        assert Path(path).name != 'training.json'
        return original_read(path)

    monkeypatch.setattr(compilation, 'read_json', no_author_history)
    monkeypatch.setattr(compilation, 'frozen_transfers', frozen)
    monkeypatch.setattr(compilation, 'score_frozen', score)
    monkeypatch.setattr(compilation, 'evaluate_python_model', lambda *a, **k: {'returncode': 0, 'mse': 9.})
    result = compilation.run_targets(plan, output, learned, original, [])
    assert len(result['retention']) == 10 and len(result['new_parameters']) == 8
    assert len(result['paired_baseline']['new_parameters']) == 8
    assert result['restored_practice_trials'] == 13 and result['new_practice_trials'] == 0
    assert result['method_published'] is True and result['live_consumer_model_calls'] == 0
    assert result['author_usage'] == learned['author_usage']
    assert original == before
    assert read_json(baseline / 'accepted_draft.json') == baseline_before
    assert (published_compilation / 'author/training.json').read_text().startswith('not a JSON')


def test_restore_accepted_source_keeps_later_failed_practices(
        tmp_path, confirmed_compilation_source, monkeypatch):
    plan = compilation.protocol(confirmed_compilation_source)
    plan['runtime_snapshot'] = []
    previous = read_json(Path(plan['continued_author']))
    accepted_source = previous['method_source']
    failed_source = accepted_source + '# later exhausted revision\n'
    failed = deepcopy(previous['practice_trials'][-1])
    failed.update(directory='trials/trial_999', method_source=failed_source,
                  status='method_step_limit', fresh_confirmation=None)
    previous['practice_trials'].append(failed)
    previous.update(status='model_call_budget_exhausted', method_source=None)
    previous['outgoing_knowledge']['files'][METHOD_PATH] = failed_source
    write_json(Path(previous['workspace']) / failed['directory'] / 'result.json', failed)
    resume = tmp_path / 'retained_author.json'
    write_json(resume, previous)
    plan['continued_author'] = str(resume)
    accepted = tmp_path / 'accepted'
    write_json(accepted / 'accepted_draft.json', {
        'status': 'user_accepted_draft', 'method_source': accepted_source})
    write_json(accepted / 'protocol.json', {'accepted_draft': {
        'source_author': str(confirmed_compilation_source / 'author/training.json'),
        'confirmed_cases': list(plan['practice_cases'])[:-1],
        'unconfirmed_cases': [list(plan['practice_cases'])[-1]]}})
    operator_practice = tmp_path / 'operator_practice.json'
    tool_evidence = {'origin': 'host_zero_llm', 'paid_rows': 72, 'model_calls': 0,
                     'operator_result': {'status': 'candidates_require_evaluation'}}
    write_json(operator_practice, tool_evidence)
    plan.update(restore_accepted=str(accepted), author_only=True,
                operator_practice=str(operator_practice))

    def author_run(author, *, on_checkpoint):
        assert (author.workspace / METHOD_PATH).read_text() == accepted_source
        assert author.practice_trials == previous['practice_trials']
        assert author.practice_trials[-1]['method_source'] == failed_source
        assert (author.workspace / failed['directory']).resolve() == (
            Path(previous['workspace']) / failed['directory']).resolve()
        evidence = read_json(author.workspace / 'restored_method.json')
        assert evidence['accepted_study'] == str(accepted)
        assert evidence['retained_paid_practices'] == len(previous['practice_trials'])
        context = read_json(author.workspace / 'resume_context.json')
        assert context['current_source_coverage'] == author._current_source_coverage()
        assert 'restored_method.json' in author.common_interface
        assert read_json(author.workspace / 'native_operator_practice.json') == tool_evidence
        assert 'host-executed' in author.common_interface
        assert author.model is None and author.published_source is None
        return author.result('model_call_budget_exhausted')

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    result = compilation.run_compilation(plan, tmp_path / 'restored', provider_factory=lambda _: None)
    assert result['method_published'] is False and result['target_episodes_run'] == 0
    assert read_json(resume) == previous
    assert read_json(accepted / 'accepted_draft.json')['method_source'] == accepted_source


def test_published_repair_preserves_old_cases_and_registers_new_development_and_targets(
        tmp_path, confirmed_compilation_source):
    plan = compilation.protocol(confirmed_compilation_source)
    study = tmp_path / 'published'
    previous = read_json(Path(plan['continued_author']))
    previous.update(status='completed', method_source=SOURCE)
    write_json(study / 'author/training.json', previous)
    write_json(study / 'protocol.json', plan)
    outcomes = [{'case': case['id'], 'costs': {'observation_cost': 120}}
                for case in plan['transfer_cases']]
    summary = {'status': 'completed', 'method_published': True, 'author_usage': {},
        'retention': outcomes[:10], 'new_parameters': outcomes[10:]}
    write_json(study / 'summary.json', summary)

    repaired = compilation.repair_protocol(study)
    assert repaired['source'] == plan['source']
    assert repaired['continued_author'] == str(study / 'author/training.json')
    assert {name: repaired['practice_cases'][name] for name in plan['practice_cases']} == plan['practice_cases']
    assert len(repaired['practice_cases']) == 12
    assert len(repaired['repair_development_cases']) == 4
    for item, prior in zip(repaired['repair_development_cases'], plan['transfer_cases'][10:]):
        new = repaired['practice_cases'][item['practice_case']]
        assert item['practice_case'] not in plan['practice_cases']
        assert item['prior_case'] == prior['id']
        assert new['apparatus_seed'] == prior['apparatus_seed']
        assert new['prefix_seed'] != prior['prefix_seed']
        assert new['confirmation_seed'] != prior['confirmation_seed']
    assert repaired['transfer_cases'][:10] == plan['transfer_cases'][:10]
    development_seeds = {case['apparatus_seed'] for case in repaired['practice_cases'].values()}
    assert not development_seeds & {case['apparatus_seed'] for case in repaired['transfer_cases'][10:]}
    assert repaired['prior_author_phases'][-1]['costs'] == previous['costs']
    assert repaired['prior_transfer_phases'][-1]['cases'] == outcomes
    assert read_json(study / 'summary.json') == summary
    assert read_json(study / 'protocol.json') == plan


def test_repair_feedback_uses_paid_proposals_without_terminal_labels(tmp_path):
    study, workspace = tmp_path / 'published', tmp_path / 'author'
    workspace.mkdir()
    source = study / 'frozen/conditional_target_004'
    paid = {'status': 'confirmation_failed', 'costs': {'observation_cost': 216},
        'method_source': SOURCE, 'paid_observations': [], 'events': [],
        'method_results': [{'step': 0, 'path': METHOD_PATH, 'proposal': {'action': 'run', 'arguments': {
            'operator': 'native_condition_research'}}, 'feedback': {
            'native_condition_research': {'status': 'candidates_require_evaluation',
                'candidates': [], 'proposed_experiments': [{
                    'purpose': 'conditional_boundary', 'predicted_separation': .718,
                    'observation_cost': 1, 'query_action': {'arguments': {
                        'actions': [], 'after_observation': [{'operator': 'native_condition_research'}]}}}]}}}]}
    write_json(source / 'training.json', paid)
    write_json(source / 'evaluation.json', {'secret_terminal_marker': 987654})
    author = SimpleNamespace(workspace=workspace, files={}, common_interface='existing task',
                             knowledge={'files': {'original.py': SOURCE}})
    original = deepcopy(author.knowledge)
    compilation.attach_transfer_repair_experience(author, {
        'repair_study': str(study), 'repair_development_cases': [{
            'practice_case': 'design_case_011', 'prior_case': 'conditional_target_004'}]})
    evidence = json.loads(author.files['repair_evidence.json'])
    assert evidence['records'][0]['execution_summary']['latest_native_result']['status'] == 'candidates_require_evaluation'
    assert 'conditional_boundary' in author.files['repair_evidence.json']
    assert '.718' in author.files['repair_evidence.json']
    assert author.knowledge == original
    linked = workspace / 'repair_evidence/design_case_011/result.json'
    assert linked.resolve() == source / 'training.json'
    assert not (linked.parent / 'evaluation.json').exists()
    assert all('secret_terminal_marker' not in path.read_text()
               for path in workspace.rglob('*.json'))


@pytest.fixture
def completed_published_comparison(tmp_path, published_revision_inputs):
    online, frozen, study, source_case = published_revision_inputs
    plan = read_json(study / 'protocol.json')
    plan['practice_cases'].update({f'design_case_{index:03d}': {
        key: deepcopy(value) for key, value in source_case.items()
        if key not in {'id', 'parameter_pair', 'evaluation_seed'}} for index in (12, 13)})
    source = SOURCE + '# published fourteen-case method\n'
    previous_source = SOURCE + '# published thirteen-case method\n'
    cases = list(plan['practice_cases'])
    drafts = []
    for index in range(116):
        method = (source if index == 115 else previous_source if index == 114 else
                  SOURCE + f'# retained draft {index}\n')
        numbers = (range(284, 298) if index == 115 else range(268, 282) if index == 114 else
                   [282 + index - 112] if index >= 112 else
                   [number for number in range(268) if number % 112 == index])
        trials = []
        for number in numbers:
            case = (cases[(number - 284 + 13) % 14] if index == 115 else
                    cases[number - 268] if index == 114 else 'old_case')
            success = index == 115 or (index == 114 and number < 281)
            trial = {'case': case, 'directory': f'trials/trial_{number:03d}',
                'status': 'completed' if success else 'method_stopped',
                'fresh_confirmation': {'passed': True, 'rows': 96, 'mse': 0.} if success else None,
                'selected_prior_predictor': case == cases[0]}
            path = study / 'author' / trial['directory']
            paid = {**deepcopy(trial), 'costs': {'observation_cost': 808 if index == 115 else 552,
                'python_execution_seconds': 5.}, 'method_results': [], 'events': [],
                'paid_observations': [], 'model': None}
            write_json(path / 'result.json', paid)
            write_json(path / 'evaluation.json', {'secret_terminal_marker': 830177})
            (path / METHOD_PATH).parent.mkdir(parents=True, exist_ok=True)
            (path / METHOD_PATH).write_text(method)
            trials.append(trial)
        drafts.append({'current': index == 115,
            'source_path': f'{trials[0]["directory"]}/{METHOD_PATH}', 'trials': trials})
    write_json(study / 'author/method_candidates.json', drafts)
    write_json(study / 'author/method_decision.json', {})
    write_json(study / 'protocol.json', plan)
    write_json(study / 'published_method.json', {'published': True, 'source': source})
    summary = {'method_published': True, 'author_costs': {'model_calls': 8},
               'author_usage': {'known_tokens': 123}}
    write_json(study / 'summary.json', summary)
    publication = {**read_json(frozen / 'published_result.json'),
        'status': 'completed', 'method_source': source,
        'costs': summary['author_costs'], 'author_usage': summary['author_usage'],
        'full_result': str(study / 'author/training.json')}
    baseline = tmp_path / 'previous_frozen'
    write_json(baseline / 'published_result.json', {**publication, 'method_source': previous_source})
    retained, new = 'nested_basis_target_000', 'window_parameter_target_000'
    frozen_plan = {**deepcopy(plan), 'old_target_ids': [retained],
        'transfer_cases': [{'id': retained}, {'id': new}],
        'regression_groups': {'previously_successful_nested_eight': [retained]},
        'frozen_publication': {'source_study': str(study)},
        'paired_baseline': {'source_study': str(baseline), 'directory': 'baseline_13of13'}}
    write_json(frozen / 'protocol.json', frozen_plan)
    write_json(frozen / 'published_result.json', publication)
    write_json(frozen / 'summary.json', {'status': 'completed', 'method_published': True,
        'terminal': {'secret_terminal_marker': 830177}})
    for index, directory in enumerate((baseline / 'frozen' / retained, frozen / 'frozen' / retained,
                                      frozen / 'baseline_13of13/frozen' / new, frozen / 'frozen' / new)):
        failed = index == 2
        write_json(directory / 'training.json', {
            'status': 'method_stopped' if failed else 'completed',
            'costs': {'observation_cost': 2120 if failed else 808,
                      'python_execution_seconds': 3000. if failed else 5.},
            'fresh_confirmation': None if failed else {'passed': True, 'mse': 0., 'rows': 96},
            'model': None, 'method_results': [], 'events': [], 'paid_observations': []})
        write_json(directory / 'evaluation.json', {'secret_terminal_marker': 830177})
    return frozen, study, source, drafts


def test_published_revision_restores_all_drafts_without_changing_cases_or_scheduling_targets(
        tmp_path, completed_published_comparison, monkeypatch):
    frozen, study, source, drafts = completed_published_comparison
    output = tmp_path / 'selection_author'
    original = read_json(study / 'protocol.json')
    protected = {path: path.read_bytes() for path in (
        study / 'protocol.json', study / 'published_method.json',
        study / 'author/method_candidates.json', study / 'author/training.json',
        frozen / 'protocol.json', frozen / 'published_result.json', frozen / 'summary.json')}
    original_read = compilation.read_json

    def no_full_author_export(path):
        assert Path(path) != study / 'author/training.json'
        assert Path(path).name != 'evaluation.json'
        return original_read(path)

    monkeypatch.setattr(compilation, 'read_json', no_full_author_export)
    plan = compilation.prepare_published_revision(frozen, output)
    restored = read_json(output / 'author_resume.json')
    assert plan['practice_cases'] == original['practice_cases'] and len(plan['practice_cases']) == 14
    assert plan['limits'] == original['limits']
    assert plan['author_only'] and plan['transfer_cases'] == []
    assert plan['author_limits']['max_model_calls'] == 32
    assert restored['workspace'] == str(study / 'author')
    assert restored['method_source'] == source
    assert len(restored['practice_trials']) == 298
    assert len({trial['method_source'] for trial in restored['practice_trials']}) == 116
    assert plan['restored_publication']['retained_paid_practices'] == 298
    assert plan['restored_publication']['retained_drafts'] == 116
    assert set(plan['published_revision']['restored_paid_summaries']) == {
        f'trials/trial_{number}' for number in (280, 281, 284, 297)}
    assert len(plan['repair_development_cases']) == 4
    assert {path: path.read_bytes() for path in protected} == protected
    assert not (output / 'author').exists()
    assert not (output / 'transfer_archive.json').exists()


def test_published_revision_attaches_paid_comparison_and_unchanged_finish_is_retention(
        tmp_path, completed_published_comparison, monkeypatch):
    frozen, study, source, _ = completed_published_comparison
    output = tmp_path / 'selection_author'
    plan = compilation.prepare_published_revision(frozen, output)
    plan['runtime_snapshot'] = []

    def author_run(author, *, on_checkpoint):
        assert (author.workspace / METHOD_PATH).read_text() == source
        assert len(author.practice_trials) == 298
        assert not author._current_source_coverage()['missing_cases']
        for trial in author.practice_trials:
            assert (author.workspace / trial['directory']).resolve() == (
                study / 'author' / trial['directory']).resolve()
        evidence = read_json(author.workspace / 'repair_evidence.json')
        assert len(evidence['records']) == 4
        assert [row['costs']['observation_cost'] for row in evidence['records']] == [808, 808, 2120, 808]
        assert evidence['records'][2]['fresh_confirmation'] == {}
        for item in plan['repair_development_cases']:
            directory = author.workspace / 'repair_evidence' / item['practice_case']
            assert (directory / 'result.json').resolve() == Path(item['source_directory']) / 'training.json'
            assert not (directory / 'evaluation.json').exists()
        assert 'existing program, additional evidence or structural revision' in author.common_interface
        assert 'no fixed scientific route or lower observation threshold' in author.common_interface
        assert 'secret_terminal_marker' not in author.files['repair_evidence.json']
        assert not any(name.startswith('repair_evidence') for name in author.knowledge['files'])
        author.model, author.published_source = METHOD_PATH, source
        return author.result('completed')

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    monkeypatch.setattr(compilation, 'run_targets', lambda *args, **kwargs:
        pytest.fail('published revision must not start targets'))
    result = compilation.run_compilation(plan, output, provider_factory=lambda _: None)
    assert result['method_published'] and result['target_episodes_run'] == 0
    assert result['published_revision'] == {'frozen_study': str(frozen),
        'source_changed': False, 'outcome': 'unchanged_method_retained'}
    assert result['author_costs']['model_calls'] == result['author_costs']['observation_cost'] == 0


@pytest.fixture
def published_author_revision(tmp_path, completed_published_comparison):
    frozen, original, parent_source, drafts = completed_published_comparison
    study = tmp_path / 'published_selection_author'
    plan = compilation.prepare_published_revision(frozen, study)
    plan['complete_practices'] = True
    plan['prior_author_phases'].append({'phase': 'paid_completion', 'costs': {'model_calls': 3}})
    source = parent_source + '# published sampling revision\n'
    drafts = deepcopy(drafts)
    for draft in drafts:
        draft['current'] = False
        for trial in draft['trials']:
            path = study / 'author' / trial['directory']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(original / 'author' / trial['directory'], target_is_directory=True)
    trials = []
    for number, case in enumerate(plan['practice_cases'], 298):
        trial = {'case': case, 'directory': f'trials/trial_{number:03d}',
            'status': 'completed', 'fresh_confirmation': {'passed': True, 'rows': 96, 'mse': 0.},
            'selected_prior_predictor': number == 298,
            'costs': {'observation_cost': 808, 'compute_cost': 1234},
            'execution_summary': {'step_count': 4, 'action_counts': {'query': 2, 'evaluate': 1}}}
        path = study / 'author' / trial['directory']
        write_json(path / 'result.json', {'already_summarized_paid_record': number})
        (path / METHOD_PATH).parent.mkdir(parents=True, exist_ok=True)
        (path / METHOD_PATH).write_text(source)
        trials.append(trial)
    drafts.append({'current': True, 'source_path': f'{trials[0]["directory"]}/{METHOD_PATH}',
                   'trials': trials})
    write_json(study / 'protocol.json', plan)
    write_json(study / 'published_method.json', {'published': True, 'source': source})
    write_json(study / 'summary.json', {'status': 'author_development_completed',
        'method_published': True, 'author_costs': {'model_calls': 3},
        'author_usage': {'known_tokens': 1700}})
    write_json(study / 'author/method_candidates.json', drafts)
    write_json(study / 'author/method_decision.json', {})
    (study / 'author/research_notes.md').write_text('Retain all paid source comparisons.')
    (study / 'author/training.json').write_text('large author export must not be loaded')
    (study / 'author' / METHOD_PATH).parent.mkdir(parents=True, exist_ok=True)
    (study / 'author' / METHOD_PATH).write_text(source)
    return study, frozen, source, plan


def test_published_author_revision_keeps_latest_source_all_paid_trials_and_inherited_comparison(
        tmp_path, published_author_revision, monkeypatch):
    study, frozen, source, original = published_author_revision
    protected = {path: path.read_bytes() for path in (
        study / 'protocol.json', study / 'summary.json', study / 'published_method.json',
        study / 'author/method_candidates.json', study / 'author/training.json')}
    original_read = compilation.read_json

    def no_full_author_export(path):
        assert Path(path) != study / 'author/training.json'
        assert Path(path).name != 'evaluation.json'
        return original_read(path)

    monkeypatch.setattr(compilation, 'read_json', no_full_author_export)
    output = tmp_path / 'next_selection_author'
    plan = compilation.prepare_published_revision(study, output)
    restored = read_json(output / 'author_resume.json')
    assert restored['method_source'] == source
    assert len(restored['practice_trials']) == 312
    assert len({trial['method_source'] for trial in restored['practice_trials']}) == 117
    assert restored['author_research_notes'] == 'Retain all paid source comparisons.'
    assert plan['practice_cases'] == original['practice_cases']
    assert plan['limits'] == original['limits']
    assert plan['repair_development_cases'] == original['repair_development_cases']
    assert len(plan['repair_development_cases']) == 4
    assert plan['prior_transfer_phases'] == original['prior_transfer_phases']
    assert plan['prior_author_phases'][:-1] == original['prior_author_phases']
    assert plan['prior_author_phases'][-1]['result'] == str(study / 'author/training.json')
    assert plan['prior_author_phases'][-1]['costs'] == {'model_calls': 3}
    assert plan['published_revision']['frozen_study'] == str(frozen)
    assert plan['published_revision']['parent_author_publication'] == str(study)
    assert plan['restored_publication']['publication_study'] == str(study)
    assert plan['author_only'] and plan['transfer_cases'] == []
    assert 'complete_practices' not in plan
    assert {path: path.read_bytes() for path in protected} == protected
    assert not (output / 'author').exists()


@pytest.mark.parametrize('change_source', [False, True])
def test_published_author_revision_registers_latest_baseline_and_reports_actual_source_change(
        tmp_path, published_author_revision, monkeypatch, change_source):
    study, frozen, source, _ = published_author_revision
    output = tmp_path / 'next_selection_author'
    plan = compilation.prepare_published_revision(study, output)
    plan['runtime_snapshot'] = []

    def author_run(author, *, on_checkpoint):
        assert author.knowledge['files'][METHOD_PATH] == source
        assert (author.workspace / METHOD_PATH).read_text() == source
        assert len(author.practice_trials) == 312
        assert not author._current_source_coverage()['missing_cases']
        assert len(read_json(author.workspace / 'repair_evidence.json')['records']) == 4
        assert 'CURRENT DELIVERABLE: complete the restored source' not in author.common_interface
        selected = source + '# scripted new research choice\n' if change_source else source
        author.model, author.published_source = METHOD_PATH, selected
        return author.result('completed')

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    monkeypatch.setattr(compilation, 'run_targets', lambda *a, **k:
        pytest.fail('published author revision must not start targets'))
    result = compilation.run_compilation(plan, output, provider_factory=lambda _: None)
    assert result['method_published'] and result['target_episodes_run'] == 0
    assert result['published_revision'] == {'frozen_study': str(frozen),
        'source_changed': change_source,
        'outcome': 'revised_source_published' if change_source else 'unchanged_method_retained'}


@pytest.mark.parametrize('published', [False, True])
@pytest.mark.parametrize('author_only', [False, True])
def test_author_only_experience_publication_gate_and_terminal_order(
        tmp_path, confirmed_compilation_source, monkeypatch, published, author_only):
    plan = compilation.protocol(confirmed_compilation_source)
    plan['runtime_snapshot'] = []
    plan['author_only'] = author_only
    output = tmp_path / 'compilation'
    original = read_json(Path(plan['source']) / 'initial_archive.json')
    restored = read_json(Path(plan['continued_author']))
    original_result = deepcopy(restored)
    new_source = SOURCE + '# scripted compilation fixture\n'
    order = []
    returned = {}

    def author_run(author, *, on_checkpoint):
        order.append('author')
        assert author.costs['observation_cost'] == author.costs['model_calls'] == 0
        assert author.practice_trials == restored['practice_trials']
        assert NATIVE_CONDITION_INTERFACE in author.common_interface
        assert 'Fresh confirmation ends this practice' in author.common_interface
        assert 'Failed confirmation contributes paid counterexamples, visible on the next invocation.' not in author.common_interface
        assert "After a program's failed confirmation, continuation" not in author.common_interface
        public = json.loads(author.files['practice_interfaces.json'])
        assert set(public) == set(plan['practice_cases'])
        for name, interface in public.items():
            assert set(interface) == {'public_spec', 'limits'}
            assert not {'seed', 'stage', 'apparatus_seed', 'kind', 'threshold', 'width'} & interface['public_spec'].keys()
            assert interface['limits']['max_model_calls'] == interface['limits']['token_budget'] == 0
            assert interface['limits']['confirmation_mse_threshold'] == (
                1e-6 if plan['practice_cases'][name]['kind'] == 'conditional_window' else 1e-10)
        experience = read_json(author.workspace / 'online_research_experience.json')
        assert (author.workspace / experience['training']).resolve() == (
            Path(plan['online_research_experience']) / 'training.json')
        assert (author.workspace / experience['model']).is_file()
        assert 'online_only.py' not in author.knowledge['files']
        if published:
            author.model, author.published_source = METHOD_PATH, new_source
        result = author.result('completed' if published else 'model_call_budget_exhausted')
        returned.update(result)
        on_checkpoint(result)
        return result

    def frozen(part, branch, memory, sources, method_source, *, agent_type):
        assert published and order == ['author']
        assert agent_type is ExperimentalDesignMethodAgent
        assert part['transfer_cases'] == plan['transfer_cases']
        assert method_source == new_source
        expected = register_published_method(original, returned, max_steps=plan['method_steps'])
        assert memory == expected
        assert not any('online_research_experience' in path or 'online_only' in path
                       for path in memory['files'])
        assert read_json(branch / 'transfer_archive.json') == memory
        order.extend(case['id'] for case in part['transfer_cases'])
        order.append('all_research_stopped')
        return [(case, {'status': 'completed'}) for case in part['transfer_cases']]

    def score(part, frozen_results):
        assert order[-1] == 'all_research_stopped'
        assert len(frozen_results) == 14
        order.append('terminal')
        return [{'case': case['id'], 'terminal': {'passed': True}}
                for case, result in frozen_results]

    def source_answer_score(workspace, model, rows, **kwargs):
        assert published and order[-1] in {'terminal', 'source_answer'}
        assert Path(workspace) == Path(plan['online_research_experience'])
        assert model == plan['source_model'] and len(rows) == plan['evaluation']['rows']
        order.append('source_answer')
        return {'returncode': 0, 'mse': 1., 'predictions': [0.] * len(rows)}

    monkeypatch.setattr(AnonymousResearchMethodLearner, 'run', author_run)
    monkeypatch.setattr(compilation, 'frozen_transfers', frozen)
    monkeypatch.setattr(compilation, 'score_frozen', score)
    monkeypatch.setattr(compilation, 'evaluate_python_model', source_answer_score)
    result = compilation.run_compilation(plan, output, provider_factory=lambda _: None)
    assert result['method_published'] is published
    assert read_json(Path(plan['continued_author'])) == original_result
    saved = read_json(output / 'protocol.json')
    assert saved['prior_author_phases'] == plan['prior_author_phases']
    assert saved['source_research_costs'] == plan['source_research_costs']
    if not published or author_only:
        assert order == ['author']
        assert result['target_episodes_pending'] == (0 if author_only else 14)
        assert not (output / 'transfer_archive.json').exists()
        if author_only:
            assert result['target_episodes_run'] == 0
            assert result['status'] == ('author_development_completed' if published else 'method_unpublished')
    else:
        assert len(result['retention']) == 10 and len(result['new_parameters']) == 4
        assert len(result['frozen_source_answer']) == 4
        assert all(not row['terminal']['passed'] for row in result['frozen_source_answer'])
        assert all('predictions' not in row['terminal'] for row in result['frozen_source_answer'])
        assert result['restored_practice_trials'] == len(restored['practice_trials'])
        assert result['new_practice_trials'] == 0
        assert result['retained_original_results'] == read_json(
            Path(plan['original_summary']))['frozen']['revised']
        assert result['live_consumer_model_calls'] == 0
