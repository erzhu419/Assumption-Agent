"""Exercise memory removal and explicit recall using actual paid author history."""

from copy import deepcopy
import json

from scripts import run_python_autonomous_recall_study as study


def original_history():
    return json.loads((study.ORIGIN / 'author_2/learning/result.json').read_text())


def make_author(path, history, provider=None, prepared=None):
    plan = json.loads((study.ORIGIN / 'protocol.json').read_text())
    sources = json.loads((study.ORIGIN / 'existing_source_outcomes.json').read_text())['sources']
    source = sources[plan['author_source_index']]
    memory = study.prior.archive_memory(source)
    memory['files'][study.prior.METHOD_PATH] = (study.ORIGIN / 'unchanged_method.py').read_text()
    prepare = study.prior.trial_preparer(source, plan)

    def prepare_trial(spec, directory, method, limits):
        if prepared is not None:
            prepared.append((deepcopy(spec), directory, method))
        return prepare(spec, directory, method, limits)

    limits = {**plan['author_limits'], 'max_model_calls': 8, 'token_budget': 350000,
        'observation_budget': 2400, 'compute_budget': 1500000, 'execution_seconds': 180.}
    author = study.AutonomousRecallAuthor(provider, path, knowledge=memory,
        cases=plan['author_replicas'][2]['practice_cases'], lab_factory=study.prior.laboratory,
        config=study.prior.InquiryLLMConfig(**limits), prepare_trial=prepare_trial,
        counterexample_case='case_5')
    study.prior.supply_counterexample(author, study.ORIGIN / 'paid_source_counterexample.json',
                                     case_label='case_5')
    author.restore_practice(history)
    author.starting_sources = study.source_support(history)
    return author


def test_memory_ablation_removes_supported_sources_and_leaves_the_next_trial_slot_free(tmp_path):
    history = original_history()
    original = deepcopy(history)
    support = study.source_support(history)
    removed = {source for source, evidence in support.items() if evidence['supported']}
    assert removed == {history['practice_trials'][3]['method_source']}
    ablated = study.remove_supported_history(history, tmp_path / 'ablated')
    kept = [trial for trial in history['practice_trials'] if trial['method_source'] not in removed]
    assert history == original and len(ablated['practice_trials']) == 4
    for index, (old, new) in enumerate(zip(kept, ablated['practice_trials'])):
        assert new['directory'] == f'trials/trial_{index:03d}'
        assert {key: value for key, value in new.items() if key != 'directory'} == {
            key: value for key, value in old.items() if key != 'directory'}
        assert (tmp_path / 'ablated' / new['directory'] / study.prior.METHOD_PATH).read_text() == old['method_source']
    assert not any(item['supported'] for item in study.source_support(ablated).values())

    author = make_author(tmp_path / 'author', ablated)
    before = {trial['directory']: (author.workspace / trial['directory'] / 'result.json').read_bytes()
              for trial in author.practice_trials}
    author._prepare_program_state()
    decision = json.loads(author.files['method_decision.json'])
    assert all(not entry['supported_cases'] for entry in decision['candidates'])
    assert author.files[study.prior.METHOD_PATH] == history['outgoing_knowledge']['files'][study.prior.METHOD_PATH]
    result = author._execute('run', {'operator': 'practice', 'case': 'case_0'}, author.interface)
    assert result['directory'] == 'trials/trial_004'
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert len(author.practice_trials) == 5 and author.costs['observation_cost'] > 0
    for directory, content in before.items():
        assert (author.workspace / directory / 'result.json').read_bytes() == content
    assert author.published_source is None and author.costs['model_calls'] == 0


def test_scripted_explicit_recall_requires_all_six_cases_without_host_fallback_or_targets(tmp_path, monkeypatch):
    history = original_history()
    supported = history['practice_trials'][3]
    assert supported['directory'] == 'trials/trial_003' and supported['status'] == 'completed'

    class ScriptedRecall:
        calls = 0

        def next_action(self, request):
            decision = json.loads(request['files']['method_decision.json'])
            if self.calls == 0:
                assert request['files'][study.prior.METHOD_PATH] != supported['method_source']
                entry = next(item for item in decision['candidates'] if item['trial'] == supported['directory'])
                assert entry['supported_cases'] == ['case_5'] and not entry['current']
                action = entry['restore_action']
            else:
                assert self.calls == 1
                assert request['files'][study.prior.METHOD_PATH] == supported['method_source']
                assert decision['current_source_coverage']['confirmed_cases'] == ['case_5']
                assert decision['current_source_coverage']['missing_cases'] == [f'case_{i}' for i in range(5)]
                action = {'action': 'finish', 'arguments': {'path': study.prior.METHOD_PATH,
                    'lesson': 'Scripted interface test of explicit method recall.'}}
            self.calls += 1
            return {**action, 'usage': {'backend': 'scripted_no_api', 'total_tokens': 1}}

    def forbid_terminal_scoring(*args, **kwargs):
        raise AssertionError('author practice must not be counted as a new target evaluation')

    monkeypatch.setattr(study.prior, 'score_episode', forbid_terminal_scoring)
    prepared, provider = [], ScriptedRecall()
    author = make_author(tmp_path / 'author', history, provider, prepared)
    assert author._incumbent_trial(author.files[study.prior.METHOD_PATH]) is None
    assert 'the host never restores a different source' in author.common_interface
    assert 'An earlier source passing this comparison' not in author.common_interface
    result = author.run()
    assert result['status'] == 'completed' and result['method_source'] == supported['method_source']
    assert provider.calls == result['costs']['model_calls'] == 2
    assert result['events'][0]['feedback']['restored_method'] == supported['directory']
    assert 'edit_practice' not in result['events'][0]['feedback']
    publication = result['events'][1]['feedback']
    assert 'incumbent_comparison' not in publication
    assert publication['current_source_coverage']['confirmed_cases'] == [f'case_{i}' for i in range(6)]
    assert publication['current_source_coverage']['scoped_revision']['distinct_cases'] is True
    paid = result['practice_trials'][len(history['practice_trials']):]
    assert [trial['case'] for trial in paid] == [f'case_{i}' for i in range(5)]
    assert all(trial['method_source'] == supported['method_source'] and trial['status'] == 'completed' for trial in paid)
    assert len(prepared) == 5 and all(directory.parent == author.workspace / 'trials'
                                     for _, directory, _ in prepared)
    assert author.choice_events[0]['action'] == 'run' and author.choice_events[0]['selected_supported_source']
    assert author.choice_events[1]['action'] == 'finish' and author.choice_events[1]['published']
    for key in ('observation_cost', 'measurement_cost', 'compute_cost', 'python_execution_seconds'):
        assert result['costs'][key] == sum(trial['costs'][key] for trial in paid)
