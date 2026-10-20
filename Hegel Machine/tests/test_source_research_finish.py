"""Source research termination and exact selected-work support, without an LLM."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_paid_scene_method_repair import make_repair, REFIT_METHOD


def test_inconclusive_finish_exits_after_one_call_and_can_restore_paid_work(tmp_path):
    author, checkpoint, _ = make_repair(tmp_path / 'old')
    author._practice('same_scene')
    author.research_work = {'selected_source': METHOD_PATH, 'method': METHOD_PATH,
        'question': {'id': 'remaining', 'question': 'Which evidence is missing?', 'depends_on': []}}
    author.config = replace(author.config, max_model_calls=3, token_budget=1000,
                            confirmation_rows=0)
    before = checkpoint.read_bytes()

    class Provider:
        calls = 0

        def next_action(self, request):
            self.calls += 1
            assert request['research_work'] == author.research_work
            return {'action': 'finish', 'arguments': {'inconclusive': True,
                    'question_id': 'source_method_repair',
                    'lesson': 'The selected fixture does not supply the needed evidence.',
                    'notes': 'Retain the unresolved evidence for the next explicit revision.'},
                    'usage': {'total_tokens': 5}}

    provider = author.provider = Provider()
    result = author.run()
    assert provider.calls == 1
    assert result['status'] == 'source_repair_inconclusive'
    assert result['source_repair']['latest_practice'] == 'trials/trial_000'
    assert result['model'] is None and result['method_source'] is None
    assert result['published'] is False
    assert result['author_research_notes'] == 'Retain the unresolved evidence for the next explicit revision.'
    assert author.costs['observation_cost'] == 0
    assert checkpoint.read_bytes() == before

    # The same public continuation entry must accept an explicit unresolved
    # return without discarding its accounting or automatically restarting it.
    from scripts.run_native_method_compilation import continuation_protocol
    saved = author.workspace / 'training.json'
    saved.write_text(json.dumps(result))
    (author.workspace.parent / 'protocol.json').write_text(json.dumps({
        'study': 'luna_paid_source_procedure_repair',
        'author_limits': asdict(author.config), 'limits': asdict(author.practice_limits)}))
    continued = continuation_protocol(saved)
    assert continued['continued_author'] == str(saved)
    assert continued['author_limits']['max_model_calls'] == 2
    assert continued['author_limits']['token_budget'] == 995
    assert continued['prior_author_phases'][-1]['status'] == 'source_repair_inconclusive'

    restored, _, _ = make_repair(tmp_path / 'new')
    restored.restore_practice(result)
    assert restored.research_work == author.research_work
    assert restored.research_root['assignment'] == author.research_root['assignment']
    assert restored.research_root['status'] == 'open'
    assert restored._terminal_status() is None
    assert restored.source_repair is None
    assert (restored.workspace / 'trials/trial_000').is_symlink()
    assert restored.costs['model_calls'] == 0


def test_changed_work_cannot_inherit_old_source_confirmation(tmp_path):
    author, checkpoint, _ = make_repair(tmp_path)
    selected = 'shared/chosen_program.py'
    (author.workspace / selected).write_text(REFIT_METHOD)
    work = {'id': 'research', 'question': 'Refit the existing candidate.', 'depends_on': []}
    result = author._execute('run', {'operator': 'procedure', 'path': selected,
                                     'question': work, 'notes': 'Selected fixture program.'}, author.interface)
    assert result['status'] == 'completed'
    assert len(author.practice_trials) == 1  # Adoption does not run a second edit practice.
    assert author._current_source_coverage()['confirmed_cases'] == ['same_scene']
    assert (author.workspace / selected).read_text() == REFIT_METHOD
    assert (author.workspace / METHOD_PATH).read_text() == REFIT_METHOD
    assert result['research_work_result']['confirmation']['passed'] is True
    assert author.practice_trials[-1]['research_work'] == author.research_work
    assert author.files['research_notes.md'] == 'Selected fixture program.'

    # A changed assignment that cannot start is not supported by the old trial.
    author.config = replace(author.config, observation_budget=author.costs['observation_cost'])
    revised = deepcopy(work)
    revised['question'] = 'A different research assignment.'
    denied = author._execute('run', {'operator': 'procedure', 'path': selected,
                                     'question': revised}, author.interface)
    assert 'error' in denied
    assert len(author.practice_trials) == 1
    assert author._current_source_coverage()['confirmed_cases'] == []
    assert 'error' in author._execute('finish', {'path': METHOD_PATH}, author.interface)
    assert author.model is None


def test_invalid_selected_program_returns_repairable_error_before_adoption(tmp_path):
    author, _, _ = make_repair(tmp_path)
    before = (author.workspace / METHOD_PATH).read_text()
    (author.workspace / 'broken.py').write_text('def select_actions(\n')
    with pytest.raises(ValueError, match='selected procedure has invalid Python'):
        author._execute('run', {'operator': 'procedure', 'path': 'broken.py'}, author.interface)
    assert (author.workspace / METHOD_PATH).read_text() == before
    assert author.research_work is None and not author.practice_trials
