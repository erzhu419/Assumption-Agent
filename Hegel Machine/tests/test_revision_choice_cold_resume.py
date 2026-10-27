"""Resume a cold-moved checkpoint through its actual warm workspace link."""

from copy import deepcopy
import json

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts import run_anonymous_revision_choice as study
from test_anonymous_revision_choice import frozen_source, structural_case


def test_cold_checkpoint_link_keeps_the_original_online_context_and_models(tmp_path, monkeypatch):
    source, workspace = tmp_path / 'source', tmp_path / 'warm'
    memory, case = frozen_source(source), structural_case()
    prior = CommonArchiveInquiryAgent(None, workspace, arm='hegel', knowledge=memory,
        source_task=False, confirmation_seed=case['confirmation_seed'],
        config=InquiryLLMConfig(observation_budget=576, confirmation_rows=96,
            execution_seconds=30., compute_budget=20000))
    lab = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
    actions = [{'kind': 'trajectory', 'sequence': [{'u': .2}] * 8, 'channels': []}]
    feedback = prior._execute('query', {'actions': actions}, lab)
    prior.events.append({'step': 0, 'action': 'query', 'arguments': {'actions': actions},
        'feedback': feedback, 'usage': {'backend': 'scripted_test', 'total_tokens': 1}})
    prior.costs.update(model_calls=1, reported_tokens=1)
    code = 'def predict(row): return row["features"]["u"]\n'
    prior._save('candidate.py', code)
    prior._save('research_notes.md', 'The paid warm investigation remains unresolved.')
    (workspace / 'research_context.json').write_text(json.dumps(prior._request(lab)))
    previous = prior.result('model_call_budget_exhausted')
    cold = tmp_path / 'cold'
    cold.mkdir()
    payload = json.dumps(previous)
    (cold / 'training.json').write_text(payload)
    previous = json.loads(payload)
    checkpoint = workspace / 'training.json'
    checkpoint.symlink_to(cold / 'training.json')
    restored_from, requests = [], []
    restore_research = CommonArchiveInquiryAgent.restore_research

    def restore(agent, previous_result, old_workspace, apparatus, **kwargs):
        restored_from.append(old_workspace)
        return restore_research(agent, previous_result, old_workspace, apparatus, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError('cold directory must not replace the warm online context')

    class InspectProvider:
        def next_action(self, request):
            requests.append(deepcopy(request))
            return {'action': 'provider_error', 'error': {'fixture': 'Stop after restore'},
                'usage': {'backend': 'scripted_test', 'total_tokens': 1}}

    def laboratory(selected):
        assert selected == case
        restored = AnonymousHistoryLab(kind='memoryless', seed=case['seed'])
        restored.execute = unexpected  # No paid observations may be bought again.
        return restored

    monkeypatch.setattr(study, 'laboratory', laboratory)
    monkeypatch.setattr(CommonArchiveInquiryAgent, 'restore_research', restore)
    monkeypatch.setattr(CommonArchiveInquiryAgent, 'restore_practice_research', unexpected)
    output = tmp_path / 'continued'
    summary = study.run_choice(source, output, cases=[case], resume_result=checkpoint,
        provider_factory=lambda _: InspectProvider())
    plan = json.loads((output / 'protocol.json').read_text())
    assert plan['resume_mode'] == 'online_research'
    assert plan['resume_result'] == str(checkpoint.absolute())
    assert restored_from == [workspace]
    assert requests[0]['files']['candidate.py'] == code
    assert requests[0]['research_notes'] == 'The paid warm investigation remains unresolved.'
    assert requests[0]['laboratory_observations'] == previous['paid_observations']
    trial, = summary['trials']
    assert trial['prior_research_cost'] == previous['costs']
    assert trial['continuation_cost']['observation_cost'] == 0
    assert trial['continuation_cost']['compute_cost'] == 0
    assert trial['continuation_cost']['model_calls'] == 1
    assert checkpoint.is_symlink() and (cold / 'training.json').read_text() == payload
    assert (workspace / 'candidate.py').read_text() == code
    assert not (cold / 'research_context.json').exists()
