"""A host-delegated method practice uses the original child and confirmation loop."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts import run_scientific_revision_comparison as runner
from test_common_archive_online_research import ScriptedProvider
from test_native_condition_inquiry import PaidConditionalLab, buy_fixture, prepared
from test_native_condition_research import random_sequence


class PracticeLab(PaidConditionalLab):
    def confirmation_actions(self, count, rng):
        return [{"kind": "trajectory", "channels": [],
                 "sequence": random_sequence(rng.randrange(1_000_000), count)}]


class PracticeProvider(ScriptedProvider):
    """Scripted tool choices test handoff mechanics, not autonomous selection."""

    def __init__(self):
        super().__init__([])
        self.model = None

    def next_action(self, request):
        turn = len(self.requests)
        if turn == 0:
            action = ("run", {"operator": "native_condition_research", "max_compute": 500_000,
                "matched_pairs": [{"reference_record_index": 0, "record_index": 1}]})
        elif turn == 1:
            models = request["events"][-1]["feedback"]["native_condition_research"]["exported_models"]
            best = min(models, key=lambda item: item["fit_mse"])
            self.model = best["model"]
            action = ("evaluate", {"model": self.model})
        else:
            assert turn in (2, 3), "The child must return to its parent, not reopen an older review"
            action = ("finish", {"model": self.model,
                "lesson": "Return the paid evaluation" if turn == 2 else "Confirm the selected model"})
        self.steps.append(action)
        return super().next_action(request)


def test_host_practice_resumes_last_failed_check_without_reopening_it_after_return(tmp_path, monkeypatch):
    old = prepared(tmp_path / "old")
    old.provider = ScriptedProvider([("finish", {"model": "memorized.py"})])
    old.config = replace(old.config, max_model_calls=1)
    lab = PracticeLab()
    buy_fixture(old, lab)
    # A predictor fitting the old observations but failing on unseen histories
    # reproduces a last-call failure with no remaining turn to create its review.
    memorized = {json.dumps([dict(row.features), list(row.history)], sort_keys=True): row.target
                  for row in old.rows}
    old._save("memorized.py", "import json\n"
        f"VALUES = {memorized!r}\n"
        "def predict(row):\n"
        "    return VALUES.get(json.dumps([row['features'], row.get('history', [])], sort_keys=True), 0.)\n")
    previous = old.run(lab)
    assert previous["fresh_confirmation"]["passed"] is False
    assert not any(q.get("origin") == "fresh_confirmation_failure" for q in old.questions)
    assert not old._inquiry_stack
    (old.workspace / "research_context.json").write_text(json.dumps(old._request(lab)))
    (old.workspace / "training.json").write_text(json.dumps(previous))
    (old.workspace / "initial_archive.json").write_text(json.dumps(old.knowledge))
    confirmation = old.confirmations[-1]
    initial = {"question": {"id": "scientific_revision_0", "parent_id": "task", "depends_on": [],
        "question": "Investigate the actual failed independent prediction.",
        "origin": "fresh_confirmation_failure",
        "confirmation_observation_indices": deepcopy(confirmation["observation_indices"]),
        "failed_confirmation": {"confirmation_index": 0, **deepcopy(confirmation)}},
        "assignment": "Host method practice: invoke native_condition_research on paid evidence, "
            "inspect and evaluate its candidates, then return a selected model or unresolved result.",
        "max_model_calls": 3}
    case = {"id": "case", "kind": "fixture", "confirmation_seed": 1709}
    limits = replace(old.config, max_model_calls=5)
    plan = {"resume_result": str(old.workspace / "training.json"),
        "resume_workspace": str(old.workspace), "source_study": str(old.workspace),
        "initial_archive": str(old.workspace / "initial_archive.json"),
        "initial_inquiry": initial, "comparison": "Host practice, not autonomous discovery",
        "next_stage": "Independent confirmation only",
        "online": {"runtime_snapshot": [], "arms": ["recursive"], "model": "gpt-6-luna",
                   "limits": asdict(limits), "cases": [case]}}
    provider = PracticeProvider()

    def run_source(part, output, *, knowledge, prepare_prefix, agent_factory, **kwargs):
        agent = agent_factory(provider, output / "case" / "recursive", arm="recursive",
            knowledge=knowledge, source_task=False, config=InquiryLLMConfig(**part["limits"]))
        fresh = PracticeLab()
        restored = prepare_prefix(agent, fresh, case)
        assert agent.costs == previous["costs"] and fresh.actions == []
        assert agent.events[:-1] == previous["events"]
        event = deepcopy(agent.events[-1])
        assert event["origin"] == event["usage"]["backend"] == "host_method_practice"
        assert event["usage"]["total_tokens"] == 0
        assert event["arguments"] == {"operator": "inquire", **initial}
        assert agent._inquiry_stack[-1]["parent_event_index"] == len(previous["events"])
        assert restored["prior_research_cost"] == previous["costs"]
        restoration = json.loads((agent.workspace / "restoration.json").read_text())
        assert restoration["host_method_practice"]["event_index"] == len(previous["events"])

        result = agent.run(fresh)

        assert result["events"][:len(previous["events"])] == previous["events"]
        assert len(provider.requests) == 4
        assert provider.requests[0]["llm_inquiry"]["assignment"]["question"] == initial["assignment"]
        for request in provider.requests[:3]:
            assert request["llm_inquiry"]["question_id"] == "scientific_revision_0"
            assert request["costs"]["observation_cost"] == previous["costs"]["observation_cost"]
        assert provider.requests[1]["llm_inquiry"]["question"]["status"] == "inconclusive"
        assert not provider.requests[3].get("llm_inquiry")
        assert provider.requests[3]["scientific_revision_review"]["phase"] == "returned_to_parent"
        assert provider.requests[3]["costs"]["observation_cost"] == previous["costs"]["observation_cost"]
        assert result["fresh_confirmation"]["passed"]
        assert len(agent.confirmations) == 2 and len(fresh.actions) == 1
        assert result["costs"]["observation_cost"] == previous["costs"]["observation_cost"] + 16
        assert result["costs"]["model_calls"] == previous["costs"]["model_calls"] + 4
        reviews = [q for q in result["online_questions"] if q.get("origin") == "fresh_confirmation_failure"]
        assert len(reviews) == 1 and len(reviews[0]["native_condition_research"]) == 1
        assert reviews[0]["parent_research"]["status"] == "independent_confirmation_passed"
        return {"status": "completed", "trials": []}

    monkeypatch.setattr(runner, "run_study", run_source)
    result = runner.run_comparison(plan, tmp_path / "resumed", provider_factory=lambda _: provider,
                                   previous=previous)
    assert result["status"] == "completed"


def _exhausted_paid_author(tmp_path):
    """A completed phase can overshoot its token/execution allowance."""
    workspace = tmp_path / 'old' / 'author'
    workspace.mkdir(parents=True)
    costs = dict(model_calls=4, observation_cost=7, compute_cost=21,
                 python_execution_seconds=95., reported_tokens=None)
    previous = dict(workspace=str(workspace), status='model_call_budget_exhausted',
        method_source=None, source_repair=None, costs=costs,
        events=[{'usage': {'total_tokens': 950}}, {'usage': {}}])
    earlier = {'result': 'retained/author/training.json', 'status': 'provider_failed',
        'costs': dict(model_calls=2, observation_cost=3, compute_cost=9,
                      python_execution_seconds=50., reported_tokens=200),
        'usage': {'known_tokens': 200, 'missing_usage_calls': 0}}
    limits = dict(max_model_calls=6, observation_budget=100, compute_budget=1000,
                  execution_seconds=100., token_budget=1000)
    plan = dict(study='luna_paid_source_procedure_repair', author_total_limits=limits,
        author_limits={**limits, 'execution_seconds': 50., 'token_budget': 800},
        prior_author_phases=[earlier], source_checkpoint='retained/scene/result.json')
    (workspace.parent / 'protocol.json').write_text(json.dumps(plan))
    saved = workspace / 'training.json'
    saved.write_text(json.dumps(previous))
    return saved, previous, plan


@pytest.mark.parametrize('replenish', [False, True])
def test_paid_author_continuation_keeps_spending_when_replenishing_exhausted_allowances(tmp_path, replenish):
    from scripts import run_native_method_compilation as compilation

    saved, previous, original = _exhausted_paid_author(tmp_path)
    original_bytes = saved.read_bytes()
    additions = (dict(additional_author_calls=3, additional_author_execution_seconds=300.,
                      additional_author_tokens=600) if replenish else {})

    plan = compilation.continuation_protocol(saved, **additions)

    assert saved.read_bytes() == original_bytes
    assert plan['prior_author_phases'][:-1] == original['prior_author_phases']
    assert plan['prior_author_phases'][-1]['costs'] == previous['costs']
    assert plan['prior_unknown_usage_calls'] == 1
    assert plan['author_limits']['observation_budget'] == 90
    assert plan['author_limits']['compute_budget'] == 970
    assert plan['source_checkpoint'] == original['source_checkpoint']
    assert plan['author_limits']['max_model_calls'] == (3 if replenish else 0)
    assert plan['author_limits']['execution_seconds'] == (255. if replenish else 0.)
    assert plan['author_limits']['token_budget'] == (450 if replenish else 0)
    assert plan['author_total_limits']['execution_seconds'] == (400. if replenish else 100.)
    assert plan['author_total_limits']['token_budget'] == (1600 if replenish else 1000)
    for name in ('additional_author_calls', 'additional_author_execution_seconds',
                 'additional_author_tokens'):
        assert plan[name] == additions.get(name, 0)

    if replenish:
        # A following continuation subtracts both old phases and this new phase;
        # explicit additions are not granted again by reading the saved plan.
        workspace = tmp_path / 'continued' / 'author'
        workspace.mkdir(parents=True)
        (workspace.parent / 'protocol.json').write_text(json.dumps(plan))
        current = dict(workspace=str(workspace), status='provider_failed',
            costs=dict(model_calls=1, observation_cost=0, compute_cost=0,
                       python_execution_seconds=20., reported_tokens=70),
            events=[{'usage': {'total_tokens': 70}}])
        next_saved = workspace / 'training.json'
        next_saved.write_text(json.dumps(current))
        again = compilation.continuation_protocol(next_saved)
        assert again['author_total_limits'] == plan['author_total_limits']
        assert again['author_limits']['max_model_calls'] == 2
        assert again['author_limits']['execution_seconds'] == 235.
        assert again['author_limits']['token_budget'] == 380
        assert again['prior_author_phases'][:-1] == plan['prior_author_phases']
        assert again['prior_unknown_usage_calls'] == 1
        assert all(again[name] == 0 for name in additions)


@pytest.mark.parametrize('option', ['additional_author_calls',
    'additional_author_execution_seconds', 'additional_author_tokens'])
def test_author_allowances_require_paid_continuation_and_nonnegative_values(tmp_path, monkeypatch, option):
    from scripts import run_native_method_compilation as compilation

    saved, _, _ = _exhausted_paid_author(tmp_path)
    with pytest.raises(ValueError, match='nonnegative'):
        compilation.continuation_protocol(saved, **{option: -1})
    protocol = saved.parent.parent / 'protocol.json'
    plan = json.loads(protocol.read_text())
    plan['study'] = 'luna_native_research_method_compilation'
    protocol.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='only to paid-source'):
        compilation.continuation_protocol(saved, **{option: 1})
    monkeypatch.setattr('sys.argv', ['run_native_method_compilation.py',
        '--repair-paid-source', str(saved), '--' + option.replace('_', '-'), '1'])
    with pytest.raises(SystemExit) as error:
        compilation.main()
    assert error.value.code == 2


def test_author_allowance_cli_records_explicit_additions_in_prepared_protocol(tmp_path, monkeypatch):
    from scripts import run_native_method_compilation as compilation

    saved, _, _ = _exhausted_paid_author(tmp_path)
    output = tmp_path / 'prepared'
    monkeypatch.setattr('sys.argv', ['run_native_method_compilation.py',
        '--continue-author', str(saved), '--output', str(output), '--prepare-only',
        '--additional-author-calls', '3', '--additional-author-execution-seconds', '300',
        '--additional-author-tokens', '600'])
    compilation.main()
    plan = json.loads((output / 'planned_protocol.json').read_text())
    assert plan['author_limits']['max_model_calls'] == 3
    assert plan['author_limits']['execution_seconds'] == 255.
    assert plan['author_limits']['token_budget'] == 450
    assert plan['additional_author_execution_seconds'] == 300.
    assert plan['additional_author_tokens'] == 600
