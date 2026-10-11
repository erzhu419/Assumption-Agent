"""Parallel training keeps paired evidence separate and terminal labels delayed."""

import json
from threading import Barrier

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_anonymous_revision_inquiry import protocol, run_study
from test_anonymous_revision_study import ScriptedProvider


def test_parallel_training_preserves_accounts_order_and_terminal_boundary(tmp_path):
    plan = protocol(kinds=("memoryless",), cases=2, config=InquiryLLMConfig(
        observation_budget=144, confirmation_rows=24, max_model_calls=6,
        execution_seconds=30., token_budget=10000))
    expected = [("case_000", "code_react"), ("case_000", "hegel"),
                ("case_001", "hegel"), ("case_001", "code_react")]
    training_paths = [tmp_path / case / arm / "training.json" for case, arm in expected]
    barrier, providers, terminal_calls = Barrier(2, timeout=10), {}, []

    class ConcurrentProvider(ScriptedProvider):
        def next_action(self, request):
            if not self.requests:
                # This cannot pass if a two-worker run silently trains serially.
                barrier.wait()
            return super().next_action(request)

    def provider(workspace):
        current = ConcurrentProvider()
        providers[(workspace.parent.name, workspace.name)] = current
        return current

    class GuardedLab(AnonymousHistoryLab):
        def evaluation_rows(self, *args, **kwargs):
            assert all(path.is_file() for path in training_paths)
            terminal_calls.append(self._seed)
            return super().evaluation_rows(*args, **kwargs)

    knowledge = {"files": {"shared/earlier.py": "# Same incoming archive\n"},
                 "lessons": ["Unconfirmed earlier investigation"], "manifest": {}}
    result = run_study(plan, tmp_path, provider_factory=provider, knowledge=knowledge,
        laboratory_factory=lambda case: GuardedLab(case["kind"], case["seed"]), workers=2)

    assert [(trial["case"], trial["arm"]) for trial in result["trials"]] == expected
    assert len(terminal_calls) == 4
    assert result["evidence_kind"] == "scripted_interface_check"
    assert all(trial["terminal"]["passed"] for trial in result["trials"])
    for case in plan["cases"]:
        left, right = [providers[(case["id"], arm)].requests[0] for arm in plan["arms"]]
        assert left["observations"] == right["observations"]
        assert left["costs"] == right["costs"]
        assert left["session_knowledge"] == right["session_knowledge"] == knowledge
        assert left["capabilities"] == right["capabilities"]
        assert left["observations"] is not right["observations"]
    for path in training_paths:
        training = json.loads(path.read_text())
        assert training["costs"]["observation_cost"] == 72 + 3 + 24
        assert training["costs"]["model_calls"] == 5
        assert training["workspace"] == str(path.parent)
