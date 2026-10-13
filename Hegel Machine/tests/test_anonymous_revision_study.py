"""Check the shared study wiring; scripted actions are not discovery evidence."""

import json
from pathlib import Path

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_anonymous_revision_inquiry import protocol, run_study


MODEL = '''import numpy as np
coefficients = None
def fit(rows):
    global coefficients
    x = [[1., row['features']['u']] for row in rows]
    coefficients = np.linalg.lstsq(x, [row['target'] for row in rows], rcond=None)[0]
def predict(row):
    return float(coefficients @ [1., row['features']['u']])
'''

PLAN = '''def select_actions(spec, observations, memory):
    return {'actions':[{'kind':'trajectory','sequence':[{'u':-1.},{'u':0.},{'u':1.}]}],
            'question':{'id':'q1','question':'Does the fitted candidate predict another path?',
                        'parent_id':None,'depends_on':[]},
            'revision':{'id':'r1','affected_object':'predictor','models':['shared/model.py']}}
'''


class ScriptedProvider:
    def __init__(self):
        manifest = {"hypotheses": [{"id": "h1", "path": "shared/model.py"}],
                    "research_programs": [{"id": "q1", "path": "shared/plan.py"}],
                    "support_questions": []}
        self.actions = iter([
            ("write", {"path": "shared/model.py", "code": MODEL}),
            ("write", {"path": "shared/plan.py", "code": PLAN}),
            ("query", {"path": "shared/plan.py"}),
            ("write", {"path": "shared/knowledge.json", "code": json.dumps(manifest)}),
            ("finish", {"model": "shared/model.py"}),
        ])
        self.requests = []

    def next_action(self, request):
        self.requests.append(request)
        action, arguments = next(self.actions)
        return {"action": action, "arguments": arguments,
                "usage": {"backend": "scripted_interface_test", "total_tokens": 0}}


def test_common_archive_prefix_revision_and_unseen_prediction(tmp_path, monkeypatch):
    from scripts import run_anonymous_revision_inquiry as study
    original = study.CommonArchiveInquiryAgent
    source_flags = []
    def make_agent(*args, **kwargs):
        source_flags.append(kwargs["source_task"])
        return original(*args, **kwargs)
    monkeypatch.setattr(study, "CommonArchiveInquiryAgent", make_agent)
    providers = []
    def provider(workspace):
        result = ScriptedProvider()
        providers.append(result)
        return result
    plan = protocol(kinds=("memoryless",), config=InquiryLLMConfig(
        observation_budget=144, confirmation_rows=24, max_model_calls=6,
        execution_seconds=30., token_budget=10000))
    knowledge = {"files": {"shared/earlier.py": "# equal archived code\n"},
                 "lessons": ["An earlier unconfirmed proposal."], "manifest": {}}
    result = run_study(plan, tmp_path, provider_factory=provider, knowledge=knowledge)
    assert len(result["trials"]) == 2
    for trial in result["trials"]:
        assert trial["status"] == "completed"
        assert trial["revision_count"] == 1
        assert trial["terminal"]["passed"]
        assert trial["costs"]["observation_cost"] == 72 + 3 + 24
        assert "shared/model.py" in json.loads(Path(trial["archive"]).read_text())["files"]
    left, right = [item.requests[0] for item in providers]
    assert left["observations"] == right["observations"]
    assert left["session_knowledge"] == right["session_knowledge"] == knowledge
    assert left["capabilities"] == right["capabilities"]
    assert "memoryless" not in json.dumps(left)
    assert source_flags == [False, False]  # Existing archive makes these consumer tasks.


def test_study_uses_the_same_supplied_laboratory_factory_for_training_and_terminal(tmp_path):
    from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
    calls = []
    def factory(case):
        calls.append(case['kind'])
        return AnonymousHistoryLab('memoryless', case['seed'])
    plan = protocol(kinds=('factory_defined_physics',), config=InquiryLLMConfig(
        observation_budget=144, confirmation_rows=24, max_model_calls=6,
        execution_seconds=30., token_budget=10000))
    result = run_study(plan, tmp_path, provider_factory=lambda _: ScriptedProvider(),
                       laboratory_factory=factory)
    assert calls == ['factory_defined_physics'] * 4
    assert all(trial['fresh_confirmation_passed'] and trial['terminal']['passed']
               for trial in result['trials'])


def test_evidence_kind_uses_new_provider_events_after_restored_frozen_history(tmp_path):
    # Only metadata routing is under test: this fixture makes no external call.
    from random import Random

    class ProviderMetadata(ScriptedProvider):
        def next_action(self, request):
            response = super().next_action(request)
            response["usage"]["backend"] = "codex_cli"
            return response

    def restored_prefix(agent, laboratory, case):
        actions = laboratory.confirmation_actions(72, Random(case["prefix_seed"]))
        feedback = agent._execute("query", {"actions": actions}, laboratory)
        # Real zero-LLM practice events have no external backend marker.
        agent.events.append({"step": 0, "action": "query", "arguments": {"actions": actions},
                             "feedback": feedback})
        return {"prior_research_cost": dict(agent.costs)}

    plan = protocol(kinds=("memoryless",), arms=("hegel",), config=InquiryLLMConfig(
        observation_budget=144, confirmation_rows=24, max_model_calls=6,
        execution_seconds=30., token_budget=10000))
    summary = run_study(plan, tmp_path, provider_factory=lambda _: ProviderMetadata(),
        knowledge={"files": {"shared/old.py": "# retained original evidence\n"}},
        prepare_prefix=restored_prefix)
    assert summary["evidence_kind"] == "real_external_model"
    training = json.loads((tmp_path / "case_000/hegel/training.json").read_text())
    assert "usage" not in training["events"][0]
    assert len(training["events"]) == 6
    assert training["costs"]["observation_cost"] == 99
    assert training["prior_research_cost"]["observation_cost"] == 72
    assert training["continuation_cost"]["observation_cost"] == 27
