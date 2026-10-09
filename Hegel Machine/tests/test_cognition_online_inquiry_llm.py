"""Interface regressions; scripted providers are not LLM performance evidence."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.online_inquiry import InquiryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig
from scripts.run_online_inquiry_llm import protocol, run_study, laboratory


class PythonFittingProvider:
    def __init__(self, workspace):
        self.workspace = Path(workspace)

    def next_action(self, request):
        stage = len(request["events"])
        experiment = ("def select_actions(spec,observations,memory):\n"
            " return {'knowledge_sources':['method1'],'actions':[{'kind':'response','point':dict(zip(spec['controls'],p))} "
            "for p in [(1,0),(0,1),(-.5,.3),(.25,-.8)]]}\n")
        question = {"id": "q1", "question": "Which control predicts the response?",
                    "parent_id": None, "depends_on": []}
        if stage == 0:
            inherited = request["session_knowledge"].get("files", {})
            if "shared/plan.py" in inherited:
                action, arguments = "query", {"path": "shared/plan.py", "question": question}
            else:
                action, arguments = "write", {"path": "shared/plan.py", "code": experiment}
        elif request["events"][-1]["action"] == "write":
            action, arguments = "query", {"path": "shared/plan.py", "question": question}
        elif request["events"][-1]["action"] == "query":
            # Actual subprocess fitting uses only acquired rows. No lab truth.
            experiment = ("def select_actions(spec,observations,memory):\n"
                " return {'knowledge_sources':['method1'],'actions':[{'kind':'response','point':dict(zip(spec['controls'],p))} "
                "for p in [(1,0),(0,1),(-.5,.3),(.25,-.8)]]}\n")
            manifest = {"hypotheses": [{"id": "h1", "path": "shared/hypothesis.py"}],
                "support_questions": [{"id": "q1", "question": "Which control predicts the response?",
                                       "evidence": [0, 1, 2, 3]}],
                "research_programs": [{"id": "method1", "path": "shared/plan.py", "depends_on": ["q1"]}]}
            code = "\n".join([
                "import json,numpy as np", "from pathlib import Path",
                "rows=json.load(open('model_observations.json'))",
                "names=list(rows[0]['features'])",
                "x=np.array([[r['features'][n] for n in names] for r in rows]);y=np.array([r['target'] for r in rows])",
                "coeff=np.linalg.lstsq(x,y,rcond=None)[0].tolist()",
                "source='def predict(row):\\n return sum(a*row[\"features\"][n] for a,n in zip('+repr(coeff)+','+repr(names)+'))\\n'",
                "Path('model.py').write_text(source)", "Path('shared').mkdir(exist_ok=True)",
                "Path('shared/hypothesis.py').write_text(source)",
                f"Path('shared/plan.py').write_text({experiment!r})",
                f"Path('shared/knowledge.json').write_text({json.dumps(manifest)!r})",
                "print('fitted',len(rows),'paid rows')",
            ])
            action, arguments = "run", {"code": code}
        else:
            action, arguments = "finish", {"model": "shared/hypothesis.py", "lesson": "Use independent controls to identify response."}
        return {"action": action, "arguments": arguments,
                "usage": {"backend": "scripted_interface_test", "total_tokens": 3}}


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_null_optional_manifest_uses_default_without_bypassing_hegel_validation(tmp_path, arm):
    class NullManifestProvider(PythonFittingProvider):
        def next_action(self, request):
            decision = super().next_action(request)
            if decision["action"] == "finish":
                decision["arguments"]["manifest"] = None
                (self.workspace / "shared/knowledge.json").unlink()
            return decision

    lab = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)
    agent = InquiryLLMAgent(NullManifestProvider(tmp_path), tmp_path, arm=arm,
        config=InquiryLLMConfig(observation_budget=7, confirmation_rows=3, max_model_calls=4))
    result = agent.run(lab)
    submitted = result["events"][-1]
    assert submitted["action"] == "finish" and submitted["arguments"]["manifest"] is None
    if arm == "code_react":
        assert result["status"] == "completed"
        assert result["fresh_confirmation"]["passed"]
        assert result["costs"]["observation_cost"] == lab.observation_cost == 7
        assert "error" not in submitted["feedback"]
    else:
        assert "Hegel memory requires nonempty" in submitted["feedback"]["error"]
        assert result["model"] is None and result["fresh_confirmation"] is None
        assert result["costs"]["observation_cost"] == lab.observation_cost == 4


@pytest.mark.parametrize("error_scale", [.5, 2.])
def test_confirmation_threshold_is_public_before_decision_and_matches_fresh_measurement(tmp_path, error_scale):
    from hegel_machine.cognition.online_inquiry_llm import CONFIRMATION_MSE_THRESHOLD

    class ThresholdAwareProvider:
        def next_action(self, request):
            assert str(CONFIRMATION_MSE_THRESHOLD) in request["task_interface"]
            return {"action": "finish", "arguments": {"model": "model.py"},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 3}}

    lab = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)
    agent = InquiryLLMAgent(ThresholdAwareProvider(), tmp_path, arm="code_react",
        config=InquiryLLMConfig(observation_budget=7, confirmation_rows=3, max_model_calls=1))
    names = lab.public_spec()["controls"]
    agent._query({"actions": [{"kind": "response", "point": dict(zip(names, point))}
        for point in ((1., 0.), (0., 1.), (-.5, .3), (.25, -.8))]}, lab)
    bias = error_scale * CONFIRMATION_MSE_THRESHOLD ** .5
    agent._execute("write", {"path": "model.py", "code":
        "import numpy as np\n"
        "def fit(rows):\n"
        " global names, coefficients\n"
        " names = list(rows[0]['features'])\n"
        " design = [[row['features'][name] for name in names] for row in rows]\n"
        " coefficients = np.linalg.lstsq(design, [row['target'] for row in rows], rcond=None)[0]\n"
        "def predict(row):\n"
        f" return float(sum(c * row['features'][n] for c, n in zip(coefficients, names)) + {bias!r})\n"}, lab)
    result = agent.run(lab)
    fresh = result["fresh_confirmation"]
    assert fresh["mse_threshold"] == CONFIRMATION_MSE_THRESHOLD
    assert fresh["mse"] == pytest.approx(bias * bias, rel=1e-5, abs=1e-18)
    assert fresh["passed"] == (fresh["mse"] <= fresh["mse_threshold"]) == (error_scale < 1)
    assert result["costs"]["model_calls"] == 1
    assert result["costs"]["observation_cost"] == lab.observation_cost == 7


@pytest.mark.parametrize("observation_budget,expected_status,expected_calls", [
    (6, "confirmation_budget_exhausted", 2),
    (7, "completed", 4),
])
def test_run_stops_when_confirmation_is_unaffordable_but_allows_exact_reserve(
        tmp_path, observation_budget, expected_status, expected_calls):
    class CountingProvider(PythonFittingProvider):
        calls = 0

        def next_action(self, request):
            self.calls += 1
            return super().next_action(request)

    provider = CountingProvider(tmp_path)
    lab = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)
    agent = InquiryLLMAgent(provider, tmp_path, arm="code_react",
        config=InquiryLLMConfig(observation_budget=observation_budget,
                               confirmation_rows=3, max_model_calls=6))
    result = agent.run(lab)
    assert result["status"] == expected_status
    assert provider.calls == result["costs"]["model_calls"] == expected_calls
    assert result["costs"]["reported_tokens"] == 3 * expected_calls
    assert len(json.loads((tmp_path / "model_observations.json").read_text())) == 4
    if observation_budget == 6:
        # The second call acquires four legal rows. There is no subsequent model
        # call, fit or submission after the three-row confirmation becomes unaffordable.
        assert [event["action"] for event in result["events"]] == ["write", "query"]
        assert result["model"] is None and result["fresh_confirmation"] is None
        assert result["costs"]["observation_cost"] == lab.observation_cost == 4
        assert result["costs"]["compute_cost"] == 0
        assert len(agent.rows) == len(result["paid_observations"]) == 4
    else:
        assert result["model"] == "shared/hypothesis.py"
        assert result["fresh_confirmation"]["passed"]
        assert result["fresh_confirmation"]["rows"] == 3
        assert result["costs"]["observation_cost"] == lab.observation_cost == 7


def test_same_tools_real_python_memory_and_terminal_order(tmp_path):
    events = []
    class RecordingLab(InquiryLab):
        def execute(self, action):
            events.append("training")
            return super().execute(action)
        def evaluation_rows(self, **kwargs):
            events.append(("terminal", kwargs["seed"]))
            return super().evaluation_rows(**kwargs)
    def factory(task):
        return RecordingLab(**{key: value for key, value in task.items()
            if key not in {"evaluation_seed", "task_id", "confirmation_seed"}})
    plan = protocol(model="scripted-test-only")
    summary = run_study(plan, tmp_path, provider_factory=PythonFittingProvider,
                        lab_factory=factory, workers=3)
    assert summary["evidence_kind"] == "interface_test_not_llm_performance"
    terminal_start = next(index for index, item in enumerate(events) if isinstance(item, tuple))
    assert all(isinstance(item, tuple) for item in events[terminal_start:])
    terminal_seeds = [item[1] for item in events[terminal_start:]]
    assert terminal_seeds[:2] == terminal_seeds[2:4] == terminal_seeds[4:]
    for arm, result in summary["arms"].items():
        assert result["confirmed_and_correct"] == 2
        assert result["costs_including_source"]["observation_cost"] == 14
        assert result["costs_including_source"]["compute_cost"] == 14
        chain = json.loads((tmp_path / plan["cases"][0]["chain_id"] / arm / "result.json").read_text())
        # Confirmation labels must not replace the paid fitting set at terminal.
        training_file = Path(chain["tasks"][1]["workspace"]) / "model_observations.json"
        assert len(json.loads(training_file.read_text())) == 4
        if arm == "hegel":
            assert result["inherited_entrypoint_attempts"] == result["inherited_entrypoint_successes"] == 2
            replay = chain["tasks"][1]["events"][0]["feedback"]
            assert replay["knowledge_sources"] == ["method1"]
            assert replay["inherited_source_unchanged"]
            question = replay["question"]
            assert question["opened_before_observation"] <= min(question["evidence"])
            assert question["evidence"] == [0, 1, 2, 3]
        else:
            assert result["inherited_entrypoint_attempts"] == 0
            assert not chain["tasks"][1]["inherited_files"]
        memory = chain["source_memory"]
        if arm == "code_react":
            assert memory == {}
        if arm == "textual_memory":
            assert set(memory) == {"lessons"}
            assert memory["lessons"]


def test_partial_invalid_query_charges_real_acquisitions(tmp_path):
    lab = InquiryLab(seed=3)
    agent = InquiryLLMAgent(None, tmp_path, arm="code_react")
    names = lab.public_spec()["controls"]
    feedback = agent._query({"actions": [
        {"kind": "response", "point": dict.fromkeys(names, .5)},
        {"kind": "response", "point": {names[0]: .5}}]}, lab)
    assert "error" in feedback
    assert agent.costs["observation_cost"] == lab.observation_cost == 1
    assert len(json.loads((tmp_path / "observations.json").read_text())) == 1
    denied = agent._query({"actions": [{"kind": "calibrate", "level": 0}] * 24}, lab)
    assert "error" in denied and lab.observation_cost == 1


def test_confirmation_uses_frozen_fit_and_charges_counterexample(tmp_path):
    lab = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)
    agent = InquiryLLMAgent(None, tmp_path, arm="code_react",
                            config=InquiryLLMConfig(observation_budget=4))
    names = lab.public_spec()["controls"]
    agent._query({"actions": [{"kind": "response", "point": dict.fromkeys(names, 0)}]}, lab)
    (tmp_path / "model.py").write_text("import json\ndef fit(rows):\n global seen\n rows=json.load(open('observations.json'))\n seen={tuple(r['features'].values()):r['target'] for r in rows}\n for r in json.load(open('laboratory_observations.json')):\n  if r['action']['kind']=='response': seen[tuple(r['action']['point'].values())]=r['value']\ndef predict(row):\n return seen.get(tuple(row['features'].values()),0.0)\n")
    result = agent._execute("finish", {"model": "model.py"}, lab)
    assert not result["fresh_confirmation"]["passed"]
    assert agent.model is None
    assert agent.costs["observation_cost"] == 4
    assert agent.costs["compute_cost"] == 4
    assert len(agent.rows) == 4  # failed checks become usable paid evidence
    denied = agent._execute("finish", {"model": "model.py"}, lab)
    assert "reserve observation" in denied["error"]
    assert lab.observation_cost == 4


def test_main_target_rotation_and_same_public_problem():
    plan = protocol(model="chosen-by-user", profile="main", cases=30)
    conditions = [case["consumers"][0]["condition"] for case in plan["cases"]]
    assert sorted(conditions.count(name) for name in set(conditions)) == [7, 7, 8, 8]
    source = plan["cases"][0]["source"]["lab"]
    lab = laboratory(source)
    assert "mechanism" not in lab.public_spec()
    assert "driver" not in lab.public_spec()


def test_empty_manifest_is_not_executable_research_memory(tmp_path):
    agent = InquiryLLMAgent(None, tmp_path, arm="hegel")
    manifest = {"hypotheses": [{"id": "claim-only"}], "support_questions": [], "research_programs": []}
    assert "nonempty" in agent._manifest_error(manifest, "model.py")
    manifest["support_questions"] = [{"id": "q1"}]
    manifest["research_programs"] = [{"id": "p1", "path": "missing.py"}]
    assert "existing" in agent._manifest_error(manifest, "model.py")
    (tmp_path / "model.py").write_text("def predict(row): return 0.0\n")
    (tmp_path / "plan.py").write_text("def select_actions(spec,observations,memory): return []\n")
    manifest["hypotheses"] = [{"id": "h1", "path": "model.py"}]
    manifest["research_programs"] = [{"id": "p1", "path": "plan.py"}]
    manifest["support_questions"] = []  # no fabricated child is necessary
    assert "actually execute" in agent._manifest_error(manifest, "model.py")


def test_paid_malformed_action_becomes_feedback_before_real_research_continues(tmp_path):
    class RepairingProvider(PythonFittingProvider):
        def next_action(self, request):
            if not request["events"]:
                return {"action": "provider_error", "arguments": {},
                        "error": {"decode_error": "expected exactly one research_action function call"},
                        "usage": {"backend": "scripted_interface_test", "total_tokens": 3}}
            if len(request["events"]) == 1:
                previous = request["events"][0]
                assert previous["action"] == "provider_error"
                assert previous["feedback"]["error"]
                assert request["costs"]["reported_tokens"] == 3
                assert request["costs"]["observation_cost"] == 0
                assert not request["observations"]
            # The test provider repairs its next action; the host never repairs
            # malformed arguments or executes any action from the failed reply.
            valid_history = [event for event in request["events"]
                             if event["action"] != "provider_error"]
            return super().next_action({**request, "events": valid_history})

    agent = InquiryLLMAgent(RepairingProvider(tmp_path), tmp_path, arm="code_react",
                            config=InquiryLLMConfig(max_model_calls=5))
    result = agent.run(InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1))

    assert result["status"] == "completed"
    assert result["fresh_confirmation"]["passed"]
    assert [event["action"] for event in result["events"]] == [
        "provider_error", "write", "query", "run", "finish"]
    assert result["costs"]["model_calls"] == 5
    assert result["costs"]["reported_tokens"] == 15
    assert result["costs"]["observation_cost"] == 7
    assert result["costs"]["compute_cost"] == 7
    assert result["costs"]["python_execution_seconds"] > 0


@pytest.mark.parametrize("max_calls,token_budget,expected_calls,status", [
    (3, 100, 3, "model_call_budget_exhausted"),
    (10, 6, 2, "token_budget_exhausted"),
])
def test_repeated_malformed_actions_exhaust_original_limits(
        tmp_path, max_calls, token_budget, expected_calls, status):
    class MalformedProvider:
        def next_action(self, request):
            return {"action": "provider_error", "arguments": {},
                    "error": {"decode_error": "invalid arguments_json"},
                    "usage": {"total_tokens": 3}}

    agent = InquiryLLMAgent(MalformedProvider(), tmp_path, arm="code_react",
        config=InquiryLLMConfig(max_model_calls=max_calls, token_budget=token_budget))
    result = agent.run(InquiryLab(seed=8))

    assert result["status"] == status
    assert len(result["events"]) == expected_calls
    assert result["costs"]["model_calls"] == expected_calls
    assert result["costs"]["reported_tokens"] == 3 * expected_calls
    assert result["costs"]["observation_cost"] == 0
    assert result["costs"]["python_execution_seconds"] == 0
    assert result["model"] is None


@pytest.mark.parametrize("error,tokens", [
    ({"decode_error": "invalid arguments_json"}, None),
    ({"status": 500, "message": "model_unavailable"}, 3),
    ({"kind": "empty_response", "message": "no action or text"}, 3),
])
def test_unknown_usage_and_service_failures_still_stop(tmp_path, error, tokens):
    calls = []

    class FailingProvider:
        def next_action(self, request):
            calls.append(1)
            return {"action": "provider_error", "arguments": {}, "error": error,
                    "usage": {"total_tokens": tokens}}

    agent = InquiryLLMAgent(FailingProvider(), tmp_path, arm="code_react",
                            config=InquiryLLMConfig(max_model_calls=5))
    result = agent.run(InquiryLab(seed=8))

    assert calls == [1]
    assert result["status"] == "provider_failed"
    assert result["costs"]["model_calls"] == 1
    assert result["costs"]["reported_tokens"] == tokens
    assert result["costs"]["observation_cost"] == 0
