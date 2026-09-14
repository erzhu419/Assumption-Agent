"""Persistent tool-loop behavior uses deterministic providers, not model evidence."""

import json
import pytest

from hegel_machine.cognition.code_agent import CodeAgent, CodeAgentConfig, evaluate_python_model
from hegel_machine.cognition.code_session import CodeComparisonSession, CodeResearchSession
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.models import ResearchConfig


class SharingProvider:
    def next_action(self, request):
        step = len(request["events"])
        inherited = request["session_knowledge"]["files"]
        if step == 0:
            action, arguments = "query", {"sequence": [{"u": i / 16 - 1} for i in range(24)]}
        elif step == 1 and not inherited:
            action, arguments = "write", {"path": "shared/model.py", "code": "def predict(row):\n    return row['features']['u']\n"}
        elif step == 1:
            assert "shared/model.py" in request["files"]
            assert request["session_knowledge"]["lessons"]
            action, arguments = "run", {"code": "from shared.model import predict\nprint(predict({'features':{'u':0.25}}))"}
        else:
            action, arguments = "finish", {"model": "shared/model.py", "lesson": "A reusable control-only starting model.",
                "reused_files": ["shared/model.py"] if inherited else []}
        return {"action": action, "arguments": arguments, "usage": {"backend": "test_protocol", "total_tokens": 7}}


def test_stopped_code_task_without_an_executable_model_is_visited_but_not_completed(tmp_path):
    class WriteOnlyProvider:
        def next_action(self, request):
            return {"action": "write", "arguments": {"path": "notes.txt", "code": "Research is unfinished."},
                    "usage": {"backend": "test_protocol", "total_tokens": 2}}

    session = CodeComparisonSession(tmp_path, tasks=[{"task_index": 0, "seed": 31}],
        config=CodeAgentConfig(token_budget=1), backend="local",
        provider_factory=lambda _: WriteOnlyProvider())
    session.code.run()
    state = session.to_dict()
    assert state["code"]["status"] == "completed"
    assert state["code"]["results"][0]["status"] == "token_budget_exhausted"
    assert state["code"]["results"][0]["model"] is None
    assert state["code"]["costs"]["observation_cost"] == 0
    assert state["code"]["costs"]["reported_tokens"] == 2
    assert state["summary"]["code_visited_tasks"] == state["summary"]["code_tasks_stopped"] == 1
    assert state["summary"]["code_completed_tasks"] == 0
    assert state["summary"]["hegel_completed_tasks"] == 0


def test_code_session_restores_shared_code_and_spending_then_accepts_new_instance(tmp_path):
    factory = lambda _: SharingProvider()
    config = CodeAgentConfig(observation_budget=24, compute_budget=1000, max_model_calls=3)
    totals = {"observation_budget": 48, "measurement_budget": 0, "compute_budget": 2000,
              "max_model_calls": 6, "token_budget": 100, "execution_seconds": 20}
    session = CodeResearchSession(tmp_path, tasks=[{"task_index": 0, "seed": 51}],
        config=config, totals=totals, backend="local", provider_factory=factory)
    first = session.run()
    restored = CodeResearchSession(tmp_path, state=json.loads(json.dumps(first)), provider_factory=factory)
    restored.add_task(0, 97)
    second = restored.run()
    assert second["status"] == "completed"
    assert second["costs"]["observation_cost"] == 48
    assert second["costs"]["reported_tokens"] == 42
    assert second["results"][1]["reused_files"] == ["shared/model.py"]
    assert second["results"][1]["events"][1]["feedback"]["stdout"] == "0.25\n"
    restored.add_task(1, 98)
    assert restored.run()["status"] == "budget_exhausted"


def test_code_comparison_keeps_final_evaluation_out_of_shared_knowledge(tmp_path):
    class EvaluatedCandidate(SharingProvider):
        def next_action(self, request):
            decision = super().next_action(request)
            if decision["action"] == "finish":
                decision["action"] = "evaluate"
            return decision

    config = CodeAgentConfig(observation_budget=24, compute_budget=12000, max_model_calls=3)
    session = CodeComparisonSession(tmp_path, tasks=[{"task_index": 0, "seed": 31}],
        config=config, hegel_config=ResearchConfig(observation_budget=24, compute_budget=12000, max_rounds=1),
        backend="local", provider_factory=lambda _: EvaluatedCandidate())
    state = session.run()
    assert state["status"] == "completed"
    assert state["code"]["results"][0]["status"] == "model_call_budget_exhausted"
    assert state["evaluation"] is None
    before = json.dumps(state["code"]["knowledge"])
    scores = session.evaluate()
    assert scores[0]["code"]["rows"] == scores[0]["hegel"]["rows"] == 256
    assert json.dumps(session.to_dict()["code"]["knowledge"]) == before
    restored = CodeComparisonSession(tmp_path, state=json.loads(json.dumps(session.to_dict())),
                                     provider_factory=lambda _: SharingProvider())
    assert restored.evaluate() == scores


@pytest.mark.parametrize("interrupt_after", [1, 3])
def test_action_checkpoint_resumes_paid_rows_and_completed_model_without_extra_calls(tmp_path, interrupt_after):
    config = CodeAgentConfig(observation_budget=24, compute_budget=1000, max_model_calls=3)
    session = CodeResearchSession(tmp_path, tasks=[{"task_index": 0, "seed": 51}],
        config=config, backend="local", provider_factory=lambda _: SharingProvider())
    saved = []

    def interrupt(state):
        if state["active"] and len(state["active"]["agent_state"]["events"]) == interrupt_after:
            saved.append(json.loads(json.dumps(state)))
            raise RuntimeError("process interruption")

    with pytest.raises(RuntimeError, match="process interruption"):
        session.run(on_checkpoint=interrupt)
    restored = CodeResearchSession(tmp_path, state=saved[0], provider_factory=lambda _: SharingProvider())
    result = restored.run()
    assert result["status"] == "completed"
    assert result["costs"]["observation_cost"] == 24
    assert result["costs"]["model_calls"] == 3
    assert result["costs"]["compute_cost"] == 24


def test_additional_calls_finish_saved_unfinished_task_without_rebuying_observations(tmp_path):
    config = CodeAgentConfig(observation_budget=24, compute_budget=1000, max_model_calls=2)
    session = CodeResearchSession(tmp_path, tasks=[{"task_index": 0, "seed": 51}],
        config=config, backend="local", provider_factory=lambda _: SharingProvider())
    first = session.run()
    assert first["results"][0]["status"] == "model_call_budget_exhausted"
    totals = {**first["totals"], "max_model_calls": 3}
    restored = CodeResearchSession(tmp_path, state=first, totals=totals,
        config=CodeAgentConfig(observation_budget=24, compute_budget=1000, max_model_calls=3),
        provider_factory=lambda _: SharingProvider())
    result = restored.continue_task(0)
    assert result["status"] == "completed"
    assert restored.state["costs"]["observation_cost"] == 24
    assert restored.state["costs"]["model_calls"] == 3


def test_budget_stop_retains_evaluated_candidate_after_a_later_broken_write(tmp_path):
    class CandidateProvider:
        def next_action(self, request):
            actions = [("query", {"sequence": [{"u": .2}, {"u": -.2}]}),
                ("write", {"path": "model.py", "code": "def predict(row):\n    return row['features']['u']\n"}),
                ("evaluate", {"model": "model.py"}),
                ("write", {"path": "model.py", "code": "def predict(row):\n    raise RuntimeError('unfinished revision')\n"})]
            action, arguments = actions[len(request["events"])]
            return {"action": action, "arguments": arguments, "usage": {"total_tokens": 1}}

    lab = make_lab(0, 17)
    result = CodeAgent(CandidateProvider(), tmp_path, CodeAgentConfig(max_model_calls=4)).run(lab)
    assert result["status"] == "model_call_budget_exhausted"
    assert result["model_origin"] == "last_successful_evaluation"
    rows = lab.evaluation_rows([], seed=29, length=4, trajectories=1)
    evaluation = evaluate_python_model(result["workspace"], result["model"], rows)
    assert evaluation["returncode"] == 0
    assert evaluation["predictions"] == [row.features['u'] for row in rows]
    assert 'unfinished revision' in (tmp_path / 'model.py').read_text()
