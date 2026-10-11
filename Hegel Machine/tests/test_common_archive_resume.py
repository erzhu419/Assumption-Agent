"""Resume the actual temporal interface without fresh labels or renewed budgets."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from random import Random

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_online_research import ScriptedProvider
from test_temporal_prospective_inputs import trajectory
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


SEED = 1709
CONFIG = InquiryLLMConfig(observation_budget=24, confirmation_rows=2,
    max_model_calls=8, token_budget=100, execution_seconds=30., compute_budget=1000)


def agent_at(path, *, provider=None, config=CONFIG):
    return CommonArchiveInquiryAgent(provider, path, arm="hegel", config=config,
        source_task=False, confirmation_seed=SEED,
        knowledge={"files": {
            "inherited.py": "def select_actions(spec, observations, memory): return []\n",
            "shared/knowledge.json": json.dumps({"hypotheses": [{"path": "model.py"}],
                "research_programs": [{"path": "inherited.py"}], "support_questions": []})}})


def execute(agent, lab, action, arguments):
    # Match the real provider's pre-action context and final post-action event.
    context = agent._request(lab)
    (agent.workspace / "research_context.json").write_text(json.dumps(context))
    step = agent.costs["model_calls"]
    agent.costs["model_calls"] += 1
    agent.costs["reported_tokens"] += 1
    feedback = agent._execute(action, arguments, lab)
    agent.events.append({"step": step, "action": action, "arguments": deepcopy(arguments),
                         "feedback": deepcopy(feedback), "usage": {"total_tokens": 1}})
    return feedback


def checkpoint(tmp_path):
    agent, lab = agent_at(tmp_path / "source"), TemporalInquiryLab(task_index=1)
    execute(agent, lab, "query", {"actions": [trajectory(0.)]})
    target = agent.rows[0].target
    execute(agent, lab, "evaluate", {"model": "model.py",
        "code": f"def predict(row): return {target!r}\n", "notes": "Current fit; long paths unresolved."})
    failed = execute(agent, lab, "finish", {"model": "model.py"})
    assert failed["fresh_confirmation"]["passed"] is False
    execute(agent, lab, "write", {"path": "latest.py", "code": "# A new unfinished draft\n",
        "research_question": {"id": "new_focus", "question": "Explain the new counterexamples.",
                              "parent_id": "task", "depends_on": []},
        "notes": "Latest notebook after failed confirmation."})
    return agent, lab, agent.result("token_budget_exhausted")


def test_restore_preserves_original_knowledge_latest_state_and_refuted_incumbent(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    (old.workspace / "evaluation.json").write_text('{"terminal_label": 999}')
    (old.workspace / "fit.bin").write_bytes(b"paid fitted dependency")
    before = {p.relative_to(old.workspace): p.read_bytes() for p in old.workspace.rglob("*") if p.is_file()}
    continued = agent_at(tmp_path / "continued", config=replace(CONFIG, token_budget=400))
    restored_lab = TemporalInquiryLab(task_index=1)
    restored = continued.restore_research(previous, old.workspace, restored_lab, confirmation_seed=SEED)
    assert restored["new_observations"] == 0
    assert continued.costs == old.costs and restored_lab.costs == lab.costs
    assert continued.events == old.events and continued.rows == old.rows
    assert continued.questions == old.questions and continued._active_question_id == "new_focus"
    assert continued.knowledge == old.knowledge
    assert continued.inherited_files == {"inherited.py", "shared/knowledge.json"}
    assert "latest.py" in continued.files and "latest.py" not in continued.inherited_files
    assert continued._evaluated_summary() == old._evaluated_summary()
    assert continued._evaluated_summary()["support"] == "new_paid_rows_unscored"
    assert (continued.workspace / "research_notes.md").read_text() == "Latest notebook after failed confirmation."
    assert (continued.workspace / "fit.bin").read_bytes() == b"paid fitted dependency"
    assert not (continued.workspace / "evaluation.json").exists()
    assert before == {p.relative_to(old.workspace): p.read_bytes() for p in old.workspace.rglob("*") if p.is_file()}
    denied = continued._execute("finish", {"model": "model.py"}, restored_lab)
    assert "paid evidence contradicts" in denied["error"]
    assert restored_lab.costs == lab.costs


def test_confirmation_stream_continues_after_paid_failure_without_repurchase(tmp_path, monkeypatch):
    old, lab, previous = checkpoint(tmp_path)
    expected = lab.confirmation_actions(2, old.confirmation_rng)
    resumed, target = agent_at(tmp_path / "continued"), TemporalInquiryLab(task_index=1)
    def no_purchase(*args, **kwargs):
        raise AssertionError("restoration must not buy or replay observations")
    monkeypatch.setattr(target, "execute", no_purchase)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert target.confirmation_actions(2, resumed.confirmation_rng) == expected
    assert resumed.confirmations == old.confirmations
    assert not target._laboratory.acquisitions


def test_run_uses_remaining_total_calls_and_preserves_provider_usage_numbering(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    provider = ScriptedProvider([("run", {"code": "print('continue the same investigation')"})])
    config = replace(CONFIG, token_budget=400, max_model_calls=previous["costs"]["model_calls"] + 1)
    resumed = agent_at(tmp_path / "continued", provider=provider, config=config)
    target = TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert provider.calls == previous["costs"]["model_calls"]
    result = resumed.run(target)
    assert result["status"] == "model_call_budget_exhausted"
    assert len(provider.requests) == 1
    assert result["costs"]["model_calls"] == config.max_model_calls
    assert result["costs"]["reported_tokens"] == previous["costs"]["reported_tokens"] + 1
    assert result["events"][-1]["step"] == previous["costs"]["model_calls"]
    assert result["costs"]["observation_cost"] == previous["costs"]["observation_cost"]


def test_saved_comparison_and_child_return_remain_available_to_parent(tmp_path):
    old, lab = agent_at(tmp_path / "source"), TemporalInquiryLab(task_index=1)
    execute(old, lab, "query", {"actions": [trajectory(0.)]})
    base = old.rows[0].target
    for name, extra in (("a.py", "0."), ("b.py", "len(row['history'])")):
        execute(old, lab, "write", {"path": name,
            "code": f"def predict(row): return {base!r} + {extra}\n"})
    comparison = execute(old, lab, "run", {"operator": "compare_experiments",
        "models": ["a.py", "b.py"], "experiments": [{"id": "longer", "actions": [trajectory(0., 0.)]}]})
    execute(old, lab, "write", {"path": "readout.py", "code": (
        "def readout(result,spec,memory):\n"
        "    return {'status':'supported','answer':{'selected':'a.py'},"
        "'evidence':result['observations'][0]['row_indices']}\n")})
    query = deepcopy(comparison["comparison"]["query_arguments"])
    query["revision"]["readout"] = "readout.py"
    query["question"] = {"id": "child", "question": "Compare the authored predictions.",
                          "parent_id": "task", "depends_on": []}
    execute(old, lab, "query", query)
    previous = old.result("token_budget_exhausted")
    resumed, target = agent_at(tmp_path / "continued"), TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert resumed.experiment_comparisons == old.experiment_comparisons
    assert resumed.revisions == old.revisions
    request = resumed._request(target)
    answer = request["online_research"]["parent_returns"][0]["answers"][-1]
    assert answer["answer"]["selected"] == "a.py"
    evaluated = resumed._execute("evaluate", {"model": answer["answer"]["selected"],
        "research_question": {"id": "task"}}, target)
    assert evaluated["mse"] < 1e-10
    assert resumed.costs["observation_cost"] == old.costs["observation_cost"]


def test_restored_exhausted_python_stops_without_another_model_call(tmp_path):
    old, _, previous = checkpoint(tmp_path)
    previous["costs"]["python_execution_seconds"] = CONFIG.execution_seconds + .1
    provider = ScriptedProvider([])
    resumed = agent_at(tmp_path / "continued", provider=provider,
                       config=replace(CONFIG, token_budget=400))
    lab = TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, lab, confirmation_seed=SEED)
    result = resumed.run(lab)
    assert result["status"] == "python_execution_budget_exhausted"
    assert provider.requests == []
    assert result["costs"] == previous["costs"]
    assert result["events"] == previous["events"]
    assert result["model"] is None
