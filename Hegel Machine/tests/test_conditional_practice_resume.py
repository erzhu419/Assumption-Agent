"""Resume frozen paid trajectories into the existing live researcher."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_conditional_discovery_study import laboratory


SEED = 2219
TASK = {"seed": 41, "apparatus_seed": 59, "stage": 1, "noise_sigma": 0.}
CONFIG = InquiryLLMConfig(observation_budget=20, confirmation_rows=2,
    max_model_calls=8, token_budget=1000, execution_seconds=30., compute_budget=1000,
    confirmation_mse_threshold=1e-6)
KNOWLEDGE = {"files": {"shared/research.py":
    "def select_actions(spec, observations, memory): return {'model': 'model.py'}\n"}}


def make_agent(path, *, config=CONFIG):
    return CommonArchiveInquiryAgent(None, path, arm="hegel", config=config,
        source_task=False, confirmation_seed=SEED, knowledge=KNOWLEDGE)


def execute(agent, lab, action, arguments):
    feedback = agent._execute(action, arguments, lab)
    agent.events.append({"step": len(agent.events), "action": action,
        "arguments": deepcopy(arguments), "feedback": deepcopy(feedback)})
    return feedback


def checkpoint(tmp_path):
    agent, lab = make_agent(tmp_path / "practice"), laboratory(TASK)
    action = {"kind": "trajectory", "channels": [],
        "sequence": [{"x": .2, "z": -.1, "c": .3}]}
    execute(agent, lab, "query", {"actions": [action, action]})
    target = agent.rows[0].target
    execute(agent, lab, "evaluate", {"model": "model.py",
        "code": f"def predict(row): return {target!r}\n"})
    failed = execute(agent, lab, "query", {"path": "shared/research.py"})
    assert failed["fresh_confirmation"]["passed"] is False
    result = agent.result("confirmation_failed")
    agent._prepare_program_state()
    assert not (agent.workspace / "research_context.json").exists()
    return agent, lab, result


def test_restore_frozen_practice_retains_paid_state_and_does_not_reacquire(tmp_path, monkeypatch):
    old, old_lab, previous = checkpoint(tmp_path)
    (old.workspace / "_evaluated_model" / "research_state.json").write_text('{"host": 1}')
    (old.workspace / "data").mkdir()
    (old.workspace / "data" / "result.json").write_text('{"scientific_dependency": 1}')
    before = {p.relative_to(old.workspace): p.read_bytes()
              for p in old.workspace.rglob("*") if p.is_file()}
    restored, lab = make_agent(tmp_path / "online"), laboratory(TASK)

    def must_not_purchase(*args, **kwargs):
        raise AssertionError("restoration must not simulate or reacquire paid rows")

    monkeypatch.setattr(lab, "execute", must_not_purchase)
    monkeypatch.setattr(lab, "acquire", must_not_purchase)
    recovery = restored.restore_practice_research(previous, old.workspace, lab,
        confirmation_seed=SEED)
    assert recovery["restored_paid_rows"] == 4 and recovery["new_observations"] == 0
    assert recovery["restored_confirmations"] == 1
    assert restored.costs == old.costs == recovery["costs_before"]
    assert lab.costs == old_lab.costs
    for field in ("observations", "rows", "events", "questions", "revisions",
                  "method_results", "experiment_comparisons", "confirmations"):
        assert getattr(restored, field) == getattr(old, field)
    assert restored._evaluated_summary() == old._evaluated_summary()
    assert lab._noise.getstate() == old_lab._noise.getstate()
    assert not (restored.workspace / "research_context.json").exists()
    assert not (restored.workspace / "_evaluated_model" / "research_state.json").exists()
    assert json.loads((restored.workspace / "data/result.json").read_text()) == {
        "scientific_dependency": 1}
    with pytest.raises(ValueError, match="fresh apparatus"):
        restored.restore_practice_research(previous, old.workspace, lab, confirmation_seed=SEED)
    denied = restored._execute("finish", {"model": "model.py"}, lab)
    assert "paid evidence contradicts" in denied["error"]
    assert lab.costs == old_lab.costs
    assert before == {p.relative_to(old.workspace): p.read_bytes()
                      for p in old.workspace.rglob("*") if p.is_file()}


def test_next_confirmation_continues_random_stream_and_snapshot_prediction(tmp_path):
    old, old_lab, previous = checkpoint(tmp_path)
    restored, lab = make_agent(tmp_path / "online"), laboratory(TASK)
    restored.restore_practice_research(previous, old.workspace, lab, confirmation_seed=SEED)
    expected = old_lab.confirmation_actions(2, old.confirmation_rng)
    actual = lab.confirmation_actions(2, restored.confirmation_rng)
    assert actual == expected
    assert actual != [record["action"] for record in previous["fresh_confirmation"]["observations"]]
    assert lab.execute(actual[0]) == old_lab.execute(expected[0])
    assert lab.costs == old_lab.costs
    candidate = previous["fresh_confirmation"]["submitted_candidate"]["model"]
    original = evaluate_python_model(old.workspace, candidate, old.rows)
    resumed = evaluate_python_model(restored.workspace, candidate, old.rows)
    assert original["returncode"] == resumed["returncode"] == 0
    assert original["predictions"] == resumed["predictions"]
    assert original["mse"] == resumed["mse"]
    # Independently editable COW files cannot modify the paid source.
    original_source = (old.workspace / "model.py").read_text()
    (restored.workspace / "model.py").write_text("def predict(row): return 12.\n")
    assert (old.workspace / "model.py").read_text() == original_source


def test_conditional_online_resume_reuses_the_same_restore_and_refit_contract(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    # This is an actual request of an online investigator, unlike zero-LLM practice.
    context = old._request(lab)
    (old.workspace / "research_context.json").write_text(json.dumps(context))
    resumed, target = make_agent(tmp_path / "continued"), laboratory(TASK)
    restored = resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert restored["new_observations"] == 0
    assert target.costs == lab.costs
    request = resumed._request(target)
    assert "operator:'refit'" in request["task_interface"]
    assert "validation_records" in request["task_interface"]
    assert request["online_research"]["latest_execution"]["error"] == (
        previous["events"][-1]["feedback"]["error"])


def test_conditional_noise_stream_is_restored_without_resimulation(tmp_path):
    task = {**TASK, "noise_sigma": .1}
    old, resumed = laboratory(task), laboratory(task)
    actions = old.confirmation_actions(5, Random(61))
    records = [old.execute(action) for action in actions]
    resumed.restore_paid_account(records, old.costs)
    later = old.confirmation_actions(3, Random(67))
    assert old.execute(later[0]) == resumed.execute(later[0])


def test_original_anonymous_target_resumes_paid_indices_and_confirmation(tmp_path):
    from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
    old, lab = make_agent(tmp_path / "practice"), AnonymousHistoryLab(kind="delay", seed=79)
    action = {"kind": "trajectory", "channels": [], "sequence": [{"u": .2}]}
    execute(old, lab, "query", {"actions": [action, action]})
    execute(old, lab, "evaluate", {"model": "model.py", "code":
        f"def predict(row): return {old.rows[0].target!r}\n"})
    feedback = execute(old, lab, "query", {"path": "shared/research.py"})
    assert feedback["fresh_confirmation"]["passed"] is False
    previous = old.result("confirmation_failed")
    old._prepare_program_state()
    resumed, target = make_agent(tmp_path / "continued"), AnonymousHistoryLab(kind="delay", seed=79)
    resumed.restore_practice_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert target.costs == lab.costs
    expected = lab.confirmation_actions(2, old.confirmation_rng)
    actual = target.confirmation_actions(2, resumed.confirmation_rng)
    assert actual == expected
    acquisition = target.execute(actual[0])
    assert acquisition["row_indices"][0] == previous["costs"]["observation_cost"]
    assert acquisition == lab.execute(expected[0])
