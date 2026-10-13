"""Published method continuation uses actual feedback, without automatic retry."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition import online_inquiry_llm
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_inquiry_revision import PublicLab, action


METHOD = "shared/learned_method.py"
OLD = "def predict(row): return 0.\n"
REFIT = """coefficient = 0.
def fit(rows):
    global coefficient
    pairs = [(row['history'][-1]['u'], row['target'])
             for row in rows if row['history']]
    coefficient = sum(x*y for x,y in pairs) / sum(x*x for x,y in pairs)
def predict(row):
    return coefficient * row['history'][-1]['u'] if row['history'] else 0.
"""
PROGRAM = f"""import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    own = [step for step in steps if step['path'] == {METHOD!r}]
    if not own:
        return {{'model': 'old.py'}}
    if own[-1]['feedback'].get('fresh_confirmation', {{}}).get('passed') is False:
        return {{'action': 'write', 'arguments': {{'path': 'repaired.py', 'code': {REFIT!r}}}}}
    return {{'model': 'repaired.py'}}
"""
PRIOR_EXECUTION = {
    "scope": "Earlier paid source practice, not current-task support.",
    "cases": [{"case": "source", "status": "completed",
               "fresh_confirmation": {"passed": True, "mse": 0.}}],
}


class DelayLab(PublicLab):
    """A tiny real temporal apparatus for the continuation interface test."""

    def execute(self, selected):
        record = super().execute(selected)
        for row in record["rows"]:
            row["target"] = .7 * row["history"][-1]["u"] if row["history"] else 0.
        return record

    def confirmation_actions(self, length, rng):
        return [action(*[rng.uniform(-1., 1.) for _ in range(length)])]


def prepared(tmp_path, arm="hegel"):
    knowledge = {
        "files": {"old.py": OLD, METHOD: PROGRAM},
        "manifest": {
            "hypotheses": [{"id": "old", "path": "old.py"}],
            "research_programs": [{"id": "published", "path": METHOD,
                                   "max_steps": 4, "prior_execution": PRIOR_EXECUTION}],
            "support_questions": [],
        },
    }
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm=arm, source_task=False,
        knowledge=knowledge, config=InquiryLLMConfig(observation_budget=20,
            confirmation_rows=4, compute_budget=1000, execution_seconds=30.))
    lab = DelayLab()
    agent._execute("query", {"actions": [action(0., 0., 0.)]}, lab)
    assert not (agent.workspace / "shared/knowledge.json").exists()
    return agent, lab


def method_view(agent, lab):
    request = agent._request(lab)
    context = prepare_code_context(agent.workspace, request, schema={}, instructions="test")
    methods = context["method_decision"]["methods"]
    return next(method for method in methods if method["path"] == METHOD)


def execute(agent, lab, selected):
    return agent._execute(selected["action"], selected["arguments"], lab)


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_failed_confirmation_keeps_published_invocation_and_explicit_refit(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm)
    initial = method_view(agent, lab)
    inherited = deepcopy(agent.knowledge)
    first = execute(agent, lab, initial["execute_action"])

    assert first["status"] == "confirmation_failed"
    assert first["fresh_confirmation"]["passed"] is False
    assert first["fresh_confirmation"]["mse"] > 0.
    assert len(agent.method_results) == 1
    assert not (agent.workspace / "repaired.py").exists()
    assert agent.model is None
    paid = agent.costs["observation_cost"]
    decisions = method_view(agent, lab)
    assert agent.costs["observation_cost"] == paid == 7
    assert decisions["execute_action"] == initial["execute_action"]
    assert decisions["max_steps"] == 4
    assert decisions["prior_execution"] == PRIOR_EXECUTION
    assert decisions["source_unchanged"] is True
    continuation = decisions["continuation"]
    assert continuation["status"] == "prediction_refuted"
    assert continuation["submitted_model"] == "old.py"
    assert continuation["method_step"] == 0
    assert continuation["new_paid_observation_indices"] == [3, 4, 5, 6]
    assert continuation["current_paid_rows"] == 7
    assert continuation["resume_action"] == initial["execute_action"]
    assert continuation["revision_path"] == METHOD
    saved = continuation["submitted_candidate"]
    assert saved["source_model"] == "old.py"
    assert saved["paid_rows"] == 3
    assert saved["mse"] == 0.
    assert (agent.workspace / saved["model"]).is_file()

    # The ordinary caller, not context preparation, chooses to resume.
    second = execute(agent, lab, continuation["resume_action"])
    assert second["status"] == "completed"
    assert second["fresh_confirmation"]["passed"] is True
    assert second["fresh_confirmation"]["mse"] < 1e-20
    assert agent.model == "repaired.py"
    assert agent.costs["observation_cost"] == 11
    assert agent.costs["model_calls"] == 0
    assert (agent.workspace / "old.py").read_text() == OLD
    assert (agent.workspace / METHOD).read_text() == PROGRAM
    assert agent.knowledge == inherited
    assert agent.method_results[0]["feedback"]["fresh_confirmation"]["passed"] is False
    assert agent.method_results[1]["proposal"]["action"] == "write"
    finished = method_view(agent, lab)
    assert not finished.get("continuation")
    assert finished["prior_execution"] == PRIOR_EXECUTION


def test_changed_method_is_visible_and_stop_does_not_reoffer_stale_confirmation(tmp_path):
    agent, lab = prepared(tmp_path)
    first = execute(agent, lab, method_view(agent, lab)["execute_action"])
    assert first["status"] == "confirmation_failed"
    changed = "def select_actions(spec, records, memory): return {'stop': 'revise assumptions'}\n"
    agent._execute("write", {"path": METHOD, "code": changed}, lab)
    view = method_view(agent, lab)
    assert view["source_unchanged"] is False
    paid = agent.costs["observation_cost"]
    stopped = execute(agent, lab, view["execute_action"])
    assert stopped["status"] == "method_stopped"
    assert agent.costs["observation_cost"] == paid
    assert not method_view(agent, lab).get("continuation")
    assert agent.method_results[0]["feedback"]["fresh_confirmation"]["passed"] is False
    evidence = method_view(agent, lab)["submitted_candidates"]
    assert len(evidence) == 1
    assert evidence[0]["confirmation"]["passed"] is False
    assert evidence[0]["paid_rows"] == 3


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_failed_submission_keeps_its_fitting_view_after_method_overwrites_model(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm)
    old = """import json
value = 0.
def fit(rows):
    global value
    value = sum(row['target'] for row in rows) / len(rows) + json.load(open('offset.json'))
def predict(row): return value
"""
    (tmp_path / "offset.json").write_text("0")
    agent._execute("write", {"path": "old.py", "code": old}, lab)
    first = execute(agent, lab, method_view(agent, lab)["execute_action"])
    assert first["status"] == "confirmation_failed"
    saved = method_view(agent, lab)["continuation"]["submitted_candidate"]
    assert len(agent.rows) == 7
    saved_context = (tmp_path / saved["model"]).parent
    assert len(json.loads((saved_context / "model_observations.json").read_text())) == 3

    # Subsequent fitting and source edits must not revise the earlier hypothesis.
    (tmp_path / "offset.json").write_text("99")
    agent._execute("write", {"path": "old.py", "code": "def predict(row): return -99.\n"}, lab)
    scoring = evaluate_python_model(tmp_path, saved["model"], agent.rows)
    assert scoring["returncode"] == 0
    assert scoring["predictions"] == [0.] * 7
    assert scoring["mse"] > 0
    assert (saved_context / "offset.json").read_text() == "0"
    assert len(agent.confirmations) == 1


def test_confirmation_execution_failure_is_not_labelled_a_scientific_counterexample(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path)
    evaluate = online_inquiry_llm.evaluate_python_model

    def fail_only_fresh_scoring(workspace, model, rows, **kwargs):
        if len(rows) == agent.config.confirmation_rows:
            return {"returncode": 1, "mse": None, "timeout": True,
                    "stderr": "candidate scoring exceeded its timeout"}
        return evaluate(workspace, model, rows, **kwargs)

    monkeypatch.setattr(online_inquiry_llm, "evaluate_python_model", fail_only_fresh_scoring)
    result = execute(agent, lab, method_view(agent, lab)["execute_action"])
    assert result["status"] == "confirmation_failed"
    failure = agent.method_results[-1]["feedback"]
    assert failure["fresh_confirmation"]["timeout"] is True
    assert "could not be scored" in failure["error"]
    assert "counterexample" not in failure["error"]
    before = deepcopy(agent.costs)
    continuation = method_view(agent, lab)["continuation"]
    assert continuation["status"] == "confirmation_execution_failed"
    assert continuation["confirmation"]["mse"] is None
    assert continuation["new_paid_observation_indices"] == [3, 4, 5, 6]
    assert continuation["current_paid_rows"] == 7
    assert agent.costs == before
    assert len(agent.observations) == 2
    assert agent.model is None
