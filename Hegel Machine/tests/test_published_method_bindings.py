"""Published code becomes an optional executable choice, not automatic adoption."""

from copy import deepcopy
from dataclasses import asdict
import json
from random import Random

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import (
    METHOD_PATH, method_memory_pair, register_published_method,
)


# A scripted fixture verifies invocation and accounting, not autonomous learning.
PREDICTOR = '''def fit(rows):
    global slope, intercept
    xs = [r['features']['u'] for r in rows]
    ys = [r['target'] for r in rows]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / sum((x-mx)**2 for x in xs)
    intercept = my - slope*mx
def predict(row):
    return slope*row['features']['u'] + intercept
'''
METHOD = '''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if not steps:
        return {'action':'write','arguments':{'path':'local_model.py','code':__MODEL__}}
    return {'model':'local_model.py'}
'''.replace('__MODEL__', repr(PREDICTOR))


def publication():
    trial = {"case": "case_000", "status": "completed", "method_source": METHOD,
             "fresh_confirmation": {"passed": True, "rows": 96, "mse": 0.},
             "selected_prior_predictor": False, "costs": {"observation_cost": 168}}
    return {"status": "completed", "method_path": METHOD_PATH, "method_source": METHOD,
            "method_entry": {"id": "learned_method", "path": METHOD_PATH},
            "lesson": "Use current paid evidence before requesting confirmation.",
            "practice_trials": [
                {**trial, "status": "method_stopped", "fresh_confirmation": None},
                {**trial, "case": "different_draft_case", "method_source": "an older draft"},
                trial,
            ]}


def original_memory():
    manifest = {"hypotheses": [], "research_programs": [
        {"id": "old_question", "path": "shared/old_plan.py"}], "support_questions": []}
    return {"files": {
        METHOD_PATH: "def select_actions(*args):\n    return {'stop':'Unresolved placeholder'}\n",
        "shared/old_plan.py": "def select_actions(*args):\n    return {'stop':'An earlier question'}\n",
        "shared/knowledge.json": json.dumps(manifest),
        "source_catalog.json": "[]",
    }, "manifest": manifest, "lessons": [],
        "research_roles": {"predictors": []}, "source_catalog": [],
        "archive_instructions": "Original source fitting views remain in their own directories."}


def test_registration_replaces_placeholder_and_exposes_only_current_source_support():
    original, learned = original_memory(), publication()
    before = deepcopy(original)
    retained = register_published_method(original, learned, max_steps=12)
    assert original == before
    assert retained["files"][METHOD_PATH] == METHOD
    assert set(retained["files"]) == set(original["files"])
    assert retained["research_roles"] == {"predictors": [], "method": METHOD_PATH}
    assert retained["source_catalog"] == original["source_catalog"]
    entry = retained["manifest"]["research_programs"][-1]
    assert entry["id"] == "learned_method" and entry["max_steps"] == 12
    assert entry["prior_execution"]["cases"] == [{key: value for key, value in
        learned["practice_trials"][-1].items() if key != "method_source"}]
    assert json.loads(retained["files"]["shared/knowledge.json"]) == retained["manifest"]

    # Publishing a replacement updates the actual entry rather than accumulating
    # stale actions with the same identity or path.
    replacement = {**learned, "method_source": METHOD + "\n# revised draft\n", "practice_trials": []}
    updated = register_published_method(retained, replacement, max_steps=12)
    entries = updated["manifest"]["research_programs"]
    assert len(entries) == 2 and "prior_execution" not in entries[-1]
    assert retained["files"][METHOD_PATH] == METHOD


def test_legacy_memory_pair_still_removes_only_the_published_program_and_entry():
    original = original_memory()
    original["files"].pop(METHOD_PATH)
    retained, deleted = method_memory_pair(original, publication())
    assert retained["research_roles"] == deleted["research_roles"] == original["research_roles"]
    erased = deepcopy(retained)
    erased["files"].pop(METHOD_PATH)
    erased["manifest"]["research_programs"] = [entry for entry in erased["manifest"]["research_programs"]
                                                if entry["path"] != METHOD_PATH]
    erased["files"]["shared/knowledge.json"] = json.dumps(erased["manifest"], indent=2)
    assert erased == deleted
    with pytest.raises(ValueError, match="new file"):
        method_memory_pair(original_memory(), publication())


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_advertised_binding_executes_method_while_plain_module_run_does_not(tmp_path, arm):
    knowledge = register_published_method(original_memory(), publication(), max_steps=12)
    agent = CommonArchiveInquiryAgent(None, tmp_path / arm, arm=arm, knowledge=knowledge,
        source_task=False, config=InquiryLLMConfig(observation_budget=144, confirmation_rows=96,
            max_model_calls=0, token_budget=0, execution_seconds=30.))
    lab = AnonymousHistoryLab("memoryless", seed=31)
    agent._execute("query", {"actions": lab.confirmation_actions(24, Random(91))}, lab)
    agent._prepare_program_state()
    request = {"files": agent.files, "session_knowledge": agent.knowledge,
        "observations": [asdict(row) for row in agent.rows],
        "laboratory_observations": agent.observations, "events": []}
    context = prepare_code_context(agent.workspace, request, schema={}, instructions="Test only")
    method = next(entry for entry in context["method_decision"]["methods"]
                  if entry["path"] == METHOD_PATH)
    assert method["prior_execution"]["cases"][0]["fresh_confirmation"]["passed"]
    assert method["current_executions"] == []
    assert method["execute_action"] == {"action": "run", "arguments": {
        "operator": "procedure", "path": METHOD_PATH, "max_steps": 12, "confirm": True}}
    assert agent.costs["observation_cost"] == 24 and agent.method_results == []

    module = agent._execute("run", {"path": METHOD_PATH}, lab)
    assert module["returncode"] == 0 and module["stdout"] == ""
    assert agent.costs["observation_cost"] == 24
    assert agent.method_results == [] and agent.model is None
    assert not (agent.workspace / "local_model.py").exists()

    action = method["execute_action"]
    executed = agent._execute(action["action"], action["arguments"], lab)
    assert executed["status"] == "completed"
    assert executed["fresh_confirmation"]["passed"]
    assert agent.model == "local_model.py" and len(agent.method_results) == 2
    assert agent.costs["observation_cost"] == 24 + 96
    assert agent.costs["model_calls"] == 0
    assert (agent.workspace / METHOD_PATH).read_text() == METHOD
    outgoing = agent.result("completed")["outgoing_knowledge"]
    for key in ("research_roles", "source_catalog", "archive_instructions"):
        assert outgoing[key] == knowledge[key]
    retained_method = next(entry for entry in outgoing["manifest"]["research_programs"]
                           if entry["path"] == METHOD_PATH)
    assert retained_method["max_steps"] == 12
    assert retained_method["prior_execution"] == method["prior_execution"]
    assert any(entry["path"] == "local_model.py" for entry in outgoing["manifest"]["hypotheses"])

    # The serialized binding must remain callable in a subsequent task, too.
    successor = CommonArchiveInquiryAgent(None, tmp_path / (arm + "_next"), arm=arm,
        knowledge=outgoing, source_task=False, config=agent.config)
    successor._prepare_program_state()
    next_methods = json.loads(successor.files["method_decision.json"])["methods"]
    next_entry = next(entry for entry in next_methods if entry["path"] == METHOD_PATH)
    assert next_entry["execute_action"] == method["execute_action"]
    assert next_entry["prior_execution"] == method["prior_execution"]
