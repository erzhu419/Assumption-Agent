"""Scripted interface evidence, not measurements of LLM method invention."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.history_revision import revise_history
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_instrument_research import fit_temporal_instrument
from hegel_machine.cognition.temporal_method_learning import (
    MethodConsolidationAgent, load_training_experience, method_memory_pair,
)
from hegel_machine.cognition.temporal_operator_agent import OPERATOR_INTERFACE, TemporalOperatorAgent


METHOD_PATH = "shared/consolidated_method.py"
LEARNER_LESSON = "Short resets leave the instrument ambiguous; extend the trajectory before reusing it."
OLD_PLAN = """def select_actions(spec, observations, memory):
    return {'actions':[{'kind':'trajectory','sequence':[{'u':.8},{'u':0.},{'u':-.3}], 'channels':[]}],
            'question':{'id':'old_short','question':'Does the short response still fit?',
                        'parent_id':None,'depends_on':[]}}
"""
AUTHORED_METHOD = """from random import Random
def select_actions(spec, observations, memory):
    rng = Random(991)
    actions = [{'kind':'trajectory','sequence':[{'u':rng.uniform(-1.,1.)}
               for _ in range(spec['max_probe_length'])], 'channels':['meter']} for _ in range(3)]
    return {'actions':actions,
            'question':{'id':'extend_meter','question':'Which retained meter law survives longer trajectories?',
                        'parent_id':'response','depends_on':[]},
            'after_observation':[
                {'operator':'assess_instrument','instrument':'instrument.json','records':'acquired'},
                {'operator':'response','instrument':'instrument.json','output_dir':'shared/revised'}]}
"""


class ScriptedMethodAuthor:
    """Exercise authoring and memory transfer without an external model call."""

    def __init__(self):
        self.requests = []

    def next_action(self, request):
        self.requests.append(deepcopy(request))
        if len(self.requests) == 1:
            action, arguments = "write", {"path": METHOD_PATH, "code": AUTHORED_METHOD}
        else:
            action, arguments = "finish", {
                "path": METHOD_PATH,
                "entry": {"id": "learned_long_probe", "path": METHOD_PATH,
                          "depends_on": ["extend_meter"]},
                "lesson": LEARNER_LESSON,
            }
        return {"action": action, "arguments": arguments,
                "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}


@pytest.fixture
def experience_case():
    lab = TemporalInquiryLab(task_index=0, apparatus_seed=910061, instrument_mode="recurrent")
    rng = Random(71)
    records = [lab.execute({"kind": "trajectory", "sequence": [{"u": rng.uniform(-1., 1.)}
        for _ in range(3)], "channels": ["meter"]}) for _ in range(12)]
    rows = [row for record in records for row in record["rows"]]
    instrument = fit_temporal_instrument(records, lab.public_spec())
    manifest = {"hypotheses": [{"id": "short_response", "path": "shared/model.py"}],
                "support_questions": [],
                "research_programs": [{"id": "old_short", "path": "shared/old_plan.py"}]}
    knowledge = {
        "manifest": manifest,
        "files": {"instrument.json": json.dumps(instrument),
                  "shared/model.py": revise_history(rows).to_code(),
                  "shared/old_plan.py": OLD_PLAN,
                  "shared/knowledge.json": json.dumps(manifest)},
        "lessons": ["The source response fits within three-step resets."],
    }
    task = {"agent_arm": "hegel", "task_position": 1, "status": "completed",
            "model": "shared/model.py", "paid_observations": records,
            "outgoing_knowledge": knowledge,
            "events": [{"action": "query", "arguments": {"path": "shared/old_plan.py"},
                        "feedback": {"observations": records}}],
            "fresh_confirmation": {"passed": True, "mse": 0.},
            "evaluation": {"private_terminal_labels": "NEVER_IN_LEARNER_TASK_EVALUATION"}}
    return {"arm": "hegel", "tasks": [task],
            "evaluation": [{"private_terminal_labels": "NEVER_IN_LEARNER_CHAIN_EVALUATION"}],
            "training_complete": True}


def consolidate(case, workspace):
    author = ScriptedMethodAuthor()
    experience = load_training_experience(case)
    agent = MethodConsolidationAgent(author, workspace, experience=experience,
        public_spec=TemporalInquiryLab(task_index=1).public_spec(),
        config=InquiryLLMConfig(max_model_calls=3, token_budget=1000, execution_seconds=20.))
    return author, agent, agent.run()


def test_consolidation_context_uses_paid_training_and_never_terminal_evaluation(tmp_path, experience_case):
    experience = load_training_experience(experience_case)
    assert "NEVER_IN_LEARNER" not in json.dumps(experience)
    assert "paid_observations" in experience
    assert experience["paid_observations"] == experience_case["tasks"][0]["paid_observations"]
    author, _, result = consolidate(experience_case, tmp_path)
    assert result["method_path"] == METHOD_PATH
    assert len(author.requests) == 2
    assert all("NEVER_IN_LEARNER" not in json.dumps(request) for request in author.requests)
    assert "Short resets" in result["lesson"]


def test_method_authoring_preserves_retained_predictors_instruments_and_old_plan(tmp_path, experience_case):
    before = deepcopy(experience_case)
    _, agent, result = consolidate(experience_case, tmp_path)
    old_files = experience_case["tasks"][0]["outgoing_knowledge"]["files"]
    for path, source in old_files.items():
        assert (agent.workspace / path).read_text() == source
    assert (agent.workspace / METHOD_PATH).read_text() == AUTHORED_METHOD
    assert result["method_source"] == AUTHORED_METHOD
    assert experience_case == before
    assert result["costs"]["observation_cost"] == 0
    assert result["costs"]["measurement_cost"] == 0


def test_retention_intervention_changes_only_new_procedure_and_manifest_entry(tmp_path, experience_case):
    _, _, result = consolidate(experience_case, tmp_path)
    original = experience_case["tasks"][0]["outgoing_knowledge"]
    original_before = deepcopy(original)
    retained, deleted = method_memory_pair(original, result)
    assert original == original_before
    assert retained["lessons"] == deleted["lessons"]
    assert retained["lessons"][-1] == LEARNER_LESSON
    assert retained["files"][METHOD_PATH] == AUTHORED_METHOD
    assert METHOD_PATH not in deleted["files"]
    for path in ("instrument.json", "shared/model.py", "shared/old_plan.py"):
        assert retained["files"][path] == deleted["files"][path] == original["files"][path]
    erased = deepcopy(retained)
    erased["files"].pop(METHOD_PATH)
    erased["manifest"]["research_programs"] = [entry for entry in erased["manifest"]["research_programs"]
                                               if entry["path"] != METHOD_PATH]
    # The ordinary on-disk manifest may mirror the same one-entry intervention.
    if "shared/knowledge.json" in erased["files"]:
        payload = json.loads(erased["files"]["shared/knowledge.json"])
        payload["research_programs"] = [entry for entry in payload["research_programs"]
                                        if entry["path"] != METHOD_PATH]
        assert payload == json.loads(deleted["files"]["shared/knowledge.json"])
        erased["files"]["shared/knowledge.json"] = deleted["files"]["shared/knowledge.json"]
    assert erased == deleted


def test_authored_method_executes_prospectively_and_deleted_control_keeps_old_plan(tmp_path, experience_case):
    _, _, result = consolidate(experience_case, tmp_path / "author")
    retained, deleted = method_memory_pair(experience_case["tasks"][0]["outgoing_knowledge"], result)
    agents = {}
    for condition, memory in (("retained", retained), ("deleted", deleted)):
        agent = TemporalOperatorAgent(None, tmp_path / condition, arm="hegel", knowledge=memory,
            source_task=False, config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                                                       execution_seconds=30., compute_budget=100000))
        lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
        agents[condition] = agent, lab
    agent, lab = agents["retained"]
    feedback = agent._execute("query", {"path": METHOD_PATH}, lab)
    assert feedback["inherited_program"] and feedback["inherited_source_unchanged"]
    assert feedback["continuation_origin"] == METHOD_PATH
    assert len(feedback["observations"]) == 3
    assert sum(len(record["rows"]) for record in feedback["observations"]) == 72
    assessment, response = feedback["after_observation_results"]
    assert assessment["records"] == feedback["record_indices"] == [0, 1, 2]
    assert len(response["candidates"]) == 1
    assert response["candidates"][0]["validation_mse"] < 1e-10
    assert agent.model is None
    evaluation = agent._execute("evaluate", {"model": response["candidates"][0]["path"]}, lab)
    assert evaluation["returncode"] == 0 and evaluation["mse"] < 1e-10

    control, control_lab = agents["deleted"]
    missing = control._execute("query", {"path": METHOD_PATH}, control_lab)
    assert missing["returncode"] != 0
    assert control_lab.costs["observation_cost"] == 0
    old = control._execute("query", {"path": "shared/old_plan.py"}, control_lab)
    assert old["inherited_program"] and old["inherited_source_unchanged"]
    assert sum(len(record["rows"]) for record in old["observations"]) == 3
    assert not old.get("after_observation_results")


def test_study_reports_all_failed_transfers_and_defers_terminal_evaluation(
        tmp_path, experience_case, monkeypatch):
    from scripts import run_temporal_method_learning as runner

    experience_case["tasks"][0]["costs"] = {
        "observation_cost": 36, "measurement_cost": 108, "compute_cost": 100,
        "python_execution_seconds": .1, "model_calls": 0, "reported_tokens": 0,
    }
    calls, evaluation_order = [], []

    class FailedTransfer:
        def __init__(self, workspace):
            self.workspace = workspace

        def next_action(self, request):
            calls.append(str(self.workspace))
            return {"action": "provider_error", "arguments": {},
                    "error": {"status": 503, "message": "scripted unavailable provider"},
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 2}}

    def provider(path):
        return ScriptedMethodAuthor() if path.name == "consolidation" else FailedTransfer(path)

    original_evaluate = runner.evaluate_chain

    def evaluate_only_after_training(chain, tasks, **kwargs):
        assert len(calls) == 4
        assert len(list(tmp_path.glob("changed_*/*/training.json"))) == 4
        evaluation_order.append((chain["case_id"], chain["condition"]))
        return original_evaluate(chain, tasks, **kwargs)

    monkeypatch.setattr(runner, "evaluate_chain", evaluate_only_after_training)
    summary = runner.run_method_study(experience_case, tmp_path,
                                      provider_factory=provider, workers=2)
    assert summary["status"] == "completed"
    assert summary["evidence_kind"] == "interface_test_not_llm_performance"
    assert len(evaluation_order) == 4
    assert summary["reported_tokens_new_work"] == 10
    assert summary["common_learning_costs"]["prior"]["observation_cost"] == 36
    for condition in runner.CONDITIONS:
        result = summary["conditions"][condition]
        assert result["registered_tasks"] == result["attempted_tasks"] == 2
        assert result["statuses"] == ["provider_failed", "provider_failed"]
        assert result["confirmed_and_correct"] == result["false_confirmations"] == 0
        assert result["transfer_costs"]["reported_tokens"] == 4
        assert result["transfer_costs"]["observation_cost"] == 0
        assert result["costs_including_common_learning"]["reported_tokens"] == 6
    for path in tmp_path.glob("changed_*/*/result.json"):
        assert json.loads(path.read_text())["evaluation"][0]["status"] == "missing_model"


def test_empty_initial_method_is_rejected_then_corrected_without_acquiring_data(tmp_path, experience_case):
    bad_method = """def select_actions(spec, observations, memory):
    if spec.get('observation_budget', 0) < 1:
        return []
    return {'actions':[{'kind':'trajectory','sequence':[{'u':.5}],'channels':[]}]}
"""

    class CorrectingAuthor:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            assert OPERATOR_INTERFACE in request["task_interface"]
            assert "observation_budget" not in request["capabilities"]
            step = self.calls
            self.calls += 1
            if step in (0, 2):
                if step == 2:
                    assert request["events"][-1]["action"] == "finish"
                    assert request["events"][-1]["feedback"]["error"]
                action, arguments = "write", {"path": METHOD_PATH,
                                              "code": bad_method if step == 0 else AUTHORED_METHOD}
            else:
                action, arguments = "finish", {"path": METHOD_PATH,
                    "entry": {"id": "corrected_method", "path": METHOD_PATH}, "lesson": LEARNER_LESSON}
            return {"action": action, "arguments": arguments,
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}

    author, checkpoints = CorrectingAuthor(), []
    agent = MethodConsolidationAgent(author, tmp_path,
        experience=load_training_experience(experience_case),
        public_spec=TemporalInquiryLab(task_index=1).public_spec(),
        config=InquiryLLMConfig(max_model_calls=4, token_budget=1000, execution_seconds=20.))
    result = agent.run(on_checkpoint=lambda data: checkpoints.append(deepcopy(data)))
    assert author.calls == 4
    assert checkpoints[1]["method_path"] is None
    assert checkpoints[1]["events"][-1]["feedback"]["error"]
    assert result["status"] == "completed"
    assert result["method_source"] == AUTHORED_METHOD
    assert result["costs"]["observation_cost"] == result["costs"]["measurement_cost"] == 0
    assert len(agent.observations) == len(experience_case["tasks"][0]["paid_observations"])
    for path, source in experience_case["tasks"][0]["outgoing_knowledge"]["files"].items():
        assert (agent.workspace / path).read_text() == source


def test_publication_initializes_documented_research_state(tmp_path, experience_case):
    method = """import json
def select_actions(spec, observations, memory):
    with open('research_state.json') as stream:
        state = json.load(stream)
    assert observations == [] and state['operator_results'] == []
    assert state['costs']['observation_cost'] == 0
    if state['limits']['observation_budget'] > state['costs']['observation_cost']:
        return {'actions':[{'kind':'trajectory','sequence':[{'u':.5}],'channels':[]}]}
    return {'actions':[]}
"""
    agent = MethodConsolidationAgent(None, tmp_path,
        experience=load_training_experience(experience_case),
        public_spec=TemporalInquiryLab(task_index=1).public_spec())
    agent._execute("write", {"path": METHOD_PATH, "code": method})
    feedback = agent._execute("finish", {"path": METHOD_PATH, "entry": {"id": "state_aware"}})
    assert feedback.get("published") == METHOD_PATH, feedback
    assert agent.method_source == method
    assert agent.costs["observation_cost"] == agent.costs["measurement_cost"] == 0
