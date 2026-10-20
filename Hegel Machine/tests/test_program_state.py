"""Research programs receive one complete state without intermediate JSON copies."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig
from hegel_machine.cognition import online_inquiry_llm
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from hegel_machine.cognition.temporal_paired_method import FrozenMethodAgent, ProcedureConsumerAgent
from test_inquiry_revision import PublicLab, action


PROGRAM = "def select_actions(spec, observations, memory): return []\n"


@pytest.mark.parametrize("agent_class", [
    InquiryLLMAgent, TemporalOperatorAgent, ProcedureConsumerAgent,
    CommonArchiveInquiryAgent, FrozenMethodAgent, AnonymousPreparedMethodAgent,
])
def test_program_state_is_encoded_and_written_once_with_all_layer_fields(
        tmp_path, monkeypatch, agent_class):
    manifest = {"research_programs": [
        {"id": "probe", "path": "shared/probe.py", "max_steps": 3}],
        "hypotheses": [], "support_questions": []}
    knowledge = {"files": {"shared/probe.py": PROGRAM,
                            "shared/knowledge.json": json.dumps(manifest)},
                 "manifest": manifest}
    config = InquiryLLMConfig(observation_budget=12, confirmation_rows=2,
                              execution_seconds=30., max_model_calls=8)
    if agent_class is FrozenMethodAgent:
        agent = agent_class(tmp_path, knowledge=knowledge, method_source=PROGRAM, config=config)
    else:
        agent = agent_class(None, tmp_path, arm="hegel", source_task=False,
                            knowledge=knowledge, config=config)
    agent._execute("query", {"actions": [action(2)]}, PublicLab())
    question = agent._open_question({"id": "local", "question": "Read the paid response",
                                    "parent_id": None, "depends_on": []})
    agent.revisions.append({"index": 0, "question_id": "local", "status": "observed",
        "proposal": {"id": "saved_readout"}, "record_indices": [0],
        "observations": deepcopy(agent.observations), "analysis_is_agent_authored": True,
        "returned_answer": {"evidence": [0], "answer": "Retain this local result"}})
    agent.evaluated_model = {"model": "fit.py", "source_model": "shared/fit.py",
                             "current_mse": .25, "evaluated_rows": 1,
                             "evaluation_scope": "paid_fit"}
    expected = {"questions": deepcopy(agent.questions), "revisions": deepcopy(agent.revisions),
                "limits": asdict(config), "costs": deepcopy(agent.costs),
                "evaluated_model": {"model": "_evaluated_model/fit.py",
                    "source_model": "shared/fit.py", "mse": .25,
                    "evaluated_rows": 1, "current_paid_rows": 1, "support": "paid_fit",
                    "evaluation_scope": "paid_fit", "fitting_view": {"scope": "unknown"}}}
    if isinstance(agent, TemporalOperatorAgent):
        agent.operator_results = [{"index": 0, "operator": "refit",
                                   "status": "refit_inadequate_on_validation"}]
        agent.candidate_paths = ["shared/candidate.py"]
        agent._current_research_advice = {"status": "needs_research", "current_paid_rows": 1}
        agent.legacy_session = SimpleNamespace(session=SimpleNamespace(state={
            "results": [{"task_position": 2, "research": {"status": "inconclusive"}}],
            "pending_revisits": [{"task_position": 2}]}))
        expected.update(operator_results=deepcopy(agent.operator_results),
            exported_candidate_paths=["shared/candidate.py"],
            research_advice=deepcopy(agent._current_research_advice),
            legacy_session={"tasks": [{"task_position": 2, "status": "inconclusive"}],
                            "pending_revisits": [2]})
    if isinstance(agent, ProcedureConsumerAgent):
        agent.method_results = [{"path": "shared/probe.py", "step": 0,
            "proposal": {"model": "shared/fit.py"},
            "feedback": {"fresh_confirmation": {"passed": False, "returncode": 0,
                "mse": .25, "rows": 1, "observation_indices": [0]}}}]
        expected["method_results"] = deepcopy(agent.method_results)
        expected["program_contexts"] = deepcopy(agent.program_contexts)
    if isinstance(agent, CommonArchiveInquiryAgent):
        agent._active_question_id = question["id"]
        agent.experiment_comparisons = [{"index": 0, "status": "compared",
                                         "prior_fit": [], "comparisons": []}]
        agent.observable_models = {"sensor": {"model": "shared/sensor.py"}}
        agent.conditional_models = {"branch": {"model": "shared/branch.py"}}
        agent.model_dependencies = [{"model": "shared/fit.py", "question_id": "local"}]
        agent.research_candidates = {"shared/fit.py": {"status": "evaluated"}}
        agent.research_input_bindings = {"parameter": {"question_id": "local"}}
        agent.relation_research = {"relation": {"status": "inconclusive"}}
        expected.update({name: deepcopy(getattr(agent, name)) for name in (
            "experiment_comparisons", "observable_models", "conditional_models",
            "model_dependencies", "research_candidates", "research_input_bindings",
            "relation_research")})
        expected["online_research"] = agent._online_research()

    expected = json.loads(json.dumps(expected))  # Observation histories become JSON arrays.
    state_path = agent.workspace / "research_state.json"
    state_path.write_text("unfinished old state", encoding="utf-8")
    original_read, original_dumps = Path.read_text, json.dumps
    original_write = online_inquiry_llm.write_text_incrementally
    encoded, written = [], []

    def read(path, *args, **kwargs):
        assert path != state_path, "state layers must not reload an intermediate state"
        return original_read(path, *args, **kwargs)

    def write(path, content, *args, **kwargs):
        if path == state_path:
            written.append(content)
        return original_write(path, content, *args, **kwargs)

    def dumps(value, *args, **kwargs):
        if isinstance(value, dict) and {"questions", "limits", "costs"} <= value.keys():
            encoded.append(value)
        return original_dumps(value, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_text", read)
        patched.setattr(online_inquiry_llm, "write_text_incrementally", write)
        patched.setattr(json, "dumps", dumps)
        agent._prepare_program_state()

    assert len(encoded) == len(written) == 1
    assert json.loads(state_path.read_text(encoding="utf-8")) == expected
    if isinstance(agent, CommonArchiveInquiryAgent):
        methods = json.loads((agent.workspace / "method_decision.json").read_text())["methods"]
        assert len(methods) == 1
        method = methods[0]
        assert method["execute_action"] == {"action": "run", "arguments": {
            "operator": "procedure", "path": "shared/probe.py", "max_steps": 3, "confirm": True}}
        assert method["current_executions"] == agent.method_results
        assert method["source_unchanged"] is True
        assert method["continuation"]["status"] == "prediction_refuted"
        assert method["continuation"]["current_paid_rows"] == 1
