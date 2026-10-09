"""Existing paid research must survive the LLM tool protocol unchanged."""

from copy import deepcopy
from dataclasses import asdict
import json
from random import Random
import runpy

import pytest

from hegel_machine.cognition.legacy_research_bridge import (
    LEGACY_KNOWLEDGE_PATH, METHODS_PATH, InquiryResearchLab, run_legacy_research,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


def make_agent(path, *, knowledge=None, short=False):
    agent = TemporalOperatorAgent(None, path, arm="code_react", source_task=False,
        knowledge=knowledge, config=InquiryLLMConfig(observation_budget=384,
            compute_budget=360000, execution_seconds=120, confirmation_rows=24))
    lab = TemporalInquiryLab(task_index=0 if short else 1, apparatus_seed=910061,
                            instrument_mode="finite2")
    return agent, lab


def test_legacy_loop_exports_real_predictor_and_charges_current_inquiry(tmp_path):
    agent, lab = make_agent(tmp_path)
    result = agent._execute("run", {"operator": "legacy_research"}, lab)
    assert result["operator"] == "legacy_research"
    assert result["python_execution_seconds"] > 0
    assert agent.costs["python_execution_seconds"] == result["python_execution_seconds"]
    assert agent.operator_results[-1]["operator"] == "legacy_research"
    assert agent.operator_results[-1]["compute_cost"] == result["compute_cost"]
    assert result["legacy_status"] == "completed"
    assert result["path"] and not result["export_reason"]
    assert result["requires_agent_confirmation"] and not agent.confirmations
    assert agent.model is None and agent.costs["model_calls"] == 0
    assert result["new_costs"]["observation_cost"] == lab.costs["observation_cost"]
    assert result["new_costs"]["measurement_cost"] == lab.costs["measurement_cost"]
    assert result["new_costs"]["compute_cost"] == agent.costs["compute_cost"] > 0
    report = json.loads((tmp_path / result["report_path"]).read_text())
    assert report["research_graph"]
    assert "compare_instrument_replacement" in result["actions"]
    predictor = runpy.run_path(str(tmp_path / result["path"]))["predict"]
    rows = lab.evaluation_rows(length=48, seed=104093)
    assert sum((predictor(asdict(row)) - row.target) ** 2 for row in rows) / len(rows) < 1e-7
    confirmation = agent._execute("finish", {"model": result["path"]}, lab)
    assert confirmation["fresh_confirmation"]["passed"]


def test_existing_paid_trajectories_are_used_without_purchasing_them_again(tmp_path):
    agent, lab = make_agent(tmp_path)
    rng = Random(71)
    actions = [{"kind": "trajectory", "sequence": [{"u": rng.uniform(-1, 1)} for _ in range(24)],
                "channels": ["meter"]} for _ in range(3)]
    agent._query({"actions": actions}, lab)
    paid = deepcopy(agent.observations)
    costs = deepcopy(agent.costs)
    result = run_legacy_research(agent, {"observation_budget": 96, "compute_budget": 120000}, lab)
    assert result["paid_rows_reused"] == 72
    assert agent.observations[:3] == paid
    assert result["actions"][0] == "import_paid_observations"
    assert result["new_costs"]["observation_cost"] <= 96
    assert agent.costs["observation_cost"] == 72 + result["new_costs"]["observation_cost"]
    assert agent.costs["measurement_cost"] == costs["measurement_cost"] + result["new_costs"]["measurement_cost"]
    report = json.loads((tmp_path / result["report_path"]).read_text())
    assert [batch["probe"]["sequence"] for batch in report["batches"][:3]] == [
        action["sequence"] for action in actions]
    assert report["continuation"]["paid_rows_reused"] == 72
    assert all(new["action"]["sequence"] not in [action["sequence"] for action in actions]
               for new in agent.observations[3:])
    assert agent.model is None and not agent.confirmations


def test_complete_legacy_library_is_shared_and_consumed_on_next_task(tmp_path):
    source, lab = make_agent(tmp_path / "source")
    learned = run_legacy_research(source, {}, lab)
    saved = json.loads(source.files[LEGACY_KNOWLEDGE_PATH])
    assert saved["concepts"] and saved["instrument_models"] and saved["research_workflows"]["methods"]
    assert json.loads(source.files[METHODS_PATH]) == saved["program_experiment_methods"]
    inherited = {"files": {name: source.files[name] for name in (LEGACY_KNOWLEDGE_PATH, METHODS_PATH)}}
    target, target_lab = make_agent(tmp_path / "target", knowledge=inherited)
    result = run_legacy_research(target, {}, target_lab)
    assert set(learned["concepts_added"]) <= set(result["available_concepts"])
    assert result["used_sources"]
    assert set(result["used_sources"]) <= set(result["available_concepts"])
    restored = KnowledgeLibrary.from_dict(json.loads(target.files[LEGACY_KNOWLEDGE_PATH]))
    assert {item.concept_id for item in restored.concepts} >= set(learned["concepts_added"])
    assert result["legacy_status"] == "completed"
    assert result["new_costs"]["observation_cost"] < learned["new_costs"]["observation_cost"]
    assert not target.confirmations and target.model is None


def test_measured_model_is_not_exported_as_control_only_predictor(tmp_path):
    agent, lab = make_agent(tmp_path, short=True)
    result = run_legacy_research(agent, {}, lab)
    assert result["legacy_status"] == "completed"
    assert result["selected_channels"] == ["meter"]
    assert result["path"] is None
    assert "only control" in result["export_reason"]
    assert not (tmp_path / "shared/legacy_model.py").exists()


def test_public_adapter_preserves_paid_measurements_and_reset_history(tmp_path):
    agent, lab = make_agent(tmp_path)
    adapter = InquiryResearchLab(agent, lab)
    probe = Probe("one", ({"u": .8}, {"u": -.3}, {"u": .5}))
    measured = adapter.acquire(probe, ("meter",))
    record = agent.observations[-1]
    assert measured.observation_cost == 3 and measured.measurement_cost == 9
    assert len(measured.observations) == 3
    assert measured.observations[0].history == ()
    assert measured.observations[1].history[0] == {
        **measured.observations[0].features, "__target__": measured.observations[0].target}
    assert [row.features["meter"] for row in measured.observations] == [
        item["values"]["meter"] for item in record["measurements"]]
    for row in agent.rows:
        assert set(row.features) == {"u"}
        assert all(set(past) == {"u"} for past in row.history)


def test_no_budget_is_not_an_unpaid_invocation(tmp_path):
    agent, lab = make_agent(tmp_path)
    agent.costs["compute_cost"] = agent.config.compute_budget
    with pytest.raises(ValueError, match="allowance"):
        run_legacy_research(agent, {}, lab)
    assert not agent.observations and lab.costs["observation_cost"] == 0
