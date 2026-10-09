"""Direct consumer submission still requires actual paid fresh confirmation."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent
from test_temporal_paired_method import calibrated_knowledge


SELECTED = "shared/selected_response.py"
MANIFEST = "shared/knowledge.json"
OLD_PLAN = """def select_actions(spec, records, memory):
    return {'actions':[{'kind':'trajectory','sequence':[{'u':.25}],'channels':[]}]}
"""


def knowledge_with_old_manifest(calibrated, *, selected_is_correct):
    knowledge = deepcopy(calibrated)
    prior = knowledge["research_roles"]["predictor"]
    knowledge["files"][SELECTED] = (knowledge["files"][prior] if selected_is_correct
                                      else "def predict(row):\n    return 0.\n")
    knowledge["files"]["shared/old_plan.py"] = OLD_PLAN
    manifest = knowledge["manifest"]
    manifest["hypotheses"][0].update(status="previously_confirmed", note="Retain the earlier evidence.")
    manifest["research_programs"] = [{"id": "old_plan", "path": "shared/old_plan.py",
                                      "applicability": "short initial probe", "notes": ["paid source method"]}]
    manifest["annotations"] = {"interpretation": "Prior support does not establish current applicability."}
    return knowledge


def consumer(workspace, knowledge, *, source_task=False):
    return ProcedureConsumerAgent(None, workspace, arm="hegel", knowledge=knowledge,
        source_task=source_task, confirmation_seed=195,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               compute_budget=100000, execution_seconds=30.))


@pytest.mark.parametrize("selected_is_correct", [True, False])
def test_direct_submission_adds_only_selected_path_and_freshly_tests_it(
        tmp_path, calibrated_knowledge, selected_is_correct):
    knowledge = knowledge_with_old_manifest(calibrated_knowledge,
                                           selected_is_correct=selected_is_correct)
    expected_manifest = deepcopy(knowledge["manifest"])
    if selected_is_correct:
        # Current on-disk annotations take precedence over the inherited copy.
        expected_manifest["annotations"]["current_note"] = "Preserve this current research note."
        knowledge["files"][MANIFEST] = json.dumps(expected_manifest)
    agent = consumer(tmp_path / "consumer", knowledge)
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")
    assert agent.rows == [] and agent.costs["observation_cost"] == 0
    outcome = agent._execute("finish", {"model": SELECTED}, lab)
    assert outcome["fresh_confirmation"]["rows"] == 48
    assert outcome["fresh_confirmation"]["passed"] is selected_is_correct
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 48
    assert agent.costs["compute_cost"] == 48
    assert agent.costs["measurement_cost"] == agent.costs["model_calls"] == 0
    assert sum(len(record["rows"]) for record in agent.observations) == 48

    submitted = json.loads((agent.workspace / MANIFEST).read_text())
    assert submitted["research_programs"] == expected_manifest["research_programs"]
    assert submitted["annotations"] == expected_manifest["annotations"]
    assert submitted["hypotheses"][:-1] == expected_manifest["hypotheses"]
    entry = submitted["hypotheses"][-1]
    assert entry["path"] == SELECTED
    assert set(entry) == {"id", "path"}
    assert (agent.workspace / "shared/old_plan.py").read_text() == OLD_PLAN
    assert agent.knowledge == knowledge
    if selected_is_correct:
        assert agent.model == SELECTED
        assert outcome["fresh_confirmation"]["mse"] < 1e-10
    else:
        assert agent.model is None
        assert outcome["fresh_confirmation"]["mse"] > 1e-4
        assert "fresh confirmation failed" in outcome["error"]


def test_source_learning_keeps_explicit_registration_and_executed_question_requirements(
        tmp_path, calibrated_knowledge):
    knowledge = knowledge_with_old_manifest(calibrated_knowledge, selected_is_correct=True)
    manifest_source = json.dumps(knowledge["manifest"])
    knowledge["files"][MANIFEST] = manifest_source
    agent = consumer(tmp_path / "source", knowledge, source_task=True)
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")

    unregistered = agent._execute("finish", {"model": SELECTED}, lab)
    assert unregistered["error"] == "a hypothesis path must identify the actual submitted model"
    registered = agent._execute("finish", {"model": knowledge["research_roles"]["predictor"]}, lab)
    assert "source learning must actually execute" in registered["error"]
    assert (agent.workspace / MANIFEST).read_text() == manifest_source
    assert agent.model is None
    assert "fresh_confirmation" not in unregistered and "fresh_confirmation" not in registered
    assert agent.costs["observation_cost"] == agent.costs["compute_cost"] == 0
