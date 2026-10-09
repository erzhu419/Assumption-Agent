"""Actual source repair can inform authoring without granting program support."""

import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import prepare_discoveryworld_transfer_method as consolidation


REVISED_SOURCE = "def select_actions(spec, observations, memory):\n    return {'stop': 'inspect current evidence'}\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def repaired_source(tmp_path):
    source = tmp_path / "source_study/tasks/source/hegel"
    source.mkdir(parents=True)
    learning = tmp_path / "source_learning"
    trial = learning / "author/trials/trial_040"
    trial.mkdir(parents=True)
    repair = tmp_path / "repair"
    workspace = repair / "agent"
    (workspace / "shared").mkdir(parents=True)
    (workspace / METHOD_PATH).write_text(REVISED_SOURCE)
    (workspace / "research_notes.md").write_text("Test public access before adjusting the device.")
    row = {"index": 0, "action": {"action": "READ", "arg1": 7},
           "feedback": {"success": True, "message": "Public instrument reading: 3"},
           "observation": {"lastActionMessage": "Public instrument reading: 3",
                           "agentLocation": {"x": 12, "y": 8, "faceDirection": "north"},
                           "taskProgress": [{"description": "Read and tune the instrument"}]},
           "tick": 1, "environment_steps": 1, "step_cost": 1, "agent_actions": 1}
    questions = [{"id": "procedure_repair_1", "parent_id": "task", "status": "supported",
                  "question": "Can the instrument be reached?", "answer": {"access": True},
                  "evidence": [0]}]
    costs = {"model_calls": 19, "environment_steps": 174, "reported_tokens": 1000}
    learning_costs = {"model_calls": 64, "environment_steps": 2000, "reported_tokens": 3000}
    write_json(learning / "protocol.json", {"source": str(source)})
    write_json(learning / "result.json", {"source": str(source), "costs": learning_costs})
    write_json(repair / "protocol.json", {"trial": str(trial)})
    write_json(workspace / "laboratory_observations.json", [row])
    result = {"status": "completed", "study_status": "finished", "costs": costs,
              "score": {"completed": True, "completedSuccessfully": True, "scoreNormalized": 0.4,
                        "criticalHypotheses": ["PRIVATE_EVALUATOR_RULE"]},
              "score_checkpoints": [{"criticalHypotheses": ["PRIVATE_EVALUATOR_RULE"]}],
              "research": {"questions": questions}, "events": [], "method_results": []}
    write_json(repair / "result.json", result)
    return {"source": source, "result": repair / "result.json", "learning": learning,
            "row": row, "questions": questions, "costs": costs, "learning_costs": learning_costs}


def test_actual_repair_exports_public_source_experience_and_retains_paid_provenance(repaired_source):
    fixture = repaired_source
    files, provenance = consolidation.repair_experience(fixture["result"], fixture["source"])
    assert files["shared/source_repaired_method.py"] == REVISED_SOURCE
    archive = json.loads(files["shared/source_repair.json"])
    assert archive["revised_source_path"] == "shared/source_repaired_method.py"
    assert archive["questions"] == fixture["questions"]
    assert "Test public access" in archive["notes"]
    assert "source" in archive["method_validation"].lower()
    # The actual world action is preserved through the existing public projection.
    projected = json.loads(json.dumps(DiscoveryWorldAgent._action_observation(fixture["row"])))
    assert len(archive["observations"]) == 1
    actual = archive["observations"][0]
    assert all(actual[key] == value for key, value in projected.items())
    assert actual["agentLocation"] == fixture["row"]["observation"]["agentLocation"]
    rendered = json.dumps(files)
    for hidden in ("PRIVATE_EVALUATOR_RULE", "criticalHypotheses", "scoreNormalized", "score_checkpoints"):
        assert hidden not in rendered
    assert provenance["costs"] == fixture["costs"]
    assert provenance["source_learning_result"] == str(fixture["learning"] / "result.json")
    assert provenance["source_learning_costs"] == fixture["learning_costs"]
    assert str(fixture["result"]) in provenance.values()


def test_unsuccessful_repair_cannot_be_exported_as_successful_source_experience(repaired_source):
    fixture = repaired_source
    result = json.loads(fixture["result"].read_text())
    result["score"]["completedSuccessfully"] = False
    write_json(fixture["result"], result)
    with pytest.raises(ValueError, match="(?i)success|complet"):
        consolidation.repair_experience(fixture["result"], fixture["source"])


def test_repair_from_another_development_source_is_rejected(repaired_source):
    fixture = repaired_source
    wrong_source = fixture["source"].parent / "different_source"
    with pytest.raises(ValueError, match="(?i)source"):
        consolidation.repair_experience(fixture["result"], wrong_source)


def test_child_repair_notes_survive_without_a_parent_notebook(repaired_source):
    fixture = repaired_source
    (fixture["result"].parent / "agent/research_notes.md").unlink()
    result = json.loads(fixture["result"].read_text())
    notes = {"procedure_repair_1": "Access repaired; use the numbered dialog.",
             "procedure_repair_2": "Exit the dialog and face the crystal before pickup."}
    result["research"]["llm_inquiries"] = {
        key: {"notes": value, "status": "answered", "model_calls": 2, "events": []}
        for key, value in notes.items()}
    write_json(fixture["result"], result)

    files, _ = consolidation.repair_experience(fixture["result"], fixture["source"])
    archive = json.loads(files["shared/source_repair.json"])
    assert archive["notes"] == ""
    assert archive["inquiry_notes"] == notes
    assert archive["questions"] == fixture["questions"]
    assert files[archive["revised_source_path"]] == REVISED_SOURCE
