import json

from hegel_machine.cognition.cli import main


def test_code_sequence_can_be_created_and_resumed_without_starting_a_model(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"tasks": [{"task_index": 3, "seed": 859}]}))
    checkpoint, workspace = tmp_path / "sequence.json", tmp_path / "workspace"
    arguments = ["cognition", "code-session", "--checkpoint", str(checkpoint),
                 "--workspace", str(workspace), "--steps", "0"]
    assert main([*arguments, "--request", str(request)]) == 0
    first = json.loads(checkpoint.read_text())
    assert first["code"]["results"] == [] and first["hegel"]["results"] == []
    assert first["code"]["costs"]["model_calls"] == 0
    assert main(arguments) == 0
    assert json.loads(checkpoint.read_text()) == first
    request.write_text(json.dumps({"new_tasks": [{"task_index": 1, "seed": 907}]}))
    assert main([*arguments, "--request", str(request)]) == 0
    appended = json.loads(checkpoint.read_text())
    assert appended["code"]["tasks"] == first["code"]["tasks"] + [{"task_index": 1, "seed": 907}]
    assert appended["code"]["costs"] == first["code"]["costs"]


def test_new_structure_question_runs_through_the_public_shared_agenda(tmp_path):
    request = tmp_path / "mechanism_request.json"
    request.write_text(json.dumps({"limits": {"observation_budget": 192,
        "compute_budget": 160000, "measurement_budget": 4096}, "questions": [{
            "question_id": "learn_projection", "kind": "mechanism",
            "source": {"kind": "mechanism", "family": "low_rank", "seed": 71},
            "config": {"observation_budget": 192, "compute_budget": 160000, "max_rounds": 1}}]}))
    checkpoint = tmp_path / "agenda.json"
    assert main(["cognition", "agenda", "--checkpoint", str(checkpoint),
                 "--request", str(request), "--steps", "1"]) == 0
    result = json.loads(checkpoint.read_text())
    question = result["questions"]["learn_projection"]
    assert question["status"] == "completed" and question["visits"] == 1
    assert question["result"]["episode"]["validation_rows"] > 0
    assert question["result"]["episode"]["validation_loss"] < 1e-8
    assert result["knowledge"]["structural_priors"]
    for cost in ("observation_cost", "compute_cost", "measurement_cost"):
        assert 0 < result["spent"][cost] <= result["limits"][cost]
    assert main(["cognition", "agenda", "--checkpoint", str(checkpoint), "--steps", "0"]) == 0
    assert json.loads(checkpoint.read_text())["spent"] == result["spent"]


def test_unfinished_mechanism_evidence_is_not_reported_as_a_completed_question(tmp_path):
    from hegel_machine.cognition.agenda import ResearchAgenda
    from hegel_machine.cognition.models import ResearchConfig

    agenda = ResearchAgenda(directory=tmp_path, observation_budget=4,
                            compute_budget=50, measurement_budget=20)
    agenda.add_mechanism("small", source={"kind": "mechanism", "family": "low_rank"},
        config=ResearchConfig(observation_budget=4, compute_budget=50), measurement_budget=20)
    result = agenda.run()
    assert result["questions"]["small"]["status"] == "unresolved"
    assert not result["questions"]["small"]["answer_concepts"]
    assert result["questions"]["small"]["result"]["episode"]["validation_loss"] is None


def test_simulator_research_resumes_actual_evidence_and_changes_the_experiment(tmp_path):
    checkpoint = tmp_path / "simulator.json"
    args = ["cognition", "simulator", "--checkpoint", str(checkpoint)]
    assert main([*args, "--steps", "2"]) == 0
    first = json.loads(checkpoint.read_text())
    assert first["status"] == "paused"
    assert main(args) == 0
    completed = json.loads(checkpoint.read_text())
    assert completed["status"] == "completed" and completed["selected_probe_changed"]
    assert completed["costs"]["observation_cost"] > first["costs"]["observation_cost"] > 0
    assert completed["simulator_state"]["current"]
    assert main(args) == 0
    assert json.loads(checkpoint.read_text()) == completed
