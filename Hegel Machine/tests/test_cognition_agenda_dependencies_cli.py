"""The public command continues and edits a real dependencyful research agenda."""

import json

from hegel_machine.cognition.cli import main


def test_cli_dependency_request_changes_the_next_actual_research(tmp_path, capsys):
    checkpoint = tmp_path / "agenda.json"
    request_path = tmp_path / "request.json"

    def question(identifier):
        return {"question_id": identifier, "kind": "task",
                "source": {"kind": "suite", "seed": 71, "partition": "development", "task_index": 0},
                "config": {"seed": 71, "observation_budget": 32, "compute_budget": 6000, "max_rounds": 2},
                "max_visits": 1}

    def run(request=None, steps=1):
        arguments = ["cognition", "agenda", "--checkpoint", str(checkpoint), "--steps", str(steps)]
        if request is not None:
            request_path.write_text(json.dumps(request))
            arguments += ["--request", str(request_path)]
        assert main(arguments) == 0
        capsys.readouterr()
        return json.loads(checkpoint.read_text())

    first = run({"questions": [question("consumer"), question("source")],
                 "dependencies": [{"question_id": "consumer", "depends_on": ["source"]}]})
    assert [visit["question_id"] for visit in first["visits"]] == ["source"]
    assert first["questions"]["consumer"]["visits"] == 0

    revised = run({"questions": [question("replacement")],
                   "dependencies": [{"question_id": "consumer", "depends_on": ["replacement"]}]})
    assert [visit["question_id"] for visit in revised["visits"]] == ["source", "replacement"]
    assert revised["questions"]["consumer"]["visits"] == 0

    final = run()
    assert [visit["question_id"] for visit in final["visits"]] == ["source", "replacement", "consumer"]
    assert final["questions"]["consumer"]["result"]["validation_loss"] < 1e-10
    assert final["spent"]["observation_cost"] == sum(visit["costs"]["observation_cost"] for visit in final["visits"])
    assert final["spent"]["compute_cost"] == sum(visit["costs"]["compute_cost"] for visit in final["visits"]) + final.get("evaluator_compute_cost", 0)
