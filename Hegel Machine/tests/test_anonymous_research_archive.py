"""Archive binding must preserve fitting context and unsuccessful research."""

import json

from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.code_agent import python_model_context, python_model_context_matches


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_source(root, case, model, files, *, status="completed"):
    workspace = root / case / "hegel"
    write_json(workspace / "training.json", {
        "workspace": str(workspace), "status": status, "model": model,
        "fresh_confirmation": {"passed": status == "completed", "mse": 0.0, "rows": 96},
        "events": [{"action": "run", "arguments": {"path": "learn.py"},
                    "feedback": {"stdout": f"Read {workspace}/observations.json", "workspace": str(workspace)},
                    "usage": {"backend": "private_provider", "total_tokens": None}}],
    })
    write_json(workspace / "archive.json", {
        "files": files, "lessons": ["Author inferred persistence from the purchased path comparison."],
        "question_traces": [{"id": "q1", "question": "Does the old predictor still apply?", "evidence": [2]}],
    })


def materialize(root, knowledge):
    for name, contents in knowledge["files"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def test_protocol_sources_keep_raw_code_context_and_failures_without_evaluator_labels(tmp_path):
    source = tmp_path / "sources"
    write_json(source / "protocol.json", {
        "cases": [{"id": f"case_{index}", "kind": "private_kind", "seed": 123456789}
                  for index in range(3)], "arms": ["hegel"],
    })
    root_files = {"model.py": "def predict(row):\n    return 0.\n",
                  "observations.json": "[{\"target\": 1}]", "learn.py": "print('learned source')\n"}
    retained_files = {"model.py": "raise RuntimeError('superseded')\n",
                      "observations.json": "[{\"target\": 2}]",
                      "_evaluated_model/model.py": "def predict(row):\n    return 1.\n",
                      "_evaluated_model/observations.json": "[{\"target\": 3}]"}
    unfinished_files = {"draft.py": "def predict(row):\n    return 9.\n"}
    fixture_source(source, "case_0", "model.py", root_files)
    fixture_source(source, "case_1", "_evaluated_model/model.py", retained_files)
    fixture_source(source, "case_2", None, unfinished_files, status="provider_failed")
    write_json(source / "summary.json", {"terminal": {"private_terminal_truth": 998877}})
    original = {str(path.relative_to(source)): path.read_bytes()
                for path in source.rglob('*') if path.is_file()}

    knowledge, sources = load_research_archive(source)

    assert len(sources) == len(knowledge["source_catalog"]) == 3
    assert sources[-1]["status"] == "provider_failed"
    assert sources[-1]["model"] is None
    assert knowledge["research_roles"]["predictor"] == knowledge["research_roles"]["predictors"][0]
    assert len(knowledge["research_roles"]["predictors"]) == 2
    assert all("provider" not in item for item in sources)
    public = json.dumps(knowledge)
    assert all(hidden not in public for hidden in (
        "private_kind", "123456789", "private_terminal_truth", "998877", "private_provider", str(source)))
    assert "Author inferred persistence" in public
    for item, expected in zip(sources, (root_files, retained_files, unfinished_files)):
        for name, content in expected.items():
            assert knowledge["files"][f"{item['directory']}/{name}"] == content
        entry = next(row for row in knowledge["source_catalog"] if row["id"] == item["id"])
        trace = json.loads(knowledge["files"][entry["paid_trace"]])
        assert trace["events"][0]["feedback"] == {"stdout": "Read ./observations.json"}
        assert "usage" not in trace["events"][0]
    assert json.loads(knowledge["files"]["source_catalog.json"]) == knowledge["source_catalog"]

    reference, trial = tmp_path / "reference", tmp_path / "trial"
    materialize(reference, knowledge)
    materialize(trial, knowledge)
    for item, expected_target in zip(sources[:2], (1, 3)):
        context, inner = python_model_context(trial, item["model"])
        assert inner == item["inner_model"] == "model.py"
        assert context == trial / item["model_context_directory"]
        assert json.loads((context / "observations.json").read_text())[0]["target"] == expected_target
        assert python_model_context_matches(trial, item["model"],
            reference / item["model_context_directory"], item["inner_model"])
    (trial / sources[0]["directory"] / "model.py").write_text("def predict(row): return 5.\n")
    assert not python_model_context_matches(trial, sources[0]["model"],
        reference / sources[0]["model_context_directory"], sources[0]["inner_model"])
    assert original == {str(path.relative_to(source)): path.read_bytes()
                        for path in source.rglob('*') if path.is_file()}
