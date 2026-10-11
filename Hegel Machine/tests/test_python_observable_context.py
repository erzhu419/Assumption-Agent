"""Paid channel alignment and executable target-specific fitting contexts."""

from dataclasses import asdict
import json
import shutil

import pytest

from hegel_machine.cognition.code_agent import evaluate_python_model, retain_evaluated_python_model
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


def test_target_rows_align_paid_channel_indices_and_skip_unmeasured_records():
    lab = TemporalInquiryLab(task_index=1)
    records = [lab.execute({"kind": "trajectory", "sequence": [{"u": u} for u in [.2, -.4]],
                            "channels": channels}) for channels in ([], ["meter"])]
    records[1]["measurements"].reverse()
    before = lab.costs
    selected = lab.target_rows(records, "meter")
    labels = {row["observation_index"]: row["values"]["meter"]
              for row in records[1]["measurements"]}
    assert [row.target for row in selected] == [labels[index] for index in records[1]["row_indices"]]
    assert [(row.features, row.history) for row in selected] == [
        (row.features, row.history) for row in lab.observation_rows(records[1])]
    assert len(lab.target_rows(records)) == 4
    assert lab.target_rows(records[:1], "meter") == []
    assert lab.costs == before
    with pytest.raises(ValueError, match="unknown public observation target"):
        lab.target_rows(records, "hidden_state")


def prepare_model(workspace):
    (workspace / "model.py").write_text(
        "value = None\n"
        "def fit(rows):\n    global value\n    value = sum(r['target'] for r in rows) / len(rows)\n"
        "def predict(row):\n    return value\n")
    (workspace / "observations.json").write_text(json.dumps([asdict(Observation({"u": 0.}, 100.))]))


def test_explicit_target_training_changes_fit_without_overwriting_response_data(tmp_path):
    prepare_model(tmp_path)
    rows = [Observation({"u": 0.}, 2.)]
    ordinary = evaluate_python_model(tmp_path, "model.py", rows)
    selected = evaluate_python_model(tmp_path, "model.py", rows, training_observations=rows)
    assert ordinary["mse"] == 98. ** 2
    assert selected["mse"] == 0.
    assert json.loads((tmp_path / "observations.json").read_text())[0]["target"] == 100.
    assert not (tmp_path / "model_observations.json").exists()


def test_target_retention_preserves_fit_data_and_excludes_its_own_archive(tmp_path):
    prepare_model(tmp_path)
    (tmp_path / "_target_work").mkdir()
    (tmp_path / "_target_work" / "temporary.py").write_text("temporary = True\n")
    rows = [Observation({"u": 0.}, 2.)]
    destination = "_observable_models/meter/_evaluated_model"
    retained = retain_evaluated_python_model(tmp_path, "model.py", [], 0.,
        destination=destination, training_observations=rows)
    snapshot = tmp_path / destination
    assert not (snapshot / "_observable_models").exists()
    assert not (snapshot / "_target_work").exists()
    assert (snapshot / "observations.json").read_text() == (snapshot / "model_observations.json").read_text()
    (tmp_path / "observations.json").write_text('[{"target":999}]')
    (tmp_path / "model.py").unlink()
    result = evaluate_python_model(tmp_path, f"{destination}/{retained['model']}", rows)
    assert result["mse"] == 0.


def test_parent_snapshot_keeps_executable_observable_dependency(tmp_path):
    prepare_model(tmp_path)
    rows = [Observation({"u": 0.}, 2.)]
    retain_evaluated_python_model(tmp_path, "model.py", [], 0.,
        destination="_observable_models/meter/_evaluated_model", training_observations=rows)
    (tmp_path / "parent.py").write_text(
        "def bind_models(predictors):\n    global child\n    child = predictors['auxiliary']\n"
        "def predict(row):\n    return child(row) + 1.\n")
    child_path = "_observable_models/meter/_evaluated_model/model.py"
    (tmp_path / "model_bindings.json").write_text(json.dumps({"parent.py": {"auxiliary": child_path}}))
    retain_evaluated_python_model(tmp_path, "parent.py", [], 0.)
    shutil.rmtree(tmp_path / "_observable_models")
    (tmp_path / "parent.py").unlink()
    result = evaluate_python_model(tmp_path, "_evaluated_model/parent.py",
                                   [Observation({"u": 0.}, 3.)])
    assert result["returncode"] == 0, result["stderr"]
    assert result["mse"] == 0.
    assert result["model_dependencies"] == [
        {"parent": "parent.py", "slot": "auxiliary", "model": child_path, "calls": 1}]


@pytest.mark.parametrize("uses_child", [True, False])
def test_binding_preserves_child_cwd_and_counts_actual_calls(tmp_path, uses_child):
    prepare_model(tmp_path)
    (tmp_path / "factor.json").write_text("2")
    (tmp_path / "model.py").write_text((tmp_path / "model.py").read_text().replace(
        "return value", "return value * __import__('json').load(open('factor.json'))"))
    retain_evaluated_python_model(tmp_path, "model.py", [], 0.,
        destination="_observable_models/meter/_evaluated_model",
        training_observations=[Observation({"u": 0.}, 2.)])
    (tmp_path / "factor.json").write_text("999")
    (tmp_path / "parent.py").write_text(
        "def bind_models(models):\n    global child\n    child = models['auxiliary']\n"
        "def fit(rows):\n    global offset\n    offset = rows[0]['target']\n"
        f"def predict(row):\n    return offset + {'child(row)' if uses_child else '4.'}\n")
    (tmp_path / "model_bindings.json").write_text(json.dumps({
        "parent.py": {"auxiliary": "_observable_models/meter/_evaluated_model/model.py"}}))
    result = evaluate_python_model(tmp_path, "parent.py", [Observation({"u": 0.}, 9.)],
                                  training_observations=[Observation({"u": 0.}, 5.)])
    assert result["returncode"] == 0, result["stderr"]
    assert result["mse"] == 0.
    assert result["model_dependencies"][0]["calls"] == int(uses_child)


def test_target_revision_retains_its_selected_old_target_dependency(tmp_path):
    prepare_model(tmp_path)
    old_directory = "_observable_models/meter/0/_evaluated_model"
    retain_evaluated_python_model(tmp_path, "model.py", [], 0., destination=old_directory,
                                 training_observations=[Observation({"u": 0.}, 2.)])
    (tmp_path / "revision.py").write_text(
        "def bind_models(models):\n    global old\n    old = models['old']\n"
        "def predict(row):\n    return old(row) + 1.\n")
    (tmp_path / "model_bindings.json").write_text(json.dumps({
        "revision.py": {"old": old_directory + "/model.py"}}))
    new_directory = "_observable_models/meter/1/_evaluated_model"
    rows = [Observation({"u": 0.}, 3.)]
    retain_evaluated_python_model(tmp_path, "revision.py", [], 0., destination=new_directory,
                                 training_observations=rows)
    assert not (tmp_path / new_directory / "_observable_models/meter/1").exists()
    shutil.rmtree(tmp_path / "_observable_models/meter/0")
    result = evaluate_python_model(tmp_path, new_directory + "/revision.py", rows)
    assert result["returncode"] == 0, result["stderr"]
    assert result["mse"] == 0.
    assert result["model_dependencies"][0]["calls"] == 1
