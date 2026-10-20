"""Paid-data representation witnesses travel through real Python evaluation."""

from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.code_agent import evaluate_python_model, retain_evaluated_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


COLLAPSED = """
def represent(row):
    assert 'target' not in row
    return [0.0]
def fit_features(features, targets):
    global mean
    assert all(isinstance(value, list) for value in features)
    mean = sum(targets) / len(targets)
def predict_features(features):
    assert features == [0.0]
    return mean
def predict(row):
    raise AssertionError('must execute the exposed representation')
"""


def rows():
    return [{"features": {"u": 0.0}, "history": [{"u": 0.0}] * size,
             "target": float(size + 1)} for size in range(3)]


def write_model(tmp_path, code=COLLAPSED):
    (tmp_path / "model.py").write_text(code)
    (tmp_path / "observations.json").write_text(json.dumps(rows()))


def test_real_downstream_inputs_expose_irreducible_paid_error_and_authored_repair(tmp_path):
    write_model(tmp_path)
    collapsed = evaluate_python_model(tmp_path, "model.py", rows())
    assert collapsed["returncode"] == 0
    assert collapsed["mse"] == pytest.approx(2 / 3)
    diagnostic = collapsed["representation_diagnostic"]
    assert diagnostic["status"] == "conflicting_targets"
    assert diagnostic["empirical_mse_floor"] == pytest.approx(2 / 3)
    assert diagnostic["representation_contract"] == "actual_downstream_inputs"
    assert [len(row["history"]) for row in diagnostic["witnesses"][0]["rows"]] == [0, 1, 2]
    assert diagnostic["weighting"] == "uniform_rows"
    # This synthetic fixture supplies the author's alternative explicitly. The
    # runtime detects its consequences but never writes a scientific feature.
    write_model(tmp_path, "def represent(row):\n    return [len(row['history'])]\n"
                         "def predict_features(features):\n    return features[0] + 1\n")
    repaired = evaluate_python_model(tmp_path, "model.py", rows())
    assert repaired["mse"] == 0
    assert repaired["representation_diagnostic"]["empirical_mse_floor"] == 0


def test_training_and_validation_features_do_not_mix_targets(tmp_path):
    write_model(tmp_path)
    held = [{"features": {"u": 0.0}, "history": [], "target": 9.0}]
    result = evaluate_python_model(tmp_path, "model.py", held,
                                   require_fit=True, training_observations=rows()[:2])
    assert result["fit_available"] is True
    assert result["predictions"] == [1.5]
    assert result["training_representation_diagnostic"]["empirical_mse_floor"] == .25
    assert result["representation_diagnostic"]["empirical_mse_floor"] == 0
    assert result["training_representation_diagnostic"]["rows"] == 2


def test_black_box_predictor_is_not_claimed_to_have_registered_representation(tmp_path):
    write_model(tmp_path, "def predict(row):\n    return len(row['history']) + 1\n")
    result = evaluate_python_model(tmp_path, "model.py", rows())
    assert result["mse"] == 0
    assert result["representation_diagnostic"] == {"status": "not_exposed"}


def parameter_inputs(lower, upper, *, version=1):
    return {"threshold": {"question_id": "boundary", "version": version,
                          "kind": "parameter_region",
                          "region": {"lower": lower, "upper": upper,
                                     "lower_closed": False, "upper_closed": False},
                          "scope": "current paid fixture", "evidence": [0, 1]}}


BOUND_MODEL = """
def bind_research(inputs):
    global threshold
    region = inputs['threshold']['region']
    threshold = (region['lower'] + region['upper']) / 2
def get_parameters():
    return {'threshold': threshold}
def predict(row):
    return threshold
"""


def test_bound_child_parameter_changes_actual_prediction_and_retains_old_snapshot(tmp_path):
    write_model(tmp_path, BOUND_MODEL)
    bindings = tmp_path / "model_research_inputs.json"
    first = parameter_inputs(.1, .3)
    bindings.write_text(json.dumps({"model.py": first}))
    result = evaluate_python_model(tmp_path, "model.py", rows())
    assert result["predictions"] == pytest.approx([.2] * 3)
    assert result["research_bindings"][0]["inputs"] == first
    assert result["research_bindings"][0]["actual_parameters"] == {"threshold": .2}
    saved = retain_evaluated_python_model(tmp_path, "model.py", [], result["mse"])
    bindings.write_text(json.dumps({"model.py": parameter_inputs(.7, .9, version=2)}))
    changed = evaluate_python_model(tmp_path, "model.py", rows())
    assert changed["predictions"] == pytest.approx([.8] * 3)
    old = evaluate_python_model(saved["workspace"], saved["model"], rows())
    assert old["predictions"] == pytest.approx([.2] * 3)
    assert old["research_bindings"][0]["inputs"]["threshold"]["version"] == 1


@pytest.mark.parametrize("code", [
    BOUND_MODEL.replace("(region['lower'] + region['upper']) / 2", "region['lower']"),
    BOUND_MODEL.replace("return {'threshold': threshold}", "return {}"),
    BOUND_MODEL + "\ndef fit(rows):\n    global threshold\n    threshold = 99\n",
])
def test_unconsumed_or_outside_child_region_is_execution_failure_not_model_refutation(tmp_path, code):
    write_model(tmp_path, code)
    (tmp_path / "model_research_inputs.json").write_text(json.dumps({"model.py": parameter_inputs(.1, .3)}))
    result = evaluate_python_model(tmp_path, "model.py", rows())
    assert result["returncode"] != 0
    assert "research_input_not_consumed" in result["stderr"]
    assert "mse" not in result


def test_component_predictor_uses_same_explicit_representation_path(tmp_path):
    write_model(tmp_path)
    (tmp_path / "parent.py").write_text(
        "def bind_models(models):\n    global child\n    child = models['part']\n"
        "def predict(row):\n    return child(row)\n")
    (tmp_path / "model_bindings.json").write_text(json.dumps({"parent.py": {"part": "model.py"}}))
    result = evaluate_python_model(tmp_path, "parent.py", rows())
    assert result["predictions"] == [2., 2., 2.]
    assert result["model_dependencies"][0]["calls"] == 3


def make_refit_agent(tmp_path):
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react",
        config=replace(InquiryLLMConfig(), confirmation_mse_threshold=1e-6,
                       execution_seconds=30, compute_budget=1000))
    records = []
    for index, value in enumerate([0., 0., 1.]):
        record_rows = [{"features": {"u": value}, "history": [], "target": 0.}]
        records.append({"observation_index": index, "row_indices": [index],
                        "action": {"kind": "trajectory", "sequence": [{"u": value}]},
                        "rows": record_rows})
    agent.observations = records
    agent.rows = [row for record in records for row in record["rows"]]
    agent._save("model.py", "def represent(row):\n    return [row['features']['u']]\n"
                "def fit_features(features, targets):\n    assert len(targets) == 1\n"
                "def predict_features(features):\n    return 0.0001\n")
    return agent


def test_refit_groups_duplicate_reset_trajectories_and_uses_task_tolerance(tmp_path):
    agent = make_refit_agent(tmp_path)
    before = dict(agent.costs)
    result = agent._refit({"model": "model.py", "validation_records": [0]})
    assert result["split"]["validation_records"] == [0, 1]
    assert result["split"]["training_records"] == [2]
    assert result["split"]["duplicate_groups"] == [[0, 1]]
    assert result["status"] == "refit_supported_on_validation"
    assert result["validation"]["mse"] == pytest.approx(1e-8)
    assert result["validation_mse_threshold"] == 1e-6
    assert result["validation"]["representation_diagnostic"]["rows"] == 2
    assert result["validation"]["training_representation_diagnostic"]["rows"] == 1
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert agent.costs["compute_cost"] - before["compute_cost"] == 3


def test_default_refit_split_keeps_a_distinct_training_group(tmp_path):
    agent = make_refit_agent(tmp_path)
    _, _, split = agent._split({})
    assert split["training_records"] == [0, 1]
    assert split["validation_records"] == [2]


def test_repeat_group_does_not_depend_on_target_equality(tmp_path):
    agent = make_refit_agent(tmp_path)
    agent.observations[1]["rows"][0]["target"] = 1e-8
    _, _, split = agent._split({"validation_records": [0]})
    assert split["training_records"] == [2]
    assert split["validation_records"] == [0, 1]


def test_cannot_claim_whole_trajectory_validation_with_only_one_reset_record_repeated(tmp_path):
    agent = make_refit_agent(tmp_path)
    with pytest.raises(ValueError, match="identical reset records"):
        agent._split({"records": [0, 1], "validation_records": [1]}, indices=[0, 1])
