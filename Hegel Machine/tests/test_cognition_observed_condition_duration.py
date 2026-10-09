"""Actual paid fixed-background trajectories nominate temporal conditions."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.models import Probe, SearchBudget
from hegel_machine.cognition.observed_condition_duration import propose_observed_condition_duration


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/observed_condition_decoding_20260930/development/protocol.json"


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("observed_duration_physics",
        ROOT / "scripts/run_observed_condition_decoding_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_case(study, root, stage, *, varying=False):
    task = json.loads(PROTOCOL.read_text())["roots"][root]["tasks"][stage]
    lab = study.laboratory(task)
    field = lab.channels[0].name
    background = {name: high * .5 for name, (_, high) in lab.spec.feature_bounds.items()}
    sequence = [dict(background) for _ in range(24)]
    if varying:
        control = lab.spec.feature_names[0]
        sequence[12][control] = lab.spec.feature_bounds[control][0]
    rows = lab.acquire(Probe("fixed_duration_support", tuple(sequence)), (field,)).observations
    return lab, lab.spec_for((field,)), rows


def save_evidence(path, lab, budget, claims):
    with gzip.open(path, "wt") as stream:
        json.dump({"actual_costs": lab.costs, "compute_cost": budget.spent,
            "nominees": [{"original_training_loss": entry["training_loss"],
                "program": entry["candidate"].program.to_dict(),
                "evidence": entry["observed_condition_evidence"]} for entry in claims]}, stream)


@pytest.mark.parametrize("root,stage,expected_width", [
    ("root01", 4, 7), ("root02", 4, 7), ("root02", 2, 5)])
def test_joint_condition_duration_reopens_previously_wrong_windows(
        study, root, stage, expected_width, tmp_path):
    lab, spec, rows = measured_case(study, root, stage)
    budget = SearchBudget(40000)
    # Neither the apparatus stage nor its evaluated true width is a fitter
    # argument. The only scale evidence is this actually purchased trajectory.
    claims = propose_observed_condition_duration(spec, rows, budget)
    save_evidence(tmp_path / f"{root}_stage{stage}_duration.json.gz", lab, budget, claims)
    assert claims
    selected = claims[0]
    evidence = selected["observed_condition_evidence"]
    assert evidence["status"] == "conditional_duration"
    assert evidence["width_candidates"] == [expected_width]
    assert evidence["placement_candidates"] == ["outside_window"]
    assert selected["training_loss"] <= 1e-6
    assert selected["training_loss"] == pytest.approx(loss(selected["candidate"].program,
        rows, initial_prediction(rows, .5), .5))
    assert lab.costs["observation_cost"] == lab.costs["measurement_cost"] == 24
    assert lab.costs["queries"] == 1
    assert 0 < budget.spent <= 40000


def test_changing_raw_background_requires_the_full_response_method(study):
    _, spec, rows = measured_case(study, "root01", 2, varying=True)
    budget = SearchBudget(40000)
    assert propose_observed_condition_duration(spec, rows, budget) == []
    assert budget.spent == 0


def test_smooth_window_does_not_acquire_conditional_gain(study, tmp_path):
    lab, spec, rows = measured_case(study, "root01", 6)
    budget = SearchBudget(40000)
    claims = propose_observed_condition_duration(spec, rows, budget)
    save_evidence(tmp_path / "smooth_duration.json.gz", lab, budget, claims)
    assert claims[0]["observed_condition_evidence"]["status"] == "smooth_duration"
    assert claims[0]["observed_condition_evidence"]["width_candidates"] == [5]
    assert claims[0]["training_loss"] <= 1e-6
    assert max(entry["observed_condition_evidence"]["gain"] for entry in claims) <= 1e-10
