"""The full-library study keeps its prior fixed and scores narrow activations."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("transfer_condition_study", ROOT / "scripts/run_transfer_condition_research_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def setup_case():
    study.load_runtime(ROOT / "src")
    return json.loads(study.PROTOCOL.read_text())


def test_gain_and_activation_changes_do_not_change_the_public_finite_candidate_prior():
    protocol = setup_case()
    tasks = [study.task_settings(protocol, 151, "narrow_windows", stage) for stage in range(5)]
    assert all(task["public_prior"] == tasks[0]["public_prior"] for task in tasks)
    labs = [study.TransferLaboratory(task) for task in tasks]
    assert labs[1].spec == labs[2].spec == labs[3].spec
    assert study.prior_programs(labs[1].spec) == study.prior_programs(labs[3].spec)
    center = tasks[0]["public_prior"]["window_locations"][0]
    probe = study.Probe("same_public_activation", (dict(zip(tasks[1]["names"], center)),))
    targets = [lab.acquire(probe).observations[0].target for lab in labs[1:4]]
    assert targets == [2., 3., 0.]


def test_constant_zero_cannot_pass_any_narrow_stage_and_return_is_scored_again():
    protocol = setup_case()
    losses = []
    for stage in range(5):
        settings = study.task_settings(protocol, 151, "narrow_windows", stage)
        evaluation = study.evaluate(settings, {"program": study.constant(0.).to_dict()}, protocol, 1301 + stage)
        assert evaluation["rows"] == 256
        assert evaluation["categories"][0]["loss"] == 0.
        assert sum(item["positive_targets"] for item in evaluation["categories"]) == 64
        losses.append(evaluation["loss"])
    assert losses == [1., 1., 2.25, 2.25, 1.]
