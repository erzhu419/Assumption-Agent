"""Paid initial compatibility and independent disagreement-region evaluation."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compound_study", ROOT / "scripts/run_compound_experiment_methods_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def setup_case():
    study.load_runtime(ROOT / "src")
    protocol = json.loads(study.PROTOCOL.read_text())
    return protocol, study.task_settings(protocol, 137, protocol["cases"][1], 0)


def test_both_hidden_choices_have_identical_paid_initial_evidence_and_public_programs():
    protocol, settings = setup_case()
    states = []
    for chosen in (1, 2):
        current = {**settings, "chosen_index": chosen}
        lab = study.GateApparatus(current)
        pairs, binding, inverse = study.initial_evidence(lab, current)
        programs = study.candidate_programs(current)
        fitted = study.fit_candidates(programs, pairs, study.SearchBudget(1000))
        assert [item["hypothesis_id"] for item in fitted] == ["gate_hypothesis_1", "gate_hypothesis_2"]
        states.append((pairs, programs, inverse))
    assert states[0] == states[1]
    point = dict(zip(settings["sensors"], settings["centers"]))
    point[settings["sensors"][0]] = sum(settings["lower_candidates"][1:3]) / 2
    probe = study.Probe("new_joint_disagreement", (study.commands(point, states[0][2]),))
    targets = [study.GateApparatus({**settings, "chosen_index": chosen}).acquire(probe)[0][1].target
               for chosen in (1, 2)]
    assert targets == [1., 0.]


def test_wrong_candidate_cannot_pass_by_predicting_zero_on_uniform_terminal_rows():
    protocol, settings = setup_case()
    lab = study.GateApparatus(settings)
    pairs, binding, inverse = study.initial_evidence(lab, settings)
    programs = study.candidate_programs(settings)
    wrong = 2 if settings["chosen_index"] == 1 else 1
    evaluation = study.evaluate(settings, {"selected_program": programs[wrong].to_dict(),
                                "inverse_calibration": inverse}, protocol, 99173)
    assert evaluation["rows"] == 4 * 64
    assert evaluation["loss"] == .25
    assert evaluation["categories"][0]["loss"] == 0.
    assert evaluation["categories"][1]["positive_targets"] == 64
    assert evaluation["categories"][2]["positive_targets"] == 0
    assert evaluation["categories"][3]["loss"] == 1.
