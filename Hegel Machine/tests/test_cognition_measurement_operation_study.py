"""A saved early-regime model must face the same mature operation in evaluation."""

import importlib.util
from pathlib import Path

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab


PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_measurement_operation.py"
SPEC = importlib.util.spec_from_file_location("measurement_operation_study", PATH)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def model():
    def term(value, variable):
        return Program("mul", children=(Program("const", value=value), variable))
    drive = Program("add", children=(term(.45, Program("feature", name="u")),
                                     term(.8, Program("feature", name="sensor_a"))))
    mature = Program("add", children=(term(.65, Program("lag", name="__target__", lag=1)), drive))
    return Program("if", children=(Program("available", lag=1), mature, drive))


def test_operational_score_reveals_a_drive_dormant_in_the_early_probe():
    program = model()
    options = {"backend": "process", "evaluation": {"seed_offset": 1000003,
                "length": 24, "trajectories": 1}}
    result = {"task_position": 3, "task_index": 3, "seed": 71,
              "selected_channels": ["sensor_a"],
              "episode": {"program": program.to_dict(), "fallback": 0.}}
    with make_lab(3, 71) as lab:
        early = lab.evaluation_rows(("sensor_a",), seed=1000074, length=24, trajectories=1)
    assert loss(program, early, 0., .5) < 1e-12
    mature = study.score_model(result, options)
    assert mature["mse"] > .03
    assert mature["costs"] == {"warmup_observations": 64, "scoring_observations": 24,
                                "scoring_measurements": 24, "scoring_compute": 24}


def test_operational_comparison_ignores_different_research_lengths_and_costs():
    result = {"task_position": 3, "task_index": 3, "seed": 71,
              "selected_channels": ["sensor_a"],
              "episode": {"program": model().to_dict(), "fallback": 0.}}
    spec = {"backend": "local", "evaluation": {"seed_offset": 1000003,
            "length": 24, "trajectories": 1}}
    short = study.score_model({**result, "costs": {"observation_cost": 24}}, spec)
    long = study.score_model({**result, "costs": {"observation_cost": 96}}, spec)
    assert short == long
