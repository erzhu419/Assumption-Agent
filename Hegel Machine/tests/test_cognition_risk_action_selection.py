"""Risk actions learn from paid responses under the unchanged external objective."""

import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


CONFIG = ResearchConfig(observation_budget=384, compute_budget=240000,
    minimum_improvement=1e-7, active_probes=False, max_rounds=4, seed=7907)


@pytest.fixture(scope="module")
def risk_chain():
    tasks = [{"task_index": task, "seed": seed, "profile": profile}
             for task, seed, profile in ((12, 7907, "development"),
                 (13, 7908, "development"), (13, 7909, "confirmation"),
                 (12, 7910, "development"))]
    session = UnifiedResearchSession(CONFIG, tasks=tasks, measurement_budget=1536)
    snapshots = []
    for _ in tasks:
        state = json.loads(json.dumps(session.run(max_tasks=1)))
        snapshots.append(state)
        session = UnifiedResearchSession(state=state)
        assert session.library.action_experience == state["knowledge"]["action_experience"]
    return snapshots


def test_risk_actions_compare_both_programs_on_the_same_new_fixed_quantile_response(risk_chain):
    reports = [item["research"] for item in risk_chain[-1]["results"]]
    comparisons = []
    choices = []
    for report in reports:
        assert report["risk_quantile"] == .9 and report["loss_kind"] == "pinball"
        assert report["costs"]["observation_cost"] <= CONFIG.observation_budget
        assert report["costs"]["compute_cost"] <= CONFIG.compute_budget
        assert report["costs"]["measurement_cost"] <= 1536
        for event in report["actions"]:
            if event["action"] == "choose_risk_research":
                choices.append(event)
            if event["action"] != "risk_research_feedback" or event["before_loss"] is None:
                continue
            batch = next(batch for batch in report["batches"]
                         if batch["probe"]["probe_id"] == event["probe_id"])
            rows = [Observation(item["features"], item["target"], tuple(item["history"]))
                    for item in batch["observations"]]
            # Numeric instrument readings retain the public controls. Both
            # frozen programs can execute on this same richer observation.
            before = loss(Program.from_dict(event["baseline_program"]), rows,
                          event["baseline_fallback"], .9)
            after = loss(Program.from_dict(event["program"]), rows, event["fallback"], .9)
            assert event["risk_quantile"] == .9
            assert event["before_loss"] == pytest.approx(before)
            assert event["after_loss"] == pytest.approx(after)
            assert event["observation_rows"] == len(rows)
            comparisons.append(event)
    assert choices and any(len(event["options"]) > 1 for event in choices)
    assert comparisons and any(event["trial"] is not None for event in comparisons)
    assert any(event["after_loss"] > CONFIG.minimum_improvement for event in comparisons)


def test_missing_input_cannot_be_excused_by_a_revised_noise_model(risk_chain):
    changed = risk_chain[1]["results"][-1]["research"]
    trials = [event for event in changed["actions"]
              if event["action"] == "test_learned_acquisition_method"]
    assert trials and not trials[0]["noise_assessment"]["adequate"]
    assert len(trials[0]["validation_channels"]) == 2
    assert any(event["action"] == "choose_risk_research" for event in changed["actions"])
    # Adoption must survive the ordinary rich-coordinate residual challenge.
    if changed["status"] == "completed":
        assert changed["selected_channels"]
        assert changed["noise_assessment"]["adequate"]
    else:
        assert changed["concepts_added"] == []
    assert risk_chain[-1]["results"][:-1] == risk_chain[-2]["results"]

