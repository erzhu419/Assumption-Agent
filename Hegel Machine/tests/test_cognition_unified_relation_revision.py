"""A conflicting paid trajectory must revise the relation before generic search."""

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def test_relation_revision_reaches_fresh_confirmation_after_short_transfer_fails():
    seed = 3050319137
    tasks = [{"task_index": 0, "seed": seed},
             {"environment": "objects", "readout": "total", "apparatus_seed": 6131,
              "seed": seed + 10007},
             {"environment": "objects", "readout": "event", "apparatus_seed": 6131,
              "seed": seed + 20014}]
    session = UnifiedResearchSession(ResearchConfig(observation_budget=384,
        compute_budget=360000, max_rounds=6, minimum_improvement=1e-7,
        active_probes=True), tasks=tasks, measurement_budget=4608)
    report = session.run()
    event = report["results"][-1]["research"]
    revisions = [action for action in event["actions"]
                 if action["action"] == "revise_structure_from_combined_evidence"]
    assert any(action["answered"] and action["rows"] == 72 for action in revisions)
    assert event["status"] == "completed"
    assert event["selected_kind"] == "research_program"
    assert any(step["operation"] == "resolve_relation_intervals"
               for step in event["research_program"]["selection_steps"])
    assert event["selection_basis"] == "all_acquired_evidence"
    assert event["validation_loss"] is None
    assert event["confirmation_loss"] < 1e-7
    assert event["costs"]["compute_cost"] < 360000
    evaluated = session.evaluate(length=64, trajectories=4)
    assert all(row["loss"] < 1e-7 for row in evaluated["evaluation"])
