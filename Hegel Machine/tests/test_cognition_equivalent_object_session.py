"""Ambiguous object knowledge can execute, survive restore, and be revised."""

import json

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def test_ambiguous_exchange_survives_restore_then_direct_effect_reopens_roles():
    common = {"environment": "anonymous_objects", "objects": 2,
              "apparatus_seed": 17011, "name_seed": 63940, "wiring_seed": 3792,
              "readout": "focus"}
    tasks = [{**common, "seed": seed, "variant": variant}
             for seed, variant in ((83, "exchange"), (10090, "exchange"), (20097, "anchored"))]
    config = ResearchConfig(seed=83, observation_budget=512, compute_budget=480000,
                            max_rounds=6, minimum_improvement=1e-7, active_probes=True)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=6144)
    first = session.run(max_tasks=1)
    answer = first["results"][0]["research"]
    assert answer["status"] == "completed", answer["confirmation_loss"]
    assert answer["object_equivalence"]["exhaustive"]
    assert len(answer["object_equivalence"]["schemas"]) == 2
    assert answer["learned_object_schema"] is None
    old = next(item for item in first["knowledge"]["object_models"]
               if item["model_id"] == answer["object_model_id"])
    assert not old["complete"] and all("x" not in group for group in old["groups"])

    restored = UnifiedResearchSession(state=json.loads(json.dumps(first)))
    report = restored.run()
    second = report["results"][1]["research"]
    assert any(event["action"] == "reuse_object_alternatives" for event in second["actions"])
    assert second["used_sources"]
    assert second["learned_object_schema"] is None
    changed = report["results"][2]["research"]
    assert any(event["action"] == "reopen_object_dependencies"
               and event["reason"] == "observed_role_dependency_changed" for event in changed["actions"])
    assert changed["object_research"]["complete"] and changed["learned_object_schema"]
    retained = next(item for item in report["knowledge"]["object_models"]
                    if item["model_id"] == old["model_id"])
    assert not retained["complete"] and len(retained["schema_alternatives"]["schemas"]) == 2
    terminal = restored.evaluate(length=32, trajectories=2, seed_offset=270317)
    assert all(row["loss"] < 1e-7 for row in terminal["evaluation"]), [
        row["loss"] for row in terminal["evaluation"]]
    assert all(row["measurement_cost"] == 0 for row in terminal["evaluation"])
