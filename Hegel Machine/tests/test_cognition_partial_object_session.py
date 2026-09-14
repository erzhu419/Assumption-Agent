"""Partial object knowledge survives a process boundary and a changed task."""

import json

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def test_restored_partial_graph_predicts_totals_then_researches_missing_focus():
    common = {"environment": "anonymous_objects", "objects": 3, "apparatus_seed": 9013,
              "name_seed": 42114, "wiring_seed": 1780, "variant": "exchange"}
    tasks = [{**common, "readout": readout, "seed": seed}
             for readout, seed in (("total", 71), ("total", 10078), ("focus", 20085))]
    config = ResearchConfig(observation_budget=512, compute_budget=480000, max_rounds=6,
                            max_candidates=16, minimum_improvement=1e-7, seed=71)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=6144)
    first = session.run(max_tasks=1)
    initial = first["results"][0]["research"]
    old = first["knowledge"]["object_models"][-1]
    assert initial["status"] == "completed", (initial["selected_kind"], initial["confirmation_loss"])
    assert initial["selected_channels"] == []
    assert initial["learned_object_schema"] is None
    assert not old["complete"] and old["schema"] is None and old["sensor_model"]
    assert all({"q", "u", "x"} <= set(group) and "f" not in group for group in old["groups"])

    restored = UnifiedResearchSession(state=json.loads(json.dumps(first)))
    second = restored.run(max_tasks=1)["results"][-1]["research"]
    reuse = [action for action in second["actions"] if action["action"] == "reuse_discovered_objects"]
    assert reuse and reuse[0]["object_model_id"] == old["model_id"]
    assert reuse[0]["complete"] is False and max(reuse[0]["field_losses"].values()) < 1e-7
    assert second["status"] == "completed" and second["selected_channels"] == []
    assert not second["object_research"]["complete"] and second["learned_object_schema"] is None
    assert all("f" not in group for group in second["object_research"]["groups"])

    final = restored.run()
    focal = final["results"][-1]["research"]
    assert any(action["action"] == "refine_partial_object_dependencies"
               and action["object_model_id"] == old["model_id"] for action in focal["actions"])
    assert focal["status"] == "completed", (focal["selected_kind"], focal["confirmation_loss"])
    assert focal["object_research"]["complete"] and focal["learned_object_schema"]
    previous = {group["q"]: group for group in old["groups"]}
    assert all(all(group[role] == previous[group["q"]][role] for role in ("u", "x"))
               and "f" in group for group in focal["object_research"]["groups"])
    retained = next(model for model in final["knowledge"]["object_models"] if model["model_id"] == old["model_id"])
    assert not retained["complete"] and all("f" not in group for group in retained["groups"])

    scored = restored.evaluate(length=32, trajectories=2, seed_offset=91173)
    assert all(row["loss"] < 1e-7 for row in scored["evaluation"]), scored["evaluation"]
    # Learning acquired instruments; subsequent predictions on both total tasks
    # execute from controls alone, including after knowledge was serialized.
    assert all(row["measurement_cost"] == 0 for row in scored["evaluation"][:2])
