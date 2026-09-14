"""Restored object knowledge must survive reuse and a real wiring change."""

import json

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def test_restored_anonymous_objects_are_reused_then_refuted_and_rebuilt():
    common = {"environment": "anonymous_objects", "objects": 3, "apparatus_seed": 9013,
              "name_seed": 42114, "variant": "exchange"}
    tasks = [{**common, "readout": readout, "seed": seed, "wiring_seed": wiring}
             for readout, seed, wiring in (("focus", 71, 1780), ("total", 10078, 1780),
                                            ("focus", 50106, 2680), ("total", 60113, 2680))]
    config = ResearchConfig(observation_budget=512, compute_budget=480000, max_candidates=16,
        max_rounds=6, minimum_improvement=1e-7, seed=71, active_probes=True)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=6144)
    first = session.run(max_tasks=1)
    original = first["knowledge"]["object_models"][-1]
    assert original["complete"] and original["sensor_model"]
    assert first["results"][0]["research"]["status"] == "completed"

    restored = UnifiedResearchSession(state=json.loads(json.dumps(first)))
    unchanged = restored.run(max_tasks=1)["results"][-1]["research"]
    reuse = [action for action in unchanged["actions"] if action["action"] == "reuse_discovered_objects"]
    assert reuse and reuse[0]["object_model_id"] == original["model_id"]
    assert unchanged["status"] == "completed"
    assert max(reuse[0]["field_losses"].values()) < 1e-7
    assert reuse[0]["rows"] > 0

    # The public names, order and bounds are identical. New observations, not
    # a task-label mismatch, must reopen the restored instrument dependencies.
    with AnonymousObjectLab(**{key: value for key, value in tasks[0].items() if key != "environment"}) as old_lab:
        with AnonymousObjectLab(**{key: value for key, value in tasks[2].items() if key != "environment"}) as rewired_lab:
            assert all(old_lab.spec_for(view) == rewired_lab.spec_for(view) for view in old_lab.measurement_views)
    resumed = UnifiedResearchSession(state=json.loads(json.dumps(restored.to_dict())))
    changed = resumed.run(max_tasks=1)
    report = changed["results"][-1]["research"]
    invalidated = next(model for model in changed["knowledge"]["object_models"]
                       if model["model_id"] == original["model_id"])
    replacement = changed["knowledge"]["object_models"][-1]
    assert invalidated["status"] == "counterexample" and invalidated["counterexamples"]
    assert replacement["complete"] and replacement["model_id"] != original["model_id"]
    assert replacement["schema"] != original["schema"]
    assert report["status"] == "completed"
    assert any(action["action"] == "reopen_object_dependencies" for action in report["actions"])

    # A further process boundary exercises the revised knowledge too. Terminal
    # predictions use new real trajectories only after all learning is done.
    successor = UnifiedResearchSession(state=json.loads(json.dumps(changed)))
    final = successor.run()
    later = final["results"][-1]["research"]
    assert later["status"] == "completed"
    assert any(action["action"] == "reuse_discovered_objects" and
               action["object_model_id"] == replacement["model_id"] for action in later["actions"])
    evaluated = successor.evaluate(length=32, trajectories=2, seed_offset=81173)
    assert all(row["loss"] < 1e-7 for row in evaluated["evaluation"])
    assert len(evaluated["knowledge"]["object_models"]) == 2
