"""A new readout can separate executable laws without inventing object identity."""

import json

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def test_position_readout_competes_then_survives_restore_and_reuses_knowledge():
    common = {"environment": "anonymous_objects", "objects": 2,
              "apparatus_seed": 27011, "name_seed": 63954, "wiring_seed": 3806,
              "variant": "exchange"}
    tasks = [{**common, "seed": 97 + 10007 * index, "readout": readout}
             for index, readout in enumerate(("focus", "position_focus", "position_focus"))]
    config = ResearchConfig(seed=97, observation_budget=512, compute_budget=480000,
                            max_rounds=6, minimum_improvement=1e-7, active_probes=True)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=6144)
    state = session.run(max_tasks=2)
    answer = state["results"][1]["research"]
    assert answer["status"] == "completed", answer["confirmation_loss"]
    assert answer["object_competition"]["hypotheses"]
    assert answer["learned_object_schema"] is None
    assert len(answer["object_competition"]["schemas"]) == 2
    stored = next(model for model in state["knowledge"]["object_models"]
                  if model["model_id"] == answer["object_model_id"])
    assert stored["program_competitions"] and not stored["complete"]
    assert any(prior.get("object_competition") for prior in state["knowledge"]["meta_priors"])
    restored = UnifiedResearchSession(state=json.loads(json.dumps(state)))
    completed = restored.run()
    successor = completed["results"][2]["research"]
    assert successor["status"] == "completed"
    assert successor["used_sources"] and successor["learned_object_schema"] is None
    terminal = restored.evaluate(length=32, trajectories=2, seed_offset=270317)
    assert all(row["loss"] < 1e-7 for row in terminal["evaluation"]), terminal["evaluation"]
    assert all(row["measurement_cost"] == 0 for row in terminal["evaluation"])
