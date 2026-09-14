"""An unresolved program boundary leads to measured revision and later reuse."""

import json

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def test_program_disagreement_is_measured_before_restored_successor_reuses_the_law():
    # This apparatus previously completed its boundary research entirely via
    # selected-model probes: the generic competing-program pool bought none.
    seed = 6060319271
    common = {"environment": "anonymous_objects", "objects": 2, "apparatus_seed": 31013,
              "name_seed": seed + 63857, "wiring_seed": seed + 3709, "variant": "exchange"}
    tasks = [{**common, "seed": seed + 10007 * index, "readout": readout}
             for index, readout in enumerate(("focus", "position_focus", "position_focus"))]
    config = ResearchConfig(seed=seed, observation_budget=512, compute_budget=480000,
        max_rounds=6, minimum_improvement=1e-7, active_probes=True)
    session = UnifiedResearchSession(config, tasks=tasks, measurement_budget=6144)
    state = session.run(max_tasks=2)
    researched = state["results"][1]["research"]
    assert researched["status"] == "completed", researched["confirmation_loss"]
    directed = [item for item in researched["actions"]
                if item["action"] == "test_competing_object_predictions"
                and item["plan"].get("proposal_source") == "program_disagreement"]
    assert directed
    acquisitions = {item["probe_id"]: item for item in researched["actions"]
                    if item["action"] == "acquire"}
    for tested in directed:
        assert acquisitions[tested["probe_id"]]["measurement_cost"] > 0
        assert tested["plan"]["predicted_separation"] > config.minimum_improvement
        assert all(hypothesis["predictions"] for hypothesis in tested["plan"]["hypotheses"])
    assert any(item["evidence"]["status"] == "distinguished" for item in directed)
    assert researched["object_competition"]["counterexamples"]
    assert researched["learned_object_schema"] is None

    restored = UnifiedResearchSession(state=json.loads(json.dumps(state)))
    completed = restored.run()
    successor = completed["results"][2]["research"]
    assert successor["status"] == "completed" and successor["used_sources"]
    terminal = restored.evaluate(length=32, trajectories=2, seed_offset=270317)
    assert all(row["loss"] < 1e-7 for row in terminal["evaluation"]), terminal["evaluation"]
    assert all(row["measurement_cost"] == 0 for row in terminal["evaluation"])
