#!/usr/bin/env python3
"""Receipt for promoting a noisy source return after independent confirmation."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_confirmation_20261004"
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.knowledge_returns import return_priority, transfer_condition
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec


def main():
    spec = TaskSpec("repeat_confirmation", "predict the acquired response", ("x",), (), ())
    observation_sets = [
        [Observation({"x": value}, target)
         for value, target in ((-1., -1.2), (0., .2), (1., 1.0), (2., 2.4))],
        [Observation({"x": value}, target)
         for value, target in ((-1., -1.22), (0., .18), (1., .98), (2., 2.38))],
    ]
    library = KnowledgeLibrary()
    source_id, _ = library.learn(
        Candidate("source", Program("abs", children=(Program("feature", name="x"),)), "representation"),
        TaskSpec("source", "predict the acquired response", ("x",), (), ()))
    rows = []
    for serial, observations in enumerate(observation_sets, 1):
        condition = transfer_condition(spec, observations)
        library.begin_transfer_episode(spec, observations)
        candidate = Candidate(f"repeat_{serial}", library.concepts[0].program, "reuse",
                              memory_id=source_id)
        library.record_transfer_result(candidate, spec, accepted=True, loss=.1,
            baseline_loss=1., evidence_rows=len(observations), comparison_compute=3,
            comparison_id=serial, observations=observations,
            origin={"template_id": None, "mapping": {}, "mapping_roles": {},
                    "condition": condition, "adaptation_work": []})
        result = library.finish_transfer_episode(spec, final_candidate=candidate, final_loss=.1,
            baseline_loss=1., compute_cost=10, observation_cost=len(observations),
            validation_rows=len(observations), final_status="completed")
        record = result["returns"][0]
        rows.append({"visit": serial, "condition": condition,
                     "confirmation_status": record["confirmation_status"],
                     "confirmation_replicates": record["confirmation_replicates"],
                     "return_priority": return_priority(
                         library.concepts[0].reuse_history, condition)})
    assert rows[0]["confirmation_status"] == "provisional"
    assert rows[0]["confirmation_replicates"] == 1
    assert rows[1]["confirmation_status"] == "confirmed"
    assert rows[1]["confirmation_replicates"] == 2
    assert all(item["confirmation_status"] == "confirmed"
               for item in library.concepts[0].reuse_history)
    assert rows[0]["return_priority"] == (0., 0.)
    assert rows[1]["return_priority"][0] < 0.
    payload = {"status": "complete", "protocol": {
        "same_condition": True, "independent_completed_visits": 2,
        "requires_response_residual": True, "external_model_calls": 0},
        "results": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows),
                      "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
