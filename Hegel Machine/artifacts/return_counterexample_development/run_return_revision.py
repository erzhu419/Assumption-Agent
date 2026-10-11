"""Replay already observed return failures as development cases for the patch."""

import json
from pathlib import Path
import sys
from time import perf_counter

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


seed = int(sys.argv[1])
root = Path(__file__).resolve().parent
source = root / "v080_prefix" / "runs" / "hegel_revised" / str(seed) / "hegel_checkpoint.json"
state = json.loads(source.read_text())["session"]
original = json.loads((root.parent / "object_relation_revision_20260913" / "runs" /
    "hegel_revised" / str(seed) / "hegel_checkpoint.json").read_text())["session"]
assert [row["research"] for row in state["results"]] == [row["research"] for row in original["results"][:5]]
session = UnifiedResearchSession(state=state)
started = perf_counter()
changed = session.run(max_tasks=1)
learning_wall_seconds = perf_counter() - started
report = changed["results"][5]["research"]
checks = []
with laboratory_for(changed["tasks"][5]) as lab:
    program = Program.from_dict(report["program"])
    for length, test_seed in ((1, 1921), (3, 1922), (64, 92341)):
        rows = lab.acquire(lab.random_probe(test_seed, length=length), report["selected_channels"]).observations
        checks.append({"length": length, "seed": test_seed,
            "mse": sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)})
directed_ids = {action["probe_id"] for action in report["actions"]
    if action["action"] == "acquire" and action["purpose"] == "distinguish_transfer_conditions"}
directed = [Observation(row["features"], row["target"], tuple(row["history"]))
    for batch in report["batches"] if batch["probe"]["probe_id"] in directed_ids
    for row in batch["observations"]]
paid_errors = [(program.predict(row) - row.target) ** 2 for row in directed] if not report["selected_channels"] else []
result = {"scope": "Development reproduction of an observed v0.80 return failure; not new apparatus evidence",
    "seed": seed, "source_prefix": str(source.relative_to(root.parent.parent)),
    "source_prefix_research_matches_v080": True,
    "v080_costs": original["results"][5]["research"]["costs"],
    "v080_learning_wall_seconds": original["results"][5].get("learning_wall_seconds"),
    "learning_wall_seconds": learning_wall_seconds,
    "research": report, "development_checks": checks, "knowledge": changed["knowledge"],
    "paid_directed_rows": len(directed),
    "paid_directed_mse": sum(paid_errors) / len(paid_errors) if paid_errors else None,
    "paid_directed_max_absolute_error": max(paid_errors) ** .5 if paid_errors else None}
(root / f"{seed}_revision.json").write_text(json.dumps(result))
print(json.dumps({"seed": seed, "old": result["v080_costs"], "new": report["costs"],
    "learning_wall_seconds": learning_wall_seconds,
    "confirmation_loss": report["confirmation_loss"], "development_checks": checks,
    "paid_directed_max_absolute_error": result["paid_directed_max_absolute_error"],
    "sources": report["used_sources"]}))
