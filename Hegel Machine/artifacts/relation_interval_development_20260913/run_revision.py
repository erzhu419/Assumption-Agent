"""Replay two already observed relation failures; these are development cases."""

import json
from pathlib import Path
import sys
from time import perf_counter

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


seed = int(sys.argv[1])
root = Path(__file__).resolve().parent
state = json.loads((root / f"{seed}_prefix_v0801.json").read_text())
baseline = json.loads((root / f"{seed}_baseline_v0801.json").read_text())
start = perf_counter()
current = UnifiedResearchSession(state=state).run(max_tasks=1)
wall = perf_counter() - start
report = current["results"][4]["research"]
with laboratory_for(current["tasks"][4]) as lab:
    checks = {}
    for name, result in (("v0801", baseline["research"]), ("v0802", report)):
        rows = lab.evaluation_rows(result["selected_channels"],
            seed=current["tasks"][4]["seed"] + 1000003, length=64, trajectories=4)
        program = Program.from_dict(result["program"])
        checks[name] = sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)
summary = {"seed": seed, "scope": "Development on two previously observed failures",
    "baseline_costs": baseline["research"]["costs"], "costs": report["costs"],
    "baseline_wall_seconds": baseline["learning_wall_seconds"], "wall_seconds": wall,
    "development_terminal_mse": checks, "confirmation_loss": report["confirmation_loss"]}
(root / f"{seed}_revision_v0802.json").write_text(json.dumps({**summary,
    "research": report, "knowledge": current["knowledge"]}))
print(json.dumps(summary), flush=True)
