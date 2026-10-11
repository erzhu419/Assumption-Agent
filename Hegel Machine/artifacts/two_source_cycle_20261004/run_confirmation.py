"""Run the frozen source snapshot on the registered independent cases."""
import json
from pathlib import Path
import sys
from time import monotonic

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "runtime/scripts"))
from run_two_source_research_cycle import protocol, run_cycle
import hegel_machine.cognition.unified_research as research

assert Path(research.__file__).is_relative_to(root / "runtime")
registered = json.loads((root / "protocol.json").read_text())
started = monotonic()
rows = []
for case in registered["cases"]:
    destination = root / f"case_{case['seed']}_{case['profile']}.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = run_cycle(protocol(**case))
    destination.write_text(json.dumps(result, indent=2) + "\n")
    row = {**case, "cycle_passed": result["cycle_passed"],
        "failed_checks": [key for key, value in result["cycle_checks"].items() if not value],
        "source_deletion_effects": result["source_deletion_effects"],
        "cost_interaction": result["cost_interaction"],
        "arms": {arm: {key: value[key] for key in ("solved", "mean_mse", "costs")}
                 for arm, value in result["results"].items()}}
    rows.append(row)
    print(json.dumps({**case, "cycle_passed": row["cycle_passed"],
        "failed_checks": row["failed_checks"], "all_marginals_save": all(
            v["sequence_pareto_saving"] for v in row["source_deletion_effects"].values())}), flush=True)
summary = {"protocol": registered, "case_count": len(rows),
    "cycle_pass_count": sum(row["cycle_passed"] for row in rows),
    "all_marginals_save_count": sum(all(v["sequence_pareto_saving"]
        for v in row["source_deletion_effects"].values()) for row in rows),
    "rows": rows, "elapsed_seconds": monotonic() - started, "external_model_calls": 0}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key not in ("protocol", "rows")}), flush=True)
