#!/usr/bin/env python3
"""Record executed retention of paid-counterexample repairs and final model survival."""
import gzip
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
protocol = json.loads((output / "protocol.json").read_text())
result = {"basis": "Recorded retain_paid_counterexample_repair learning actions. "
    "The revised cached-loss bound is not a new evaluation or exact rescoring of prior data.", "arms": {}}
for arm in protocol["arms"]:
    records = []
    for root in protocol["roots"]:
        with gzip.open(output / "runs" / arm / root / "checkpoint.json.gz", "rt") as stream:
            state = json.load(stream)
        for item in state["results"]:
            report = item["research"]
            for index, action in enumerate(report["actions"]):
                if action["action"] != "retain_paid_counterexample_repair":
                    continue
                records.append({"root": root, "position": item["task_position"],
                    "action_index": index, **action,
                    "retained_to_final_program": action["program"] == report["program"]})
    result["arms"][arm] = {"repair_retention_actions": len(records),
        "tasks_with_repair_retention": len({(r["root"], r["position"]) for r in records}),
        "repairs_retained_to_final_program": sum(r["retained_to_final_program"] for r in records),
        "repair_retention_records": records}
(output / "paid_retention_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print({arm: {key: value for key, value in data.items() if not isinstance(value, list)}
       for arm, data in result["arms"].items()})
