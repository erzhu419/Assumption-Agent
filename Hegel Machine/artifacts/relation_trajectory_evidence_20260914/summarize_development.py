"""Summarize recorded outcomes on the seven explicitly reused development cases."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
protocol = json.loads((OUTPUT / "protocol.json").read_text())


def describe(path):
    result = json.loads(path.read_text())
    acquisitions = result["acquired"]
    repeats = []
    for index, item in enumerate(acquisitions):
        previous = next((other for other in range(index)
            if acquisitions[other]["batch"]["channels"] == item["batch"]["channels"]
            and acquisitions[other]["probe"]["sequence"] == item["probe"]["sequence"]), None)
        if previous is not None:
            repeats.append({"first": previous, "repeated": index,
                            "rows": item["batch"]["observation_cost"]})
    return {"file": str(path.relative_to(ROOT)), "costs": result["costs"],
        "learning_wall_seconds": result["learning_wall_seconds"],
        "solved": result["terminal"]["solved"],
        "random_mse": result["terminal"]["random_mse"], "boundary_mse": result["terminal"]["boundary_mse"],
        "same_channel_same_probe_repeats": repeats,
        "repeat_count": len(repeats), "repeated_rows": sum(item["rows"] for item in repeats)}


cases = []
for seed in protocol["development_apparatus"]:
    for case in protocol["development_case_subset"][str(seed)]:
        old = describe(ROOT / protocol["reused_source_directories"][str(seed)] / str(seed) / (case + ".json"))
        new = describe(OUTPUT / "development/revised" / str(seed) / (case + ".json"))
        cases.append({"seed": seed, "case": case, "frozen_084": old, "revised": new,
            "cost_change": {name: new["costs"][name] - old["costs"][name] for name in old["costs"]}})

arms = {}
for arm in ("frozen_084", "revised"):
    rows = [row[arm] for row in cases]
    arms[arm] = {"solved": sum(row["solved"] for row in rows), "cases": len(rows),
        "costs": {key: sum(row["costs"][key] for row in rows)
                  for key in ("observation_cost", "measurement_cost", "compute_cost", "queries")},
        "learning_wall_seconds": sum(row["learning_wall_seconds"] for row in rows),
        "same_channel_same_probe_repeats": sum(row["repeat_count"] for row in rows),
        "repeated_rows": sum(row["repeated_rows"] for row in rows)}
summary = {"scope": "Known-case development; existing formal and development cases, not independent confirmation",
    "formal_evidence": False, "runtime_source": protocol["new_development_runtime"],
    "source_costs": "Existing source preparations reused unchanged; costs below are branch-only",
    "arms": arms, "cases": cases}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(arms, indent=2))
