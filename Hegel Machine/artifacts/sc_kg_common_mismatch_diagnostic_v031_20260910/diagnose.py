"""Read pre-observation forecasts and paid targets; no truth calls or threshold search."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "artifacts/sc_kg_prior_adoption_v031_20260910"
CHANGED = ROOT / "artifacts/sc_kg_prior_mechanism_change_v031_20260910/summary.json"


def detect(policy, *, live_noise=False):
    rows = policy["observations"]
    initial = rows[:4]
    reference_error = min(mean(row[key] for row in initial)
                          for key in ("default_error", "learned_error"))
    reference_noise = mean(row["noise_variance"] for row in initial)
    floor = policy["minimum_gain"]
    reference_scale = max(floor, reference_error, reference_noise)
    consecutive, longest, detection = [], 0, None
    exceedances = []
    for index, row in enumerate(rows[4:], 4):
        threshold = 9 * (max(floor, reference_error, row["noise_variance"])
                         if live_noise else reference_scale)
        failed = min(row["default_error"], row["learned_error"]) > threshold
        if failed:
            record = {"index": index, "stage": row.get("context", {}).get("stage"),
                      "default_error": row["default_error"], "learned_error": row["learned_error"],
                      "noise_variance": row["noise_variance"], "threshold": threshold}
            consecutive.append(record)
            exceedances.append(record)
        else:
            consecutive = []
        longest = max(longest, len(consecutive))
        if len(consecutive) == 4 and detection is None:
            detection = {"index": index, "stage": row.get("context", {}).get("stage"),
                         "evidence": list(consecutive)}
    return {"reference_rows": 4, "reference_error": reference_error,
            "reference_noise": reference_noise, "reference_scale": reference_scale,
            "longest_run": longest, "detection": detection,
            "exceedance_count": len(exceedances),
            "maximum_recorded_noise": max(row["noise_variance"] for row in rows),
            "exceedances": exceedances if detection else []}


changed = json.loads(CHANGED.read_text())
result = {"rule": {"initial_reference_rows": 4, "squared_error_multiplier": 9,
                    "consecutive_counterexamples": 4,
                    "reference": "Maximum of first-batch mean noise, the better model's first-batch MSE, and the existing minimum_gain",
                    "restriction": "Fixed before replay; no mechanism label, change point, or final truth used by the detector."},
          "new_simulator_calls": 0, "mechanism_change_case": {
              "frozen_reference": detect(changed["policy"]),
              "live_noise_comparator": detect(changed["policy"], live_noise=True)},
          "static_cases": [], "static_groups": {}}
for path in sorted(STATIC.glob("*_auto.json")):
    saved = json.loads(path.read_text())
    row = {"case": path.stem, "reported_noise": saved["task"]["noise"],
           "research_status": saved["research_status"],
           "frozen_reference": detect(saved["adoption_policy"]),
           "live_noise_comparator": detect(saved["adoption_policy"], live_noise=True)}
    result["static_cases"].append(row)
for noise in sorted({row["reported_noise"] for row in result["static_cases"]}):
    cases = [row for row in result["static_cases"] if row["reported_noise"] == noise]
    result["static_groups"][str(noise)] = {"runs": len(cases), "forecast_rows": 32 * len(cases),
        "frozen_reference_triggers": sum(row["frozen_reference"]["detection"] is not None for row in cases),
        "live_noise_triggers": sum(row["live_noise_comparator"]["detection"] is not None for row in cases),
        "maximum_frozen_run": max(row["frozen_reference"]["longest_run"] for row in cases)}
Path(__file__).with_name("diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"mechanism_change_case": result["mechanism_change_case"],
                  "static_groups": result["static_groups"],
                  "static_triggered": [row for row in result["static_cases"]
                                       if row["frozen_reference"]["detection"]]}, indent=2))
