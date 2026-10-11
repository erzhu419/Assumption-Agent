"""Read prequential logs to locate paid opportunities for active confirmation."""

import json
from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/sc_kg_joint_revision_competition_v033_20260910/runs"


def read_rows(path):
    data = json.loads(path.read_text())
    recovery = data.get("recovery") or data["adoption_policy"]
    rows = recovery["observations"]
    first = rows[:4]
    noise = mean(row["noise_variance"] for row in first)
    scale = max(1e-7, noise, min(mean(row[key] for row in first)
                               for key in ("default_error", "learned_error")))
    return data, recovery, rows, scale, noise


def compact(row, threshold):
    target = row.get("observation", {}).get("target", row.get("target"))
    return {"stage": row["context"]["stage"], "point": row["context"]["point"],
            "global_index": row["context"].get("global_index"),
            "target": target,
            **{key: row[key] for key in ("default_mean", "learned_mean", "default_error",
                                         "learned_error", "noise_variance")},
            "joint_error_threshold_ratio": min(row["default_error"], row["learned_error"]) / threshold,
            "joint_anomaly": min(row["default_error"], row["learned_error"]) > threshold}


late_path = RUNS / "new_confirmation_1_mechanism_change_noise_0.03_joint_revision.json"
data, recovery, rows, scale, noise = read_rows(late_path)
threshold = 9 * scale
first_index = next(index for index in range(4, len(rows))
                   if min(rows[index]["default_error"], rows[index]["learned_error"]) > threshold)
first = compact(rows[first_index], threshold)
episode = recovery["episodes"][0]
late = {"source": str(late_path.relative_to(ROOT)), "reference_scale": scale,
        "reference_noise": noise, "threshold": threshold,
        "reference_scale_to_noise_ratio": scale / noise,
        "first_joint_anomaly": first,
        "remaining_calls_after_first_anomaly": 64 - first["stage"],
        "earliest_repeat_stage": first["stage"] + 1,
        "remaining_calls_after_repeat": 63 - first["stage"],
        "same_direction_repeat_target_upper_bound": min(first["default_mean"], first["learned_mean"]) - sqrt(threshold),
        "actual_reopen_stage": recovery["events"][0]["stage"],
        "actual_reopen_scope_start": episode["scope_start"],
        "actual_status": episode["status"],
        "proposed_scope_start_global_index": first["global_index"],
        "later_actual_same_point_stages": [row["context"]["stage"] for row in rows[first_index + 1:]
                                           if row["context"]["point"] == first["point"]],
        "rows": [compact(row, threshold) for row in rows]}
stable = []
counts = Counter()
for path in sorted(RUNS.glob("*stable*.json")):
    data, _, rows, scale, noise = read_rows(path)
    anomalies = [compact(row, 9 * scale) for row in rows[4:]
                 if min(row["default_error"], row["learned_error"]) > 9 * scale]
    for item in anomalies:
        item["remaining_calls"] = 64 - item["stage"]
        item["can_reserve_repeat_plus_four_coverage_plus_four_validation"] = item["remaining_calls"] >= 9
    stable.append({"source": str(path.relative_to(ROOT)), "arm": data["arm"],
                   "observation_count": len(rows), "reference_scale": scale,
                   "reference_noise": noise, "anomalies": anomalies})
    counts[(data["arm"], "runs")] += 1
    counts[(data["arm"], "prediction_rows")] += len(rows)
    counts[(data["arm"], "single_joint_anomalies")] += len(anomalies)
    counts[(data["arm"], "anomalies_with_nine_remaining")] += sum(item["remaining_calls"] >= 9 for item in anomalies)
summary = {arm: {key: counts[(arm, key)] for key in
                ("runs", "prediction_rows", "single_joint_anomalies", "anomalies_with_nine_remaining")}
           for arm in sorted({item["arm"] for item in stable})}
report = {"new_simulator_calls": 0,
          "method": "Preserve the existing reference and nine-times threshold. Search paid pre-measurement prediction records for the earliest row jointly exceeding that threshold. Stable run labels only select retrospective controls; no hidden mechanism or evaluation targets enter the signal.",
          "proposal": "At an actionable single joint anomaly, freeze its exact control, both pre-observation means and reference threshold. Spend the next ordinary call measuring that same control. Confirm only if both squared errors against the frozen means again exceed the unchanged threshold, and the residual for each model keeps its original sign. Start the research scope at the original anomaly and retain both paid outcomes. Compare updated GP predictions separately; they cannot overwrite this confirmation question.",
          "limitation": "No existing later observation repeats the late case's first anomalous point. This establishes when a confirmation experiment could run, not what its unseen outcome would be. Static opportunity counts are an initial one-step cost estimate; repeating controls changes subsequent acquisitions. Persistent local prediction failure does not by itself establish a global mechanism change.",
          "stable_summary": summary, "late_case": late, "stable_runs": stable}
Path(__file__).with_name("diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(summary))
print(json.dumps({"first_anomaly_stage": first["stage"], "earliest_repeat_stage": first["stage"] + 1,
                  "actual_reopen_stage": late["actual_reopen_stage"],
                  "proposed_scope_start": late["proposed_scope_start_global_index"]}))
