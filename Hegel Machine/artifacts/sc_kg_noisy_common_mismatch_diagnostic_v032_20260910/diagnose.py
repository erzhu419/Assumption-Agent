"""Replay existing pre-measurement forecast errors; acquire no new outcomes."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/sc_kg_joint_revision_v032_20260910/runs"
OLD = ROOT / "artifacts/sc_kg_prior_adoption_v031_20260910"
MINIMUM_ERROR = 1e-7
BATCH_SIZE = 4
MULTIPLIER = 9


def replay(path, keep_rows=False):
    payload = json.loads(path.read_text())
    policy = payload.get("recovery") or payload["adoption_policy"]
    rows = policy["observations"]
    initial = rows[:BATCH_SIZE]
    initial_mse = {key: mean(row[key] for row in initial)
                   for key in ("default_error", "learned_error")}
    noise = mean(row["noise_variance"] for row in initial)
    reference = max(MINIMUM_ERROR, min(initial_mse.values()), noise)
    threshold = MULTIPLIER * reference
    streak = longest = 0
    consecutive_first = None
    for row in rows[BATCH_SIZE:]:
        streak = streak + 1 if min(row["default_error"], row["learned_error"]) > threshold else 0
        longest = max(longest, streak)
        if streak >= BATCH_SIZE and consecutive_first is None:
            consecutive_first = row["context"]["stage"]
    windows = []
    for start in range(BATCH_SIZE, len(rows) - BATCH_SIZE + 1, BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        default_mse = mean(row["default_error"] for row in batch)
        learned_mse = mean(row["learned_error"] for row in batch)
        anomalous = [row["context"]["stage"] for row in batch
                     if min(row["default_error"], row["learned_error"]) > threshold]
        above = min(default_mse, learned_mse) > threshold
        windows.append({"start_stage": batch[0]["context"]["stage"],
                        "end_stage": batch[-1]["context"]["stage"],
                        "default_mse": default_mse, "learned_mse": learned_mse,
                        "mean_live_noise": mean(row["noise_variance"] for row in batch),
                        "minimum_mse_threshold_ratio": min(default_mse, learned_mse) / threshold,
                        "anomalous_stages": anomalous, "plain_batch_alert": above,
                        "repeated_batch_alert": above and len(anomalous) >= 2})
    result = {"source": str(path.relative_to(ROOT)), "observation_count": len(rows),
              "reference": reference, "threshold": threshold,
              "initial_mse": initial_mse, "initial_mean_noise": noise,
              "longest_consecutive_failures": longest,
              "consecutive_first_stage": consecutive_first,
              "plain_batch_first_stage": next((w["end_stage"] for w in windows if w["plain_batch_alert"]), None),
              "repeated_batch_first_stage": next((w["end_stage"] for w in windows if w["repeated_batch_alert"]), None),
              "maximum_batch_threshold_ratio": max(w["minimum_mse_threshold_ratio"] for w in windows)}
    if keep_rows:
        result["windows"] = windows
        result["rows"] = [{"stage": row["context"]["stage"],
                           **{key: row[key] for key in ("default_error", "learned_error", "noise_variance")}}
                          for row in rows]
    return result


negative = replay(RUNS / "forward_1_mechanism_change_noise_0.03_joint_revision.json", True)
current_controls = [replay(path) for path in sorted(RUNS.glob("*_stable_*_frozen_v031_auto.json"))]
earlier_controls = [replay(path) for path in sorted(OLD.glob("*_auto.json"))]
controls = current_controls + earlier_controls
summary = {"new_simulator_calls": 0, "current_static_runs": len(current_controls),
           "earlier_static_runs": len(earlier_controls),
           "static_prediction_rows": sum(item["observation_count"] for item in controls),
           "plain_batch_static_alerts": sum(item["plain_batch_first_stage"] is not None for item in controls),
           "repeated_batch_static_alerts": sum(item["repeated_batch_first_stage"] is not None for item in controls),
           "maximum_static_batch_threshold_ratio": max(item["maximum_batch_threshold_ratio"] for item in controls)}
report = {"method": "First four prequential outcomes freeze max(1e-7, best model MSE, mean reported noise); compare later nonoverlapping four-row batches to nine times that reference. Decisions use forecast errors and reported noise only.",
          "limitation": "Offline detection replay does not establish downstream recovery success or general false-alarm rates. Initial underfitting raises the frozen reference. A plain batch mean alone can trigger on one extreme outlier; the production rule also requires two jointly anomalous rows.",
          "summary": summary, "negative_case": negative,
          "current_controls": current_controls, "earlier_controls": earlier_controls}
Path(__file__).with_name("diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(summary))
print(json.dumps({key: negative[key] for key in ("reference", "threshold", "consecutive_first_stage", "repeated_batch_first_stage")}))
