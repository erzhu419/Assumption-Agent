"""Keep the failed release beside its correction and the unchanged 0.22 baseline."""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATE = "20260909"


def main():
    release_path = ROOT / f"artifacts/hegel_machine_v023_release_cross_version_{DATE}.json"
    corrected_path = ROOT / f"artifacts/hegel_machine_v023_corrected_cross_version_{DATE}.json"
    output = ROOT / f"artifacts/v023_corrected_three_version_effects_{DATE}.json"
    if output.exists():
        raise FileExistsError(output)
    release = {row["seed"]: row for row in json.loads(release_path.read_text())["results"]}
    corrected = json.loads(corrected_path.read_text())["results"]
    rows = []
    for current in corrected:
        for arm in ("methods", "ordinary"):
            versions = {}
            for name, source, side in (("v022", current, "old"),
                    ("v023_release", release[current["seed"]], "new"),
                    ("v023_corrected", current, "new")):
                result = source["arms"][arm]
                versions[name] = {"squared_error": result["by_prediction_loss"]["squared_error"][f"{side}_mean_loss"],
                    "pinball": result["by_prediction_loss"]["pinball"][f"{side}_mean_loss"],
                    "decision_loss": result["decision_loss"][f"{side}_mean_loss"],
                    **{key: result[side][key] for key in ("observation_cost", "compute_cost", "completed_visits")}}
            rows.append({"seed": current["seed"], "arm": arm, "versions": versions})
    aggregate = {}
    for arm in ("methods", "ordinary"):
        selected = [row for row in rows if row["arm"] == arm]
        aggregate[arm] = {name: {
            **{key: mean(row["versions"][name][key] for row in selected)
               for key in ("squared_error", "pinball", "decision_loss")},
            **{key: sum(row["versions"][name][key] for row in selected)
               for key in ("observation_cost", "compute_cost", "completed_visits")}}
            for name in ("v022", "v023_release", "v023_corrected")}
    report = {"sources": [str(release_path), str(corrected_path)],
        "interpretation": "The four fixed seeds were already evaluated before the correction; this is a repair rerun, not new confirmation. Each seed contains the same 58 squared-error and seven risk tasks. Aggregate losses average equal-size seed groups; research costs sum complete sessions.",
        "per_seed": rows, "aggregate": aggregate,
        "corrected_regressions_from_v022": {row["seed"]: row["material_regressions"] for row in corrected}}
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
