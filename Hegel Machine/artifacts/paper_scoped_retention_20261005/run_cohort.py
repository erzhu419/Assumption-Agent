"""Run the specified retention interventions from the adjacent frozen source."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/scripts"))
from run_scoped_retention_comparison import protocol, run_benchmark
import hegel_machine.cognition.unified_research as research

RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
REFERENCE = "conditional_retention"


def execute(case):
    path = ROOT / f"case_{case['apparatus_seed']}_{case['profile']}.json"
    if path.exists():
        raise FileExistsError(path)
    start = monotonic()
    try:
        result = run_benchmark(protocol(**case))
        path.write_text(json.dumps(result, indent=2) + "\n")
        arms = {}
        for arm, report in result["results"].items():
            arms[arm] = {key: report[key] for key in (
                "solved", "fresh_confirmed", "mean_mse", "costs", "evaluation_costs",
                "training_cpu_seconds", "evaluation_cpu_seconds", "revisit_count")}
            arms[arm]["successors"] = [{key: task[key] for key in (
                "solved", "mse", "costs", "fresh_confirmation_passed")}
                for task in report["tasks"][3:]]
        return {**case, "status": "completed", "elapsed_seconds": monotonic() - start,
            "activated": result["challenge"]["activated"], "arms": arms}
    except Exception:
        failure = {**case, "status": "execution_failed", "traceback": traceback.format_exc(),
                   "elapsed_seconds": monotonic() - start}
        path.write_text(json.dumps(failure, indent=2) + "\n")
        return failure


def totals(rows, arms):
    result = {}
    for arm in arms:
        reports = [row["arms"][arm] for row in rows]
        result[arm] = {
            "tasks": 6 * len(rows), "solved": sum(v["solved"] for v in reports),
            "fresh_confirmed": sum(v["fresh_confirmed"] for v in reports),
            "mean_mse": mean(v["mean_mse"] for v in reports),
            "costs": {key: sum(v["costs"][key] for v in reports) for key in RESOURCES},
            "evaluation_costs": {key: sum(v["evaluation_costs"][key] for v in reports) for key in RESOURCES},
            "training_cpu_seconds": sum(v["training_cpu_seconds"] for v in reports),
            "evaluation_cpu_seconds": sum(v["evaluation_cpu_seconds"] for v in reports),
            "revisits": sum(v["revisit_count"] for v in reports),
            "successors": [{"phase": name,
                "solved": sum(v["successors"][i]["solved"] for v in reports),
                "mean_mse": mean(v["successors"][i]["mse"] for v in reports),
                "costs": {key: sum(v["successors"][i]["costs"][key] for v in reports)
                          for key in RESOURCES}}
                for i, name in enumerate(("first_changed", "original_return", "changed_return"))]}
    return result


def uncertainty(rows, arms):
    rng = Random(2026100502)
    groups = [[i for i, row in enumerate(rows) if row["profile"] == profile]
              for profile in ("development", "confirmation")]
    samples = [sum((rng.choices(group, k=len(group)) for group in groups), []) for _ in range(10000)]
    metrics = ("solved", "mean_mse", *RESOURCES, "training_cpu_seconds")
    answer = {}
    for arm in arms:
        if arm == REFERENCE:
            continue
        answer[arm] = {}
        for phase in ("sequence", "first_changed", "original_return", "changed_return"):
            comparisons = {}
            for metric in metrics:
                if phase != "sequence" and metric == "training_cpu_seconds":
                    continue
                def value(row, name):
                    item = row["arms"][name]
                    if phase != "sequence":
                        item = item["successors"][("first_changed", "original_return", "changed_return").index(phase)]
                    key = "mse" if phase != "sequence" and metric == "mean_mse" else metric
                    return item["costs"][key] if key in RESOURCES else item[key]
                differences = [value(row, REFERENCE) - value(row, arm) for row in rows]
                sampled = sorted(sum(differences[i] for i in sample) / len(sample) for sample in samples)
                def percentile(q):
                    position = q * (len(sampled) - 1)
                    left = int(position)
                    return sampled[left] + (position - left) * (sampled[min(left + 1, len(sampled) - 1)] - sampled[left])
                comparisons[metric] = {"mean_conditional_minus_control_per_instance": mean(differences),
                    "paired_95_percent_interval": [percentile(.025), percentile(.975)]}
            answer[arm][phase] = comparisons
    return answer


def main():
    assert Path(research.__file__).is_relative_to(ROOT / "runtime")
    plan = json.loads((ROOT / "protocol.json").read_text())
    start, rows = monotonic(), []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(execute, case) for case in plan["cases"]]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({"apparatus_seed": row["apparatus_seed"], "profile": row["profile"],
                "status": row["status"], "activated": row.get("activated"),
                "solved": {arm: report["solved"] for arm, report in row.get("arms", {}).items()},
                "elapsed_seconds": round(row["elapsed_seconds"], 2)}), flush=True)
    rows.sort(key=lambda row: row["apparatus_seed"])
    completed = [row for row in rows if row["status"] == "completed"]
    summary = {"protocol": plan, "planned_cases": len(plan["cases"]), "completed_cases": len(completed),
        "execution_failures": [row for row in rows if row["status"] != "completed"],
        "activated_cases": sum(row["activated"] for row in completed), "rows": rows,
        "elapsed_seconds": monotonic() - start, "external_model_calls": 0}
    if completed:
        summary["totals"] = totals(completed, plan["arms"])
        summary["profiles"] = {profile: totals([row for row in completed if row["profile"] == profile], plan["arms"])
            for profile in ("development", "confirmation") if any(row["profile"] == profile for row in completed)}
    if len(completed) == len(plan["cases"]):
        summary["paired_uncertainty"] = uncertainty(completed, plan["arms"])
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "planned_cases", "completed_cases", "activated_cases", "elapsed_seconds", "external_model_calls")}), flush=True)


if __name__ == "__main__":
    main()
