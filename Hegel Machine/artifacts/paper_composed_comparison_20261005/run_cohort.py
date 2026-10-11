"""Execute the registered paired cohort using the adjacent frozen source."""
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
from run_composed_offline_comparison import protocol, run_benchmark
import hegel_machine.cognition.unified_research as research

RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def execute(case):
    path = ROOT / f"case_{case['apparatus_seed']}_{case['profile']}.json"
    if path.exists():
        raise FileExistsError(path)
    start = monotonic()
    try:
        result = run_benchmark(protocol(**case))
        path.write_text(json.dumps(result, indent=2) + "\n")
        return {**case, "status": "completed", "elapsed_seconds": monotonic() - start,
            "arms": {arm: {key: values[key] for key in (
                "solved", "fresh_confirmed", "mean_mse", "costs", "evaluation_costs",
                "training_cpu_seconds", "evaluation_cpu_seconds", "revisit_count")}
                for arm, values in result["results"].items()},
            "paired": result["paired_full_vs_control"]}
    except Exception:
        failure = {**case, "status": "execution_failed", "traceback": traceback.format_exc(),
                   "elapsed_seconds": monotonic() - start}
        path.write_text(json.dumps(failure, indent=2) + "\n")
        return failure


def totals(rows, arms):
    return {arm: {
        "solved": sum(row["arms"][arm]["solved"] for row in rows),
        "tasks": 6 * len(rows),
        "all_six_solved_instances": sum(row["arms"][arm]["solved"] == 6 for row in rows),
        "fresh_confirmed": sum(row["arms"][arm]["fresh_confirmed"] for row in rows),
        "mean_mse": mean(row["arms"][arm]["mean_mse"] for row in rows),
        "costs": {key: sum(row["arms"][arm]["costs"][key] for row in rows) for key in RESOURCES},
        "evaluation_costs": {key: sum(row["arms"][arm]["evaluation_costs"][key] for row in rows)
                             for key in RESOURCES},
        "training_cpu_seconds": sum(row["arms"][arm]["training_cpu_seconds"] for row in rows),
        "evaluation_cpu_seconds": sum(row["arms"][arm]["evaluation_cpu_seconds"] for row in rows),
        "revisits": sum(row["arms"][arm]["revisit_count"] for row in rows),
    } for arm in arms}


def paired_uncertainty(rows, arms):
    rng = Random(20261005)
    groups = [[i for i, row in enumerate(rows) if row["profile"] == profile]
              for profile in ("development", "confirmation")]
    selections = [sum((rng.choices(group, k=len(group)) for group in groups), []) for _ in range(10000)]
    metrics = ("solved", "mean_mse", *RESOURCES, "training_cpu_seconds")
    answer = {}
    for arm in arms:
        if arm == "full":
            continue
        answer[arm] = {}
        for metric in metrics:
            def value(row, arm):
                item = row["arms"][arm]
                return item["costs"][metric] if metric in RESOURCES else item[metric]
            differences = [value(row, "full") - value(row, arm) for row in rows]
            sampled = sorted(sum(differences[i] for i in sample) / len(sample) for sample in selections)
            def percentile(q):
                position = q * (len(sampled) - 1)
                left = int(position)
                return sampled[left] + (position - left) * (sampled[min(left + 1, len(sampled) - 1)] - sampled[left])
            answer[arm][metric] = {"mean_full_minus_control_per_instance": mean(differences),
                "paired_95_percent_interval": [percentile(.025), percentile(.975)]}
        answer[arm]["same_quality_full_pareto_instances"] = sum(
            row["paired"][arm]["both_solve_all_with_full_pareto_saving"] for row in rows)
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
                "status": row["status"], "solved": {arm: result["solved"]
                    for arm, result in row.get("arms", {}).items()},
                "elapsed_seconds": round(row["elapsed_seconds"], 2)}), flush=True)
    rows.sort(key=lambda row: row["apparatus_seed"])
    completed = [row for row in rows if row["status"] == "completed"]
    summary = {"protocol": plan, "planned_cases": len(plan["cases"]), "completed_cases": len(completed),
        "execution_failures": [row for row in rows if row["status"] != "completed"],
        "rows": rows, "elapsed_seconds": monotonic() - start, "external_model_calls": 0}
    if completed:
        summary["totals"] = totals(completed, plan["arms"])
        summary["profiles"] = {profile: totals([row for row in completed if row["profile"] == profile], plan["arms"])
            for profile in ("development", "confirmation") if any(row["profile"] == profile for row in completed)}
    if len(completed) == len(plan["cases"]):
        summary["paired_uncertainty"] = paired_uncertainty(completed, plan["arms"])
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
        if key in ("planned_cases", "completed_cases", "elapsed_seconds", "external_model_calls")}), flush=True)


if __name__ == "__main__":
    main()
