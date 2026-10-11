"""Run the registered revision branches using the adjacent frozen source."""
from collections import Counter
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
from run_revision_choice_study import protocol, run_study
import hegel_machine.cognition.unified_research as research

RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
BRANCHES = ("unchanged", "change_a", "change_b")


def task_receipt(task, source_c, limits):
    result = {key: task[key] for key in ("task_position", "solved", "mse", "costs",
        "fresh_confirmation_passed", "selected_kind", "unchanged_source_definitions_called")}
    result.update(calls_original_c=bool(set(source_c).intersection(task["nested_call_sources"])),
        nested_call_sources=task["nested_call_sources"],
        action_counts=dict(Counter(action["action"] for action in task.get("actions", []))),
        stopping_reason=task.get("stopping_reason", "unknown"),
        remaining_task_resources={key: limits[key] - task["costs"][key] for key in RESOURCES})
    if "frozen_response_reference" in task:
        result["frozen_response_reference"] = task["frozen_response_reference"]
    if not task["solved"] or not task["fresh_confirmation_passed"]:
        result["action_tail"] = [{key: action[key] for key in (
            "action", "status", "reason", "accepted", "loss", "confirmation_loss", "channels") if key in action}
            for action in task.get("actions", [])[-8:]]
    return result


def execute(case):
    path = ROOT / f"case_{case['apparatus_seed']}_{case['profile']}.json"
    if path.exists():
        raise FileExistsError(path)
    start = monotonic()
    try:
        result = run_study(protocol(**case))
        path.write_text(json.dumps(result, indent=2) + "\n")
        config = result["protocol"]["research_config"]
        limits = {"observation_cost": config["observation_budget"],
            "measurement_cost": result["protocol"]["measurement_budget"], "compute_cost": config["compute_budget"]}
        compact = {}
        for name, report in {"prefix": result["prefix"], **result["results"]}.items():
            compact[name] = {key: report[key] for key in (
                "solved", "fresh_confirmed", "mean_mse", "costs", "evaluation_costs",
                "training_cpu_seconds", "evaluation_cpu_seconds", "revisit_count", "screening_compute_cost")}
            compact[name]["tasks"] = [task_receipt(task, result["source_ids"]["c"], limits)
                for task in report["tasks"]]
            if name != "prefix":
                compact[name].update({key: report[key] for key in (
                    "total_with_prefix", "reference_evaluation_costs", "reference_evaluation_cpu_seconds")})
        return {**case, "status": "completed", "elapsed_seconds": monotonic() - start,
            "source_ids": result["source_ids"], "reports": compact}
    except Exception:
        failure = {**case, "status": "execution_failed", "traceback": traceback.format_exc(),
            "elapsed_seconds": monotonic() - start}
        if not path.exists():
            path.write_text(json.dumps(failure, indent=2) + "\n")
        return failure


def summed_costs(items, field="costs"):
    return {key: sum(item[field][key] for item in items) for key in RESOURCES}


def stage_totals(tasks):
    actions = Counter()
    for task in tasks:
        actions.update(task["action_counts"])
    result = {"tasks": len(tasks), "solved": sum(t["solved"] for t in tasks),
        "fresh_confirmed": sum(t["fresh_confirmation_passed"] for t in tasks),
        "mean_mse": mean(t["mse"] for t in tasks), "costs": summed_costs(tasks),
        "remaining_task_resources": summed_costs(tasks, "remaining_task_resources"),
        "exact_definition_calls": {source: sum(t["unchanged_source_definitions_called"][source]
            for t in tasks) for source in ("a", "b")},
        "original_c_calls": sum(t["calls_original_c"] for t in tasks),
        "selected_kind_counts": dict(Counter(t["selected_kind"] for t in tasks)),
        "action_counts": dict(actions),
        "stopping_reason_counts": dict(Counter(t["stopping_reason"] for t in tasks))}
    if all("frozen_response_reference" in task for task in tasks):
        references = [task["frozen_response_reference"] for task in tasks]
        result["frozen_response_reference"] = {"mean_mse": mean(t["mse"] for t in references),
            "costs": summed_costs(references)}
    return result


def totals(rows):
    reports = {}
    for name in ("prefix", *BRANCHES):
        items = [row["reports"][name] for row in rows]
        reports[name] = {"tasks": sum(len(item["tasks"]) for item in items),
            "solved": sum(item["solved"] for item in items),
            "fresh_confirmed": sum(item["fresh_confirmed"] for item in items),
            "mean_mse": mean(item["mean_mse"] for item in items),
            "costs": summed_costs(items), "evaluation_costs": summed_costs(items, "evaluation_costs"),
            "training_cpu_seconds": sum(item["training_cpu_seconds"] for item in items),
            "evaluation_cpu_seconds": sum(item["evaluation_cpu_seconds"] for item in items),
            "revisits": sum(item["revisit_count"] for item in items),
            "screening_compute_cost": sum(item["screening_compute_cost"] for item in items),
            "stages": [stage_totals([item["tasks"][i] for item in items])
                for i in range(3 if name == "prefix" else 2)]}
        if name != "prefix":
            reports[name].update(total_with_prefix=summed_costs(items, "total_with_prefix"),
                reference_evaluation_costs=summed_costs(items, "reference_evaluation_costs"),
                reference_evaluation_cpu_seconds=sum(item["reference_evaluation_cpu_seconds"] for item in items))
    actual = list(reports.values())
    return {"reports": reports, "research_tasks_counting_prefix_once": sum(item["tasks"] for item in actual),
        "research_costs_counting_prefix_once": summed_costs(actual),
        "terminal_costs_counting_prefix_once": summed_costs(actual, "evaluation_costs"),
        "training_cpu_seconds_counting_prefix_once": sum(item["training_cpu_seconds"] for item in actual),
        "terminal_cpu_seconds_counting_prefix_once": sum(item["evaluation_cpu_seconds"] for item in actual),
        "fixed_reference_predictions": 6 * len(rows),
        "fixed_reference_evaluation_costs": summed_costs([reports[name] for name in BRANCHES], "reference_evaluation_costs"),
        "fixed_reference_evaluation_cpu_seconds": sum(reports[name]["reference_evaluation_cpu_seconds"] for name in BRANCHES)}


def uncertainty(rows):
    rng = Random(2026100503)
    groups = [[i for i, row in enumerate(rows) if row["profile"] == profile]
        for profile in ("development", "confirmation")]
    samples = [sum((rng.choices(group, k=len(group)) for group in groups), []) for _ in range(10000)]
    result = {}
    for name in ("change_a", "change_b"):
        result[name] = {}
        for resource in RESOURCES:
            differences = [row["reports"][name]["tasks"][0]["costs"][resource]
                - row["reports"]["unchanged"]["tasks"][0]["costs"][resource] for row in rows]
            sampled = sorted(sum(differences[i] for i in sample) / len(sample) for sample in samples)
            def percentile(q):
                position = q * (len(sampled) - 1)
                left = int(position)
                return sampled[left] + (position - left) * (sampled[min(left + 1, len(sampled) - 1)] - sampled[left])
            result[name][resource] = {"mean_first_successor_change_minus_unchanged": mean(differences),
                "paired_95_percent_interval": [percentile(.025), percentile(.975)],
                "per_case_differences": differences}
    return {"interpretation": "Resource response to an environmental change, not an algorithm superiority comparison.",
        "samples": 10000, "profile_stratified": True, "comparisons": result}


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
                "status": row["status"], "solved": {name: report["solved"]
                    for name, report in row.get("reports", {}).items()},
                "elapsed_seconds": round(row["elapsed_seconds"], 2)}), flush=True)
    rows.sort(key=lambda row: row["apparatus_seed"])
    completed = [row for row in rows if row["status"] == "completed"]
    failures = [{"apparatus_seed": row["apparatus_seed"], "profile": row["profile"],
        "branch": name, "stage": task["task_position"], **task}
        for row in completed for name, report in row["reports"].items() for task in report["tasks"]
        if not task["solved"] or not task["fresh_confirmation_passed"]]
    summary = {"protocol": plan, "planned_cases": len(plan["cases"]), "completed_cases": len(completed),
        "planned_research_terminal_tasks": 9 * len(plan["cases"]),
        "planned_fixed_reference_predictions": 6 * len(plan["cases"]),
        "execution_failures": [row for row in rows if row["status"] != "completed"],
        "research_failures": failures, "rows": rows,
        "elapsed_seconds": monotonic() - start, "external_model_calls": 0}
    if completed:
        summary["totals"] = totals(completed)
        summary["profiles"] = {profile: totals([row for row in completed if row["profile"] == profile])
            for profile in ("development", "confirmation") if any(row["profile"] == profile for row in completed)}
    if len(completed) == len(plan["cases"]):
        summary["paired_uncertainty"] = uncertainty(completed)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "planned_cases", "completed_cases", "planned_research_terminal_tasks", "elapsed_seconds", "external_model_calls")}), flush=True)


if __name__ == "__main__":
    main()
