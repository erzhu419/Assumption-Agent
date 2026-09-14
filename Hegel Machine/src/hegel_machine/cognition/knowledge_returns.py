"""Measured downstream returns of executable knowledge under acquired conditions."""

from copy import deepcopy

from .method_conditions import method_profile
from .language import Program


def transfer_condition(spec, observations):
    """Describe an adaptation's information, without names or hidden task labels."""
    profile = method_profile(spec, observations)
    data = profile["data"]
    return {"objective": profile["objective"], "data": {key: deepcopy(data[key]) for key in (
        "feature_dimensions", "control_dimensions", "max_history", "linear_data_support",
        "response_levels", "coverage", "conditional_coverage", "object_fields")}}


def return_priority(history, condition):
    """Prefer measured retained prediction gain per paid unit of adaptation work.

    Unknown conditions remain trials. Shared work is kept as shared work; this
    function never assigns a whole visit's bill to a source that merely appeared.
    """
    relevant = [item for item in history if item.get("kind") == "research_return"
                and item.get("condition") == condition and item.get("comparisons")]
    if not relevant:
        return (0.0, 0.0)
    gain = sum(item["retained_normalized_gain"] for item in relevant)
    cost = sum(item["adaptation_compute"] + item["comparison_compute"] for item in relevant)
    return (-gain / max(1, cost), cost / len(relevant))


def producer_outcomes(episode, retained_programs):
    """Summarize each producer's actual later work, sharing neither gain nor bills twice.

    A second source offering the same tested program is an alternative origin.
    Its shared fit and result remain visible, but do not become exclusive value
    for either production method. No whole-visit expenditure enters this return.
    """
    sources = {}
    compared = {}
    for attempt in episode["attempts"].values():
        source_id = attempt["source_id"]
        source = sources.setdefault(source_id, {"conditions": [], "comparisons": {}})
        if attempt["condition"] not in source["conditions"]:
            source["conditions"].append(deepcopy(attempt["condition"]))
        for key, row in attempt["comparisons"].items():
            source["comparisons"][key] = row
            compared.setdefault(key, set()).add(source_id)
    for source_id, source in sources.items():
        exclusive_work, shared_work = [], []
        for row in episode["work"].values():
            if source_id in row["source_ids"]:
                preserved = row.get("preserved_work", [])
                fitting = {"compute_cost": row["compute_cost"] - sum(item["compute_cost"] for item in preserved)}
                (exclusive_work if len(row["source_ids"]) == 1 else shared_work).append(fitting)
                for item in preserved:
                    if source_id in item["source_ids"]:
                        (exclusive_work if len(item["source_ids"]) == 1 else shared_work).append(item)
        comparisons = []
        for key, row in source["comparisons"].items():
            retained = Program.from_dict(row["program"]).expanded().simplified() in retained_programs
            gain = (row["baseline_loss"] - row["loss"]) / max(abs(row["baseline_loss"]), abs(row["loss"]), 1e-8)
            comparisons.append({**deepcopy(row), "retained": retained,
                "retained_gain": gain if retained else min(0., gain),
                "source_ids": sorted(compared[key]), "exclusive": len(compared[key]) == 1})
        exclusive = [row for row in comparisons if row["exclusive"]]
        shared = [row for row in comparisons if not row["exclusive"]]
        def average_gain(rows):
            count = sum(max(1, row["evidence_rows"]) for row in rows)
            return sum(row["retained_gain"] * max(1, row["evidence_rows"]) for row in rows) / count if count else None
        yield source_id, {"source_concept_id": source_id, "downstream_task": episode["task_id"],
            "downstream_conditions": source["conditions"], "retained_gain": average_gain(exclusive),
            "shared_gain": average_gain(shared), "comparisons": comparisons,
            "exclusive_compute": sum(row["compute_cost"] for row in exclusive_work) + sum(row["compute_cost"] for row in exclusive),
            "shared_compute": sum(row["compute_cost"] for row in shared_work) + sum(row["compute_cost"] for row in shared),
            "exclusive_adaptation_compute": sum(row["compute_cost"] for row in exclusive_work),
            "shared_adaptation_compute": sum(row["compute_cost"] for row in shared_work)}
