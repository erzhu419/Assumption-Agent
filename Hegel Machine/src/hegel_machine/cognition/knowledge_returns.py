"""Measured downstream returns of executable knowledge under acquired conditions."""

from copy import deepcopy
from math import sqrt

from .method_conditions import method_profile
from .language import Program


def _solve_linear(matrix, values):
    """Solve a small ridge system used only for a measured response summary."""
    width = len(matrix[0]) if matrix else 0
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row, value in zip(matrix, values):
        for left in range(width):
            rhs[left] += row[left] * value
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    for index in range(width):
        gram[index][index] += 1e-8
    for pivot in range(width):
        row = max(range(pivot, width), key=lambda candidate: abs(gram[candidate][pivot]))
        if abs(gram[row][pivot]) < 1e-12:
            continue
        gram[pivot], gram[row] = gram[row], gram[pivot]
        rhs[pivot], rhs[row] = rhs[row], rhs[pivot]
        scale = gram[pivot][pivot]
        gram[pivot] = [item / scale for item in gram[pivot]]
        rhs[pivot] /= scale
        for row in range(width):
            if row == pivot:
                continue
            scale = gram[row][pivot]
            if not scale:
                continue
            gram[row] = [left - scale * right for left, right in zip(gram[row], gram[pivot])]
            rhs[row] -= scale * rhs[pivot]
    return rhs


def _response_residual(observations):
    """Measure unexplained paid-response variation without another observation.

    Fit the measured response coordinates once and evaluate the in-sample
    residuals. The signature is a coarse transfer gate, so leave-one-out
    refitting adds repeated cubic solves without changing the decision signal
    used here.
    """
    names = sorted({name for row in observations for name in row.features})
    depth = max((len(row.history) for row in observations), default=0)
    columns = [(0, name) for name in names]
    columns.extend((lag, name) for lag in range(1, depth + 1) for name in names)
    matrix, values = [], []
    for row in observations:
        vector = [1.0]
        for lag, name in columns:
            source = row.features if lag == 0 else row.history[-lag] if len(row.history) >= lag else {}
            vector.append(float(source.get(name, 0.0)))
        matrix.append(vector)
        values.append(float(row.target))
    if not matrix:
        return None
    if len(matrix) < 3:
        return None
    fitted = _solve_linear(matrix, values)
    errors = [sum(coef * value for coef, value in zip(fitted, row)) - target
              for row, target in zip(matrix, values)]
    return sqrt(sum(error * error for error in errors) / len(errors))


def _response_signature(observations):
    """Keep a coarse measured response regime in a transfer condition.

    Structural coverage alone cannot distinguish two laws with the same
    controls and history but different response coefficients.  Quantized
    paid-response moments and a single-fit residual are deliberately coarse:
    they gate reuse when the measured regime materially changes without
    turning the return record into a model of the hidden mechanism.
    """
    targets = [float(row.target) for row in observations]
    if not targets:
        return None
    step = .05
    residual = _response_residual(observations)
    return {"mean_bin": int(round(sum(targets) / len(targets) / step)),
            "spread_bin": int(round((max(targets) - min(targets)) / step)),
            # Noise is a smaller signal than the response range; keep its
            # gate finer than the coarse level and spread bins.
            "residual_bin": int(round((residual or 0.) / .01))}


def transfer_condition(spec, observations):
    """Describe an adaptation's information, without names or hidden task labels."""
    profile = method_profile(spec, observations)
    data = profile["data"]
    return {"objective": profile["objective"], "data": {key: deepcopy(data[key]) for key in (
        "feature_dimensions", "control_dimensions", "max_history", "linear_data_support",
        "response_levels", "coverage", "conditional_coverage", "object_fields")},
            "response_signature": _response_signature(observations)}


def return_scope_matches(record, *, program=None, source_ids=()):
    """Match an executable offer, or aggregate singleton source history."""
    recorded = set(record.get("joint_source_ids", ()))
    # Older singleton receipts may not have ``joint_source_ids`` but still
    # carry their producer.  Use that explicit owner when a caller supplies a
    # source scope; source-local legacy fixtures without either field remain
    # usable because their history already supplies the context.
    if not recorded and record.get("source_id") is not None:
        recorded = {record["source_id"]}
    requested = set(source_ids)
    if requested and recorded and requested != recorded:
        return False
    if program is None:
        return len(recorded) <= 1
    canonical = program.expanded().simplified()
    return any(Program.from_dict(item["program"]).expanded().simplified() == canonical
               for item in record.get("comparisons", ()) if item.get("program"))


def matching_returns(history, condition, *, program=None, source_ids=()):
    """Use the same condition and binding scope for ranking and withdrawal."""
    return [item for item in history if item.get("kind") == "research_return"
            and item.get("condition") == condition and item.get("comparisons")
            and return_scope_matches(item, program=program, source_ids=source_ids)]


def return_priority(history, condition, *, program=None, source_ids=()):
    """Prefer measured retained prediction gain per paid unit of adaptation work.

    Unknown conditions remain trials. Shared work is kept as shared work; this
    function never assigns a whole visit's bill to a source that merely appeared.
    A concrete offer only inherits returns of the same canonical program.
    Without a program, singleton returns aggregate for source ranking and
    shared returns are excluded.
    """
    relevant = [item for item in matching_returns(history, condition,
                program=program, source_ids=source_ids)
                # A retained unresolved gain is provisional, but an actual
                # negative comparison remains actionable evidence for
                # withdrawing that source under this condition.
                if (item.get("confirmation_status", "confirmed") == "confirmed"
                     or item.get("retained_normalized_gain", 0.) < 0.)]
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
        joint_sources = tuple(sorted(set(attempt.get("joint_source_ids", [source_id]))))
        lineage_id = attempt.get("lineage_id")
        # Equal singleton offers in the same lineage are alternative origins
        # of one measured program. Preserve explicit joint bindings so their
        # gain cannot overwrite a separately tested singleton offer.
        scope_key = (joint_sources if len(joint_sources) > 1 else (), lineage_id)
        for key, row in attempt["comparisons"].items():
            comparison_key = (*key, scope_key)
            source["comparisons"][comparison_key] = row
            compared.setdefault(comparison_key, set()).update(joint_sources)
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
                "lineage_id": key[-1][1],
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
