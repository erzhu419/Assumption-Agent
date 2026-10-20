"""Empirical ambiguity of the actual inputs supplied to a scalar predictor."""

from math import fsum


REPRESENTATION_INTERFACE = """
An optional explicit representation uses represent(row), predict_features(features),
and optionally fit_features(features, targets) in the model file. represent receives
public features/history without the current target and returns JSON numeric values
(a vector, nested vectors, or a named object). The runtime actually sends those
values to fit_features and predict_features; ordinary predict(row) is not used on
this path. Fitting uses only the selected paid training rows. Expose the complete
input used by the downstream deterministic predictor, including any context it
needs. Evaluation returns collisions between identical represented inputs and
different observed labels, with row witnesses and the minimum empirical row-mean
squared error for any deterministic function of those inputs. The diagnostic does
not choose a new representation or establish future risk. It does not verify hidden
reads or mutable state in arbitrary Python. Ordinary predict(row) remains available;
without this decomposition the representation is not assessed.
"""


def _value_key(value):
    if isinstance(value, dict):
        return ("object", tuple((key, _value_key(item)) for key, item in sorted(value.items())))
    if isinstance(value, list):
        return ("array", tuple(_value_key(item) for item in value))
    return ("scalar", value)


def representation_ambiguity(representations, rows, *, scope="evaluated_rows", max_witnesses=3):
    """Compute the exact finite-data MSE floor under a deterministic representation.

    Groups compare complete representation values, not rounded features. Targets
    enter only this diagnostic, after the subprocess has produced predictions.
    """
    if len(representations) != len(rows):
        raise ValueError("represented inputs and observed rows must have equal lengths")
    groups = {}
    for index, value in enumerate(representations):
        groups.setdefault(_value_key(value), []).append(index)
    conflicts = []
    for indices in groups.values():
        targets = [float(rows[index].target if hasattr(rows[index], "target")
                         else rows[index]["target"]) for index in indices]
        if min(targets) == max(targets):
            continue
        mean = fsum(targets) / len(targets)
        sse = fsum((target - mean) ** 2 for target in targets)
        conflicts.append((sse, indices, mean))
    total_sse = fsum(item[0] for item in conflicts)
    witnesses = []
    for sse, indices, mean in sorted(conflicts, reverse=True)[:max_witnesses]:
        def target(index):
            row = rows[index]
            return row.target if hasattr(row, "target") else row["target"]
        # Extremes establish the conflict; one intermediate example can expose
        # distinctions in public history without dumping a complete trajectory.
        ordered = sorted(indices, key=target)
        selected = list(dict.fromkeys([ordered[0], ordered[len(ordered) // 2], ordered[-1]]))
        examples = []
        for index in selected:
            row = rows[index]
            examples.append({"row_index": index,
                "features": row.features if hasattr(row, "features") else row["features"],
                "history": row.history if hasattr(row, "history") else row.get("history", []),
                "target": target(index)})
        witnesses.append({"representation": representations[indices[0]],
                          "group_size": len(indices), "mean_target": mean,
                          "squared_error_floor": sse, "rows": examples})
    return {"status": "conflicting_targets" if conflicts else "no_exact_conflicts",
            "scope": scope, "rows": len(rows),
            "loss_kind": "mse", "weighting": "uniform_rows",
            "representation_contract": "actual_downstream_inputs",
            "distinct_representations": len(groups), "conflicting_groups": len(conflicts),
            "empirical_mse_floor": total_sse / max(1, len(rows)), "witnesses": witnesses,
            "applies_to": "Deterministic functions of the actual captured inputs on these rows; "
                "no check of hidden reads or mutable state, and no future-risk claim."}
