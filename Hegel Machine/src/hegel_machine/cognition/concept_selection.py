"""Choose learned computations by their value on already acquired trajectories."""

from __future__ import annotations

from statistics import mean

from .evaluation import initial_prediction, point_loss, predict
from .language import Program
from .risk_search import empirical_quantile, trajectory_blocks
from .synthesis import _solve


def is_raw_affine(program: Program) -> bool:
    """A linear combination of existing coordinates adds no search primitive."""
    program = program.expanded()

    def degree(node):
        if node.op == "const":
            return 0
        if node.op in {"feature", "lag", "available"}:
            return 1
        values = [degree(child) for child in node.children]
        if node.op in {"add", "sub"}:
            return max(values)
        if node.op == "mul":
            return sum(values)
        return 2

    return degree(program) <= 1


def _fit_predict(columns, targets, training, *, quantile=0.5):
    centers = [mean(column[index] for index in training) for column in columns]
    scales = [max(abs(column[index] - center) for index in training) or 1.0
              for column, center in zip(columns, centers)]
    design = [[1.0, *((column[index] - center) / scale
                     for column, center, scale in zip(columns, centers, scales))]
              for index in range(len(targets))]
    width = len(columns) + 1
    gram = [[sum(design[index][left] * design[index][right] for index in training)
             for right in range(width)] for left in range(width)]
    rhs = [sum(design[index][column] * targets[index] for index in training) for column in range(width)]
    for index in range(1, width):
        gram[index][index] += 1e-8
    coefficients = _solve(gram, rhs)
    predictions = [sum(value * coefficient for value, coefficient in zip(row, coefficients)) for row in design]
    if quantile != 0.5:
        offset = empirical_quantile([targets[index] - predictions[index] for index in training], quantile)
        predictions = [prediction + offset for prediction in predictions]
    return predictions


def _raw_coordinates(observations, *, include_history=True):
    names = sorted(set.intersection(*(set(row.features) for row in observations)))
    terms = [Program("feature", name=name) for name in names]
    if not include_history:
        return terms
    for lag in range(1, min(4, max(len(row.history) for row in observations)) + 1):
        mature = [row for row in observations if len(row.history) >= lag]
        if len(mature) < 6:
            continue
        historical = sorted(set.intersection(*(set(row.history[-lag]) for row in mature)))
        for name in historical:
            terms.append(Program("if", children=(Program("available", lag=lag),
                                                 Program("lag", name=name, lag=lag), Program("const", value=0))))
    return terms


def select_atoms(candidates, spec, observations, budget, *, limit=4, baseline_program=None):
    """Screen fairly offered calls, then fit one small head per shortlisted call.

    Each head uses raw/history coordinates together with g and g squared. Fits
    and predictions are charged by row, as are materialized coordinate columns.
    The result schedules subsequent search; it does not adopt a new hypothesis.
    """
    start = budget.spent
    count = len(observations)
    report = {"observed_rows": count, "baseline_loss": None, "screened_atoms": [],
              "tested_atoms": [], "selected": [], "column_compute": 0,
              "screen_compute": 0, "fit_compute": 0,
              "evidence": "previously_acquired_trajectory_split", "status": "insufficient_evidence"}

    def finish(atoms=()):
        report["compute_cost"] = budget.spent - start
        return list(atoms), report

    if count < 8 or limit < 1:
        return finish()
    blocks = trajectory_blocks(observations)
    block_indices = []
    offset = 0
    for block in blocks:
        block_indices.append(list(range(offset, offset + len(block))))
        offset += len(block)
    if len(blocks) > 1:
        held = {index for block, indices in enumerate(block_indices)
                if block % 3 == (1 if len(blocks) == 2 else 2) for index in indices}
    else:
        held = set(range(max(1, count * 2 // 3), count))
    training = [index for index in range(count) if index not in held]
    validation = sorted(held)
    raw = _raw_coordinates(observations, include_history=baseline_program is None)
    if baseline_program is not None and baseline_program not in raw:
        raw.append(baseline_program)
    report["baseline_program"] = baseline_program.describe() if baseline_program is not None else None
    report["raw_coordinates"] = [term.describe() for term in raw]
    report["training_rows"], report["validation_rows"] = len(training), len(validation)
    report["validation_trajectories"] = sum(bool(held.intersection(indices)) for indices in block_indices)
    if len(training) < max(6, len(raw) + 3) or not validation:
        return finish()
    fitting_rows = len(training) * (2 if spec.risk_quantile != 0.5 else 1)
    baseline_cost = count * len(raw) + fitting_rows + count
    trial_cost = 3 * count + fitting_rows
    sample_size = min(12, count)
    screen_cost = 4 * sample_size
    if budget.remaining < baseline_cost + trial_cost + screen_cost:
        report["status"] = "insufficient_budget"
        return finish()
    budget.spend(count * len(raw))
    report["column_compute"] += count * len(raw)
    fallback = initial_prediction([observations[index] for index in training], spec.risk_quantile)
    columns = [[predict(term, row, fallback) for row in observations] for term in raw]
    targets = [row.target for row in observations]
    budget.spend(fitting_rows + count)
    report["fit_compute"] += fitting_rows + count
    baseline = _fit_predict(columns, targets, training, quantile=spec.risk_quantile)
    baseline_loss = mean(point_loss(baseline[index], targets[index], spec.risk_quantile) for index in validation)
    report["baseline_loss"] = baseline_loss
    if baseline_loss < 1e-8:
        report["status"] = "raw_coordinates_sufficient"
        return finish()
    sample = [round(index * (count - 1) / (sample_size - 1)) for index in range(sample_size)]
    residuals = [targets[index] - baseline[index] for index in sample]
    sample_baseline = mean(value * value for value in residuals)
    screen_allowance = min(max(screen_cost, budget.remaining // 2), budget.remaining - trial_cost)
    screened = []
    seen = set()
    for atom in candidates:
        if report["screen_compute"] + screen_cost > screen_allowance:
            break
        expanded = atom.expanded()
        if expanded in seen:
            continue
        seen.add(expanded)
        if any(len(row.history) < atom.minimum_history_depth or not atom.required_features().issubset(row.features)
               for row in observations):
            continue
        budget.spend(screen_cost)
        report["screen_compute"] += screen_cost
        values = [atom.predict(observations[index]) for index in sample]
        squares = [value * value for value in values]
        projected = _fit_predict([values, squares], residuals, list(range(sample_size)))
        gain = sample_baseline - mean((target - prediction) ** 2 for target, prediction in zip(residuals, projected))
        report["screened_atoms"].append({"definition_id": atom.name, "arguments": [child.describe() for child in atom.children],
                                         "residual_gain": gain})
        screened.append((gain, atom))
    screened.sort(key=lambda item: (-item[0], item[1].size, item[1].describe()))
    tested = []
    for _, atom in screened:
        if not budget.spend(trial_cost):
            break
        values = [atom.predict(row) for row in observations]
        squares = [value * value for value in values]
        report["column_compute"] += 2 * count
        report["fit_compute"] += fitting_rows + count
        predictions = _fit_predict([*columns, values, squares], targets, training, quantile=spec.risk_quantile)
        measured = mean(point_loss(predictions[index], targets[index], spec.risk_quantile) for index in validation)
        gain = baseline_loss - measured
        report["tested_atoms"].append({"definition_id": atom.name, "arguments": [child.describe() for child in atom.children],
                                       "validation_loss": measured, "gain": gain,
                                       "composition": "raw_plus_g_and_g_squared"})
        if gain > max(1e-8, 0.03 * baseline_loss):
            tested.append((measured, atom))
    tested.sort(key=lambda item: (item[0], item[1].size, item[1].describe()))
    selected = [atom for _, atom in tested[:limit]]
    report["selected"] = [{"definition_id": atom.name, "arguments": [child.describe() for child in atom.children]}
                          for atom in selected]
    report["status"] = "useful_calls_selected" if selected else "no_supported_gain"
    return finish(selected)
