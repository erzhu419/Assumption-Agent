"""Compare reset-aware window decoding with ordinary measured response fits.

The inverse window is a hypothesis about purchased target histories. Its fitted
instantaneous response is wrapped back into a forward feature-only program;
every returned loss is measured against the original process response.
"""

from copy import deepcopy
from dataclasses import replace
from math import ceil, log2

from .conditional_coordinates import _coordinates, propose_conditional_coordinates
from .evaluation import initial_prediction, loss
from .language import Program
from .models import Observation, SearchBudget
from .research_programs import _entry, _Fitter, _rank


def decode_measured_window_rows(rows, width, budget):
    """Invert each row's actually observed prefix, preserving its physical reset.

    A prefix is part of its purchased observation, not the preceding item in a
    concatenated bank. Separately acquired trajectories therefore never borrow
    each other's responses. Without the paid past targets this method supplies
    no candidate; ordinary forward response fitting remains available.
    """
    rows = list(rows)
    if width < 1 or any("__target__" not in point for row in rows for point in row.history):
        return None
    # The reset prefix belongs to the observation. Reuse its already inverted
    # exact prefixes instead of recomputing every earlier time for every row.
    # Identical prefixes have the same inverse regardless of acquisition order.
    prefixes = {(): ()}
    decoded = []
    for row in rows:
        targets = tuple([*(point["__target__"] for point in row.history), row.target])
        end = len(targets)
        while targets[:end] not in prefixes:
            end -= 1
        response = list(prefixes[targets[:end]])
        for index in range(end, len(targets)):
            if not budget.spend(4):
                return None
            previous = targets[index - 1] if index else 0.
            response.append(targets[index] - previous
                + (response[index - width] if index >= width else 0.))
            prefixes[targets[:index + 1]] = tuple(response)
        decoded.append(Observation(dict(row.features), response[-1]))
    return decoded


def _window(program, width):
    return Program("window_sum", lag=width, children=(program,)) if width > 1 else program


def _forward_fit(inner, width, rows, budget, quantile):
    """Retain the whole inner offset and its actual windowed graph coordinates."""
    if not budget.spend(inner.size + len(rows)):
        return None
    coefficients, intercept = {}, 0.

    def collect(program, weight):
        nonlocal intercept
        if program.op == "add":
            for child in program.children:
                collect(child, weight)
        elif program.op == "sub":
            collect(program.children[0], weight)
            collect(program.children[1], -weight)
        elif program.op == "mul" and any(child.op == "const" for child in program.children):
            index = 0 if program.children[0].op == "const" else 1
            collect(program.children[1 - index], weight * program.children[index].value)
        elif program.op == "const":
            intercept += weight * program.value
        else:
            coefficients[program] = coefficients.get(program, 0.) + weight

    collect(inner, 1.)
    atoms = [term for term, weight in coefficients.items() if weight != 0.]
    weights = [coefficients[term] for term in atoms]
    if intercept != 0.:
        atoms.append(Program("const", value=1.))
        weights.append(intercept)
    terms = tuple(_window(term, width) for term in atoms)
    program = _window(inner, width)
    return {"program": program,
        "loss": loss(program, rows, initial_prediction(rows, quantile), quantile),
        "terms": terms, "fit_terms": terms, "coefficients": [0., *weights]}


def propose_decoded_conditional_coordinates(spec, rows, budget, *, basis, gate_names,
        width_candidates, max_candidates=8, duration_evidence=()):
    """Offer decoded-inner and smooth hypotheses on the same original targets.

    The caller nominates scales using already paid duration or ordinary pilot
    evidence. A bounded forward pilot ranks those scales on the original rows,
    then equal fitting allowances keep an incorrect first scale from consuming
    the entire conditional attempt. Window-exterior hypotheses remain ordinary
    forward candidates supplied by the caller's existing conditional search.
    """
    rows = list(rows)
    if len(rows) < 8 or max_candidates < 1:
        return []
    count, start = len(rows), budget.spent
    upper = max(len(row.history) for row in rows)
    widths = list(dict.fromkeys(int(width) for width in width_candidates
                               if 1 <= int(width) <= max(1, upper)))
    if not widths:
        return []
    available, measured_basis = _coordinates(spec, rows, basis)
    gates = [term.name for term in available if term.op == "feature" and term.name in gate_names]
    pilot_basis = [*measured_basis, Program("const", value=1.)]
    pilot_budget = SearchBudget(budget.remaining // 3)
    fitter = _Fitter(rows, pilot_budget, spec.risk_quantile)
    pilots, duration_trials = [], []
    for width in widths:
        before = pilot_budget.spent
        fit = fitter.fit([_window(term, width) for term in pilot_basis])
        if fit is None:
            continue
        pilots.append((width, fit))
        duration_trials.append({"width": width, "loss": fit["loss"],
            "compute_cost": pilot_budget.spent - before})
    budget.spend(pilot_budget.spent)
    if not pilots:
        return []
    pilots.sort(key=lambda item: _rank(item[1]))
    options = []
    for width, fit in pilots:
        step = {"operation": "discover_measured_window_response",
            "question": "Compare a smooth common response at a paid candidate duration",
            "program": fit["program"], "loss": fit["loss"], "width": width,
            "history_bound": upper, "duration_trials": duration_trials,
            "duration_evidence": deepcopy(duration_evidence)}
        entry = _entry(fit, [step], count, budget.spent - start, len(options))
        entry["candidate"] = replace(entry["candidate"],
            candidate_id=f"decoded_window_plain:{width}",
            rationale="A measured smooth response competes with conditional window decoding")
        options.append(entry)
    # Once a smooth response already explains the bought bank, the caller can
    # confirm it before purchasing or fitting an unnecessary condition.
    if not gates or pilots[0][1]["loss"] <= 1e-12:
        return sorted(options, key=lambda entry: (entry["training_loss"],
            entry["candidate"].program.size))[:max_candidates]

    decode_price = 4 * count
    # A condition needs its coarse screen AND measured-bracket refinement.
    # Dividing the money at the coarse-screen price gives many wrong scales
    # partial fits while preventing the correct one from locating its boundary.
    # Quote the existing search's two adjacent/midpoint revisions per level.
    refinements = 2 * ceil(log2(max(2, count)))
    static_price = count * (len(pilot_basis) + 2
        + (3 + refinements) * (2 * len(pilot_basis) + 2))
    minimum = decode_price + static_price + count * (max_candidates + 2)
    scales = min(len(pilots), max(1, budget.remaining // max(1, minimum)))
    allowance = budget.remaining // scales
    parameter_trials = []
    for width, _ in pilots[:scales]:
        local = SearchBudget(min(allowance, budget.remaining))
        decoded = decode_measured_window_rows(rows, width, local)
        if decoded is None:
            budget.spend(local.spent)
            continue
        graph_reserve = count * (max_candidates + 2) + 8 * len(pilot_basis) * max_candidates
        inner_budget = SearchBudget(max(0, local.remaining - graph_reserve))
        claims = propose_conditional_coordinates(spec, decoded, inner_budget,
            basis=measured_basis, gate_names=gates, max_candidates=max_candidates)
        local.spend(inner_budget.spent)
        for claim in claims:
            inner = claim["candidate"].program
            forward = _forward_fit(inner, width, rows, local, spec.risk_quantile)
            if forward is None:
                break
            record = {"width": width, "decoded_loss": claim["training_loss"],
                "original_loss": forward["loss"], "terms": len(forward["fit_terms"])}
            parameter_trials.append(record)
            step = {"operation": "discover_decoded_conditional_coordinates",
                "question": "Decode a paid reset prefix, then test its windowed conditional response on original targets",
                "program": forward["program"], "loss": forward["loss"], "width": width,
                "placement": "inside_window", "gate_names": gates,
                "history_bound": upper, "fit_rows": count,
                "decoded_loss": claim["training_loss"],
                "instantaneous_program": inner.to_dict(),
                "branch_basis": [term.to_dict() for term in measured_basis],
                "inner_selection_steps": deepcopy(claim["research_program"]["selection_steps"]),
                "width_candidates": widths, "duration_trials": duration_trials,
                "duration_evidence": deepcopy(duration_evidence),
                "parameter_trials": parameter_trials}
            entry = _entry(forward, [step], count, budget.spent - start + local.spent, len(options))
            entry["candidate"] = replace(entry["candidate"],
                candidate_id=f"decoded_conditional:{width}:{len(options)}",
                rationale="Paid reset-aware decoding proposes a forward condition at a data-supported duration")
            options.append(entry)
        budget.spend(local.spent)
    # Distinct scales and response complexities stay available for genuinely
    # new validation. Decoded fitting loss never selects the returned winner.
    unique = {}
    for entry in options:
        program = entry["candidate"].program
        if program not in unique or entry["training_loss"] < unique[program]["training_loss"]:
            unique[program] = entry
    ranked = sorted(unique.values(), key=lambda entry: (
        0. if entry["training_loss"] <= 1e-12 else entry["training_loss"],
        entry["candidate"].program.size))[:max_candidates]
    cost = budget.spent - start
    for entry in ranked:
        entry["research_program"]["search_compute"] = cost
        entry["research_program"]["graph"]["nodes"][entry["research_program"]["output_node"]]["compute_cost"] = cost
    return ranked


def replay_decoded_conditional_coordinates(payload, rows, budget, *, risk_quantile=.5,
                                          feature_mapping=None):
    """Rebind the saved response coordinates, then rediscover its condition and scale."""
    from .models import TaskSpec
    from .scale_bindings import bind_measurement_arguments

    rows = list(rows)
    if len(rows) < 8:
        return None
    saved = next((step for step in reversed(payload.get("selection_steps", []))
        if step["operation"] == "discover_decoded_conditional_coordinates"), None)
    if saved is None:
        return None
    basis = [Program.from_dict(term) for term in saved["branch_basis"]]
    gate_terms = [Program("feature", name=name) for name in saved["gate_names"]]
    if feature_mapping:
        basis = [bind_measurement_arguments(term, feature_mapping) for term in basis]
        gate_terms = [bind_measurement_arguments(term, feature_mapping) for term in gate_terms]
    names = tuple(sorted(set.intersection(*(set(row.features) for row in rows)) - {"__target__"}))
    gates = set().union(*(term.required_features() for term in gate_terms)) & set(names)
    spec = TaskSpec("decoded_condition_replay", "Retest a measured window response",
        names, (), (), risk_quantile=risk_quantile)
    upper = max(1, max(len(row.history) for row in rows))
    coarse, width = [1], 2
    while width < upper:
        coarse.append(width)
        width *= 2
    widths = list(dict.fromkeys([saved["width"], *saved.get("width_candidates", ()),
        *coarse, upper, *range(1, upper + 1)]))
    start = budget.spent
    claims = propose_decoded_conditional_coordinates(spec, rows, budget,
        basis=basis, gate_names=gates, width_candidates=widths, max_candidates=8,
        duration_evidence=({"operation": "replay_measured_window_method",
            "previous_width": saved["width"], "previous_placement": "inside_window"},))
    if not claims:
        return None
    selected = min(claims, key=lambda entry: (
        0. if entry["training_loss"] <= 1e-12 else entry["training_loss"],
        entry["candidate"].program.size))
    selected["research_program"]["reused_procedure"] = True
    selected["research_program"]["search_compute"] = budget.spent - start
    output = selected["research_program"]["output_node"]
    selected["research_program"]["graph"]["nodes"][output]["compute_cost"] = budget.spent - start
    return selected
