"""Compose learned computations and retain the data-selected research procedure.

The operator vocabulary is finite. The current residual decides which reusable
computation to add and which temporal question to pursue; task names and private
apparatus parameters never enter this search. Fresh evidence is still needed to
select a proposal. A saved procedure can refit its chosen intermediate terms on
new observations without repeating the entire grammar search.
"""

from itertools import combinations, combinations_with_replacement
from copy import deepcopy
import json
from math import isfinite

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, loss
from .language import MissingHistory, Program
from .models import Candidate
from .research_graph import ResearchGraph
from .scale_bindings import bind_measurement_arguments
from .structure_lab import ObjectSchema


def _sources(program):
    sources = []
    if program.op == "call":
        source = program.definition.source_concept_id or program.definition.definition_id
        sources.append(source)
        sources.extend(_sources(program.definition.body))
    for child in program.children:
        sources.extend(_sources(child))
    return list(dict.fromkeys(sources))


def _temporal(program):
    yield Program("event_delta", lag=1, children=(program,))
    for width in (2, 3, 4):
        yield Program("window_sum", lag=width, children=(program,))
    for decay in (.2, .5, .8, .95):
        yield Program("accumulate", value=decay,
                      children=(program, Program("const", value=0.)))


class _Fitter:
    def __init__(self, rows, budget, quantile):
        self.rows, self.budget, self.quantile = rows, budget, quantile
        self.count = len(rows)
        self.targets = [row.target for row in rows]
        self.fallback = initial_prediction(rows, quantile)
        self.columns, self.fits = {}, {}

    def column(self, term):
        if term in self.columns:
            return self.columns[term]
        if not self.budget.spend(self.count):
            return None
        try:
            values = [term.predict(row) for row in self.rows]
        except (MissingHistory, KeyError):
            values = None
        if values is not None and not all(isfinite(value) for value in values):
            values = None
        self.columns[term] = values
        return values

    def fit(self, terms):
        terms = tuple(dict.fromkeys(terms))
        if not terms:
            return None
        if terms in self.fits:
            return self.fits[terms]
        missing = sum(term not in self.columns for term in terms)
        if (missing + 2) * self.count > self.budget.remaining:
            return None
        columns = [self.column(term) for term in terms]
        if any(column is None for column in columns):
            self.fits[terms] = None
            return None
        active = [(term, column) for term, column in zip(terms, columns)
                  if max(column) - min(column) > 1e-10]
        if not active:
            self.fits[terms] = None
            return None
        terms, columns = tuple(zip(*active))
        self.budget.spend(self.count)
        coefficients = _linear_fit(columns, self.targets)
        program = _sum_terms(terms, coefficients[1:], coefficients[0])
        self.budget.spend(self.count)
        measured = loss(program, self.rows, self.fallback, self.quantile)
        used = tuple(term for term, coefficient in zip(terms, coefficients[1:])
                     if abs(coefficient) >= 1e-8)
        result = {"terms": used, "program": program, "loss": measured,
                  "coefficients": coefficients, "fit_terms": terms}
        self.fits[terms] = result
        return result


def _rank(fit):
    # Once the recorded evidence is explained numerically, retain the smaller
    # explanation instead of choosing a much larger expression for roundoff.
    return (0. if fit["loss"] < 1e-12 else fit["loss"],
            fit["program"].size)


def _plan(fit, steps, count, compute_cost):
    graph = ResearchGraph()
    graph.evidence_count = count
    terms = fit["fit_terms"]
    for index, term in enumerate(terms):
        node = graph.add_question(f"term:{index}",
            "Evaluate a reusable computation in the current representation",
            operations=("reuse",), expression=term,
            required_measurements=sorted(term.required_features()))
        node.answer, node.status = term, "answered"
    previous = None
    for index, step in enumerate(steps):
        key = f"fit:{index}"
        node = graph.add_question(key, step["question"],
            operations=(step["operation"],),
            depends_on=([previous] if previous else []),
            expression=step["program"])
        node.answer, node.status, node.loss = step["program"], "answered", step["loss"]
        previous = key
    dependencies = [f"term:{index}" for index in range(len(terms))]
    expression = _sum_terms([Program("arg", name=key) for key in dependencies],
                            fit["coefficients"][1:], fit["coefficients"][0])
    output = graph.add_question("answer", "Refit and combine the selected computations",
        operations=("refit", "composition"),
        depends_on=[*dependencies, *([previous] if previous else [])], expression=expression)
    output.answer, output.status, output.loss = fit["program"], "answered", fit["loss"]
    output.compute_cost = compute_cost
    return {"kind": "residual_composition", "graph": graph.to_dict(),
            "term_nodes": dependencies, "output_node": "answer",
            "selection_steps": [{**step, "program": step["program"].to_dict()}
                                for step in steps],
            "fit_rows": count, "search_compute": compute_cost,
            "source_ids": _sources(fit["program"]),
            "status": "requires_fresh_validation"}


def _entry(fit, steps, count, compute_cost, index=0):
    plan = _plan(fit, steps, count, compute_cost)
    return {"candidate": Candidate(f"research_composition:{index}", fit["program"],
                "research_program", "Observed residual selects a composition of learned computations",
                tuple(plan["source_ids"])),
            "kind": "research_program", "family_id": "T03", "training_loss": fit["loss"],
            "research_program": plan}


def coordinate_search_allowance(feature_count, row_count, history_length=0):
    """Quote a bounded attempt using the constructor's two multiplication rounds."""
    first = min(feature_count, 8)
    first_columns = first + first * (first + 1) // 2
    second = min(first_columns, 8)
    second_columns = second + second * (second + 1) // 2
    # Single-column fits, four residual passes and four joint alternatives per
    # pass. Cached columns/duplicate products usually make execution cheaper.
    allowance = row_count * (3 * feature_count + 7 * (first_columns + second_columns) + 72)
    if history_length >= 2:
        # Joint temporal fits need fresh columns for each scale. They consume
        # the same research budget as coordinate composition, not a free pass.
        allowance += row_count * (min(second_columns, 24) + 3) * 16
    return allowance


def propose_coordinate_programs(spec, training, budget, *, max_candidates=8):
    """Build a sparse arithmetic explanation from measured coordinates alone.

    Two rounds of multiplication include self-composition. Residual evidence
    chooses the coordinates retained for the next round; no learned definition
    or task-specific polynomial is needed to open this research direction.
    """
    rows = list(training)
    if len(rows) < 6 or max_candidates < 1:
        return []
    names = set.intersection(*(set(row.features) for row in rows))
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    start = budget.spent
    raw = [fit for name in spec.feature_names if name in names
           if (fit := fitter.fit((Program("feature", name=name),))) is not None]
    if not raw:
        return []
    raw.sort(key=_rank)
    best = raw[0]
    tolerance = max(1e-10, abs(best["loss"]) * 1e-7)
    initial_loss = best["loss"]
    pool = list(dict.fromkeys(term for fit in raw for term in fit["terms"]))[:8]
    path = [{"operation": "fit_measured_coordinate",
        "question": "Explain the response with the currently observed coordinates",
        "program": best["program"], "loss": best["loss"]}]
    found, basis = [], list(pool)

    def product(left, right):
        # Multiplication is commutative for these scalar current coordinates.
        # Canonical factors avoid fitting duplicate columns at the next depth.
        def factors(term):
            return ([part for child in term.children for part in factors(child)]
                    if term.op == "mul" else [term])
        terms = sorted([*factors(left), *factors(right)], key=lambda term: term.name)
        result = terms[0]
        for term in terms[1:]:
            result = Program("mul", children=(result, term))
        return result

    for depth in range(2):
        candidates = list(dict.fromkeys([*pool,
            *(product(left, right) for left, right in combinations_with_replacement(pool, 2))]))
        basis = list(dict.fromkeys([*basis, *candidates]))
        singles = [fit for term in candidates if (fit := fitter.fit((term,))) is not None]
        if not singles:
            break
        singles.sort(key=_rank)
        if _rank(singles[0]) < _rank(best):
            best = singles[0]
        # Current-coordinate expressions are small enough to compare several
        # residual additions jointly; each correlation pass is budgeted.
        for _ in range(4):
            if best["loss"] <= tolerance:
                break
            eligible = [(term, fitter.columns[term]) for term in candidates
                        if term not in best["terms"] and fitter.columns.get(term) is not None]
            if not budget.spend(len(rows) * (len(eligible) + 1)):
                break
            residual = [row.target - best["program"].predict(row) for row in rows]
            residual_center = sum(residual) / len(rows)
            ranked = []
            for term, column in eligible:
                center = sum(column) / len(rows)
                norm = sum((value - center) ** 2 for value in column)
                if norm <= 1e-12:
                    continue
                covariance = sum((value - center) * (error - residual_center)
                                 for value, error in zip(column, residual))
                ranked.append((covariance * covariance / norm, term))
            alternatives = [fit for _, term in sorted(ranked, key=lambda item: item[0], reverse=True)[:4]
                            if (fit := fitter.fit((*best["terms"], term))) is not None]
            if not alternatives:
                break
            revised = min(alternatives, key=_rank)
            if revised["loss"] + tolerance >= best["loss"]:
                break
            best = revised
        path = [*path, {"operation": "compose_measured_coordinates",
            "question": "Multiply observed computations and jointly explain the remaining response",
            "program": best["program"], "loss": best["loss"], "composition_depth": depth + 1}]
        if best["loss"] + tolerance < initial_loss:
            found.append((best, path))
        if best["loss"] <= tolerance:
            break
        pool = list(dict.fromkeys([*best["terms"],
            *(term for fit in singles for term in fit["terms"]), *pool]))[:8]
    ranked = sorted(found, key=lambda item: _rank(item[0]))[:max_candidates]
    claims = [_entry(fit, steps, len(rows), budget.spent - start, index)
              for index, (fit, steps) in enumerate(ranked)]
    if best["loss"] > tolerance and max(len(row.history) for row in rows) >= 2:
        from .temporal_coordinates import propose_temporal_coordinates
        claims.extend(propose_temporal_coordinates(spec, rows, budget,
            basis=basis, max_candidates=max_candidates))
    return sorted(claims, key=lambda item: (0. if item["training_loss"] < 1e-12 else item["training_loss"],
                  item["candidate"].program.size))[:max_candidates]


def _role_coordinates(spec):
    """Public roles propose reductions; response evidence must select their use."""
    payload = spec.context.get("object_schema")
    if payload is None:
        return []
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not set(schema.feature_names) <= set(spec.feature_names):
        return []
    available = set.intersection(*(set(schema.fields[group]) for group in schema.objects))
    roles = sorted(available - {"f"}, key=lambda role: (role != "q", role))
    coordinates = []
    for role in roles:
        terms = [Program("feature", name=schema.fields[group][role]) for group in schema.objects]
        aggregate = Program("object_sum", children=tuple(terms))
        basis = [aggregate]
        if "f" in available:
            focused = Program("object_sum", children=tuple(
                Program("mul", children=(Program("feature", name=schema.fields[group]["f"]), term))
                for group, term in zip(schema.objects, terms)))
            basis.append(focused)
        coordinates.append((role, basis))
    return coordinates


def _retains_atom(program, atoms):
    """A joint revision must still use one of the computations it received."""
    return program in atoms or any(_retains_atom(child, atoms) for child in program.children)


def _joint_role_revisions(spec, fitter, raw, tolerance):
    """Retain callable mechanisms while jointly explaining a changed local law.

    A correction can require both a group reduction and a change of focal state.
    Fitting one residual term at a time can miss that joint explanation. The
    public-role blocks below are small competing hypotheses, and every fit uses
    all supplied target observations without changing the retained definitions.
    """
    coordinates = _role_coordinates(spec)
    if not coordinates:
        return []
    anchors = list(dict.fromkeys(term for fit in raw[:4] for term in fit["terms"]))
    parent = fitter.fit(anchors)
    if parent is None:
        return []
    # The raw callable fits are the first paid step; the joint refit is a
    # separate step even when it reaches zero residual immediately. Keeping
    # both steps makes the saved procedure describe the actual distinction
    # between fitting components and confirming their combination.
    first = {"operation": "fit_component",
             "question": "Fit the previously learned components before combining them",
             "program": raw[0]["program"], "loss": raw[0]["loss"]}
    joint = {"operation": "joint_callable_refit",
             "question": "Jointly refit the contributions of previously learned mechanisms",
             "program": parent["program"], "loss": parent["loss"],
             "parent_loss": raw[0]["loss"], "joint_terms": len(parent["terms"])}
    proposals = [(parent, [first, joint])]
    if parent["loss"] <= tolerance:
        return proposals
    temporal = max(len(row.history) for row in fitter.rows) >= 2
    for role, current in coordinates:
        changes = [Program("event_delta", lag=1, children=(term,)) for term in current] if temporal else []
        blocks = [(anchors, current, "current_reduction"),
                  (anchors, [*current, *changes], "reduction_and_event")]
        if temporal:
            for width in (2, 3, 4):
                windowed = [Program("window_sum", lag=width, children=(term,))
                            for term in [*current, *changes]]
                blocks.append((anchors, windowed, f"window_{width}_residual"))
                blocks.append(([Program("window_sum", lag=width, children=(term,)) for term in anchors],
                               windowed, f"window_{width}_joint_mechanism"))
        for retained, residual_terms, operation in blocks:
            fit = fitter.fit([*retained, *residual_terms])
            if fit is None or fit["loss"] + tolerance >= parent["loss"]:
                continue
            # A role-only model is not evidence that a previous mechanism was
            # locally repaired. The ordinary structural proposer handles those.
            if not _retains_atom(fit["program"], anchors):
                continue
            step = {"operation": "joint_role_residual",
                "question": "Explain the residual through measured group and focal roles",
                "program": fit["program"], "loss": fit["loss"],
                "parent_loss": parent["loss"], "role": role, "coordinate_operation": operation,
                "residual_rows": fitter.count, "joint_terms": len(fit["terms"])}
            proposals.append((fit, [first, step]))
            if fit["loss"] <= tolerance:
                return proposals
    return proposals


def _joint_temporal_roles(spec, fitter, raw, tolerance):
    """Separate measured state contributions from a retained temporal mechanism.

    Transforming an entire scalar call alone cannot isolate the change of its
    interaction component. Jointly transforming measured roles lets the evidence
    remove those contributions, while keeping the learned call and its internal
    relations unchanged. Distinct sources compete before repeated role bindings
    of one strongly correlated source can occupy all component slots.
    """
    coordinates = _role_coordinates(spec)
    if not coordinates or max(len(row.history) for row in fitter.rows) < 2:
        return []
    anchors, sources = [], set()
    for fitted in raw:
        for term in fitted["terms"]:
            source = tuple(_sources(term)) or term
            if source not in sources:
                anchors.append(term)
                sources.add(source)
        if len(anchors) >= 4:
            break
    parent = fitter.fit(anchors)
    if parent is None or parent["loss"] <= tolerance:
        return []
    role_terms = [term for _, terms in coordinates for term in terms]
    changes = [Program("event_delta", lag=1, children=(term,))
               for term in [*anchors, *role_terms]]
    alternatives = [(changes, "event"), ([*anchors, *changes], "current_and_event")]
    alternatives.extend(([Program("window_sum", lag=width, children=(term,))
                          for term in [*anchors, *changes]], f"window_{width}_current_and_event")
                        for width in (2, 3, 4))
    first = {"operation": "joint_callable_refit",
        "question": "Compare contributions from distinct previously learned mechanisms",
        "program": parent["program"], "loss": parent["loss"]}
    proposals = []
    for terms, operation in alternatives:
        fitted = fitter.fit(terms)
        if (fitted is None or fitted["loss"] + tolerance >= parent["loss"]
                or not _retains_atom(fitted["program"], anchors)):
            continue
        step = {"operation": "derive_temporal_mechanism",
            "question": "Jointly separate measured state contributions from the change of a learned mechanism",
            "program": fitted["program"], "loss": fitted["loss"],
            "parent_loss": parent["loss"], "roles": [role for role, _ in coordinates],
            "coordinate_operation": operation, "residual_rows": fitter.count,
            "joint_terms": len(fitted["terms"])}
        proposals.append((fitted, [first, step]))
        if fitted["loss"] <= tolerance:
            break
    return proposals


def _object_observation_lifts(spec, fitter, raw, tolerance):
    """Reuse a callable observation at each object, then compare its reduction.

    These are interventions on an executable function's membership arguments,
    not extra observations of the laboratory. Actual response rows must select
    the resulting hypothesis. The bounded search starts with instantaneous
    observations; temporal operators are applied after the object reduction.
    """
    coordinates = _role_coordinates(spec)
    if not coordinates:
        return []
    payload = spec.context["object_schema"]
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not all("f" in schema.fields[group] for group in schema.objects):
        return []
    membership = {schema.fields[group]["f"] for group in schema.objects}
    choices, source_counts = [], {}
    for fitted in raw:
        for call in fitted["terms"]:
            source = tuple(_sources(call))
            if (call.op != "call" or call.history_depth or not membership & call.required_features()
                    or source_counts.get(source, 0) >= 2):
                continue
            source_counts[source] = source_counts.get(source, 0) + 1
            choices.append((call, fitted))
        if len(choices) >= 8:
            break
    proposals = []
    temporal = max(len(row.history) for row in fitter.rows) >= 2
    role_terms = [term for _, terms in coordinates for term in terms]
    for call, parent in choices:
        instances = tuple(bind_measurement_arguments(call, {
            schema.fields[group]["f"]: Program("const", value=float(group == selected))
            for group in schema.objects}) for selected in schema.objects)
        lifted = Program("object_sum", children=instances)
        lifted_fit = fitter.fit([lifted])
        if lifted_fit is None:
            continue
        first = {"operation": "lift_object_observation",
            "question": "Apply the learned observation to each measured object and reduce its answers",
            "program": lifted_fit["program"], "loss": lifted_fit["loss"], "source_ids": _sources(call),
            "objects": list(schema.objects), "membership_fields": sorted(membership),
            "instances": len(instances)}
        alternatives = [(lifted_fit, lifted, "current")]
        if temporal and lifted_fit["loss"] > tolerance:
            changes = [Program("event_delta", lag=1, children=(term,)) for term in [call, *role_terms]]
            alternatives.append((fitter.fit([lifted, *changes]), lifted, "current_and_event"))
            for width in (2, 3, 4):
                lifted_window = Program("window_sum", lag=width, children=(lifted,))
                terms = [lifted_window, *(Program("window_sum", lag=width, children=(term,))
                                         for term in changes)]
                alternatives.append((fitter.fit(terms), lifted_window, f"window_{width}_current_and_event"))
        for fitted, required, operation in alternatives:
            if (fitted is None or required not in fitted["terms"]
                    or fitted["loss"] + tolerance >= raw[0]["loss"]):
                continue
            step = {"operation": "compose_lifted_observation",
                "question": "Fit the reduced observation jointly with measured temporal contributions",
                "program": fitted["program"], "loss": fitted["loss"],
                "parent_loss": parent["loss"], "coordinate_operation": operation,
                "residual_rows": fitter.count, "joint_terms": len(fitted["terms"])}
            proposals.append((fitted, [first, step]))
        if any(fitted["loss"] <= tolerance for fitted, _ in proposals):
            break
    return proposals


def propose_research_programs(spec, training, budget, calls, *, max_candidates=16):
    """Search a bounded residual-driven composition grammar over callable atoms.

    Each column, fit, residual measurement and executable comparison costs one
    pass through the supplied rows. Public object roles can supply a joint local
    mechanism correction; two generic rounds also permit temporal operations on
    an already learned composition.
    Candidate parents remain the actual source concepts, including nested calls.
    """
    rows = list(training)
    if len(rows) < 6 or max_candidates < 1:
        return []
    names = set.intersection(*(set(row.features) for row in rows))
    atoms = list(dict.fromkeys(call for call in calls
                 if call.required_features().issubset(names)))[:max_candidates]
    if not atoms:
        return []
    start = budget.spent
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    raw = [fit for atom in atoms if (fit := fitter.fit((atom,))) is not None]
    if not raw:
        return []
    raw.sort(key=_rank)
    baseline = raw[0]
    tolerance = max(1e-10, abs(baseline["loss"]) * 1e-7)
    if baseline["loss"] <= tolerance:
        return []
    baseline_step = {"operation": "fit_component", "question": "Fit a previously learned component",
                     "program": baseline["program"], "loss": baseline["loss"]}
    paths = {fit["program"]: [{"operation": "fit_component",
             "question": "Fit a previously learned component", "program": fit["program"],
             "loss": fit["loss"]}] for fit in raw}
    found = {}
    beam = raw[:3]
    pool = list(dict.fromkeys(term for fit in raw[:8] for term in fit["terms"]))
    temporal = max(len(row.history) for row in rows) >= 2

    def retain(fit, path):
        if fit is None:
            return
        paths.setdefault(fit["program"], path)
        if fit["loss"] + tolerance < baseline["loss"]:
            old = found.get(fit["program"])
            if old is None or len(path) < len(old[1]):
                found[fit["program"]] = fit, path

    for fit, path in _object_observation_lifts(spec, fitter, raw, tolerance):
        retain(fit, path)
    if not found or min(fit["loss"] for fit, _ in found.values()) > tolerance:
        for fit, path in _joint_role_revisions(spec, fitter, raw, tolerance):
            retain(fit, path)
        if not found or min(fit["loss"] for fit, _ in found.values()) > tolerance:
            for fit, path in _joint_temporal_roles(spec, fitter, raw, tolerance):
                retain(fit, path)
    if found and min(fit["loss"] for fit, _ in found.values()) <= tolerance:
        ranked = sorted(found.values(), key=lambda item: _rank(item[0]))[:max_candidates]
        return [_entry(fit, path, len(rows), budget.spent - start, index)
                for index, (fit, path) in enumerate(ranked)]

    for depth in range(2):
        # Residual fits nominate additional coordinates. The complete joint fit
        # below then decides whether the new coordinate survives recombination.
        candidates = list(pool)
        if temporal:
            seeds = pool if depth == 0 else [fit["program"] for fit in beam[:2]]
            candidates.extend(term for seed in seeds for term in _temporal(seed))
        candidates.extend(Program("mul", children=(left, right))
                          for left, right in combinations(pool[:5], 2))
        candidates = list(dict.fromkeys(candidates))
        singles = []
        for term in candidates:
            fit = fitter.fit((term,))
            if fit is None:
                continue
            singles.append(fit)
            path = [baseline_step, {"operation": "transform_component",
                "question": "Test whether a temporal or interacting computation explains the residual",
                "program": fit["program"], "loss": fit["loss"]}]
            retain(fit, path)
        singles.sort(key=_rank)
        # Include original atoms even when the whole target correlates more
        # strongly with a transformed coordinate than with its small residual.
        options = list(dict.fromkeys([*pool,
            *(term for fit in singles[:12] for term in fit["terms"])]))
        beam = sorted({fit["program"]: fit for fit in [*beam, *singles[:3]]}.values(), key=_rank)[:3]
        for _ in range(2):
            expanded = list(beam)
            for parent in beam:
                if len(parent["terms"]) >= 3 or parent["loss"] <= tolerance:
                    continue
                if not budget.spend(len(rows)):
                    break
                residual = [row.target - parent["program"].predict(row) for row in rows]
                # All correlations use cached, paid columns. These are proposals;
                # only an actual joint fit can supply an accepted training answer.
                residual_center = sum(residual) / len(rows)
                eligible = [(term, fitter.columns[term]) for term in options
                            if term not in parent["terms"] and fitter.columns.get(term) is not None]
                # Fund these additional passes before computing correlations.
                if not budget.spend(len(rows) * len(eligible)):
                    break
                ranked = []
                for term, column in eligible:
                    center = sum(column) / len(rows)
                    norm = sum((value - center) ** 2 for value in column)
                    if norm <= 1e-12:
                        continue
                    covariance = sum((value - center) * (error - residual_center)
                                     for value, error in zip(column, residual))
                    ranked.append((covariance * covariance / norm, term))
                for _, term in sorted(ranked, key=lambda item: item[0], reverse=True)[:8]:
                    fit = fitter.fit((*parent["terms"], term))
                    if fit is None or fit["loss"] + tolerance >= parent["loss"]:
                        continue
                    step = {"operation": "explain_residual_and_recombine",
                            "question": "Fit a remaining component and jointly revise its contribution",
                            "program": fit["program"], "loss": fit["loss"],
                            "parent_loss": parent["loss"], "residual_rows": len(rows)}
                    path = [*paths.get(parent["program"], [baseline_step]), step]
                    retain(fit, path)
                    expanded.append(fit)
            beam = sorted({fit["program"]: fit for fit in expanded}.values(), key=_rank)[:3]
        if beam and beam[0]["loss"] <= tolerance:
            break
        pool = list(dict.fromkeys([*pool,
            *(term for fit in beam for term in fit["terms"])]))[:12]
    ranked = sorted(found.values(), key=lambda item: _rank(item[0]))[:max_candidates]
    return [_entry(fit, path, len(rows), budget.spent - start, index)
            for index, (fit, path) in enumerate(ranked)]


def replay_research_program(payload, training, budget, *, risk_quantile=.5,
                            feature_mapping=None):
    """Execute a saved operator plan and refit its answer on newly supplied rows.

    This reuses the research procedure's selected coordinates, not its old
    coefficients or validation verdict. A resumed proposal needs fresh external
    validation just like a newly generated one.
    """
    rows = list(training)
    if len(rows) < 6:
        return None
    if any(step["operation"] == "discover_decoded_conditional_coordinates"
           for step in payload.get("selection_steps", [])):
        from .decoded_conditional_coordinates import replay_decoded_conditional_coordinates
        return replay_decoded_conditional_coordinates(payload, rows, budget,
            risk_quantile=risk_quantile, feature_mapping=feature_mapping)
    if any(step["operation"] == "discover_composed_conditions"
           for step in payload.get("selection_steps", [])):
        from .composed_conditions import replay_composed_conditions
        return replay_composed_conditions(payload, rows, budget,
            risk_quantile=risk_quantile, feature_mapping=feature_mapping)
    if any(step["operation"] == "discover_conditional_coordinates"
           for step in payload.get("selection_steps", [])):
        from .conditional_coordinates import replay_conditional_coordinates
        return replay_conditional_coordinates(payload, rows, budget,
            risk_quantile=risk_quantile, feature_mapping=feature_mapping)
    graph = ResearchGraph.from_dict(payload["graph"])
    terms = [graph.nodes[key].answer for key in payload["term_nodes"]]
    if not terms or any(term is None for term in terms):
        return None
    saved = next((step for step in reversed(payload.get("selection_steps", []))
                  if step["operation"] == "resolve_relation_intervals"), None)
    parameter_terms = [Program.from_dict(term) for term in saved.get("parameter_terms", [])] if saved else []
    if (saved and not parameter_terms and len(terms) == 1 and
            terms[0] == Program.from_dict(saved["program"])):
        parameter_terms = [Program.from_dict(saved["parameter_template"])]
    if feature_mapping:
        terms = [bind_measurement_arguments(term, feature_mapping) for term in terms]
        parameter_terms = [bind_measurement_arguments(term, feature_mapping) for term in parameter_terms]
    start = budget.spent
    fitter = _Fitter(rows, budget, risk_quantile)
    fit = fitter.fit(terms)
    if fit is None:
        return None
    if saved and len(parameter_terms) == len(terms):
        from .relation_uncertainty import _parameter_relations
        unresolved = any(term not in fit["fit_terms"] and any(
            _parameter_relations(template, item["symbol"]) for item in saved["parameters"])
            for term, template in zip(terms, parameter_terms))
        if unresolved:
            # A correction can be zero on this pilot yet matter at its boundary.
            # Retest the saved complete response before removing that dependency.
            retained = Program.from_dict(saved["program"])
            template = Program.from_dict(saved["parameter_template"])
            if feature_mapping:
                retained = bind_measurement_arguments(retained, feature_mapping)
                template = bind_measurement_arguments(template, feature_mapping)
            complete_fit = fitter.fit([retained])
            if complete_fit is not None and _rank(complete_fit)[0] <= _rank(fit)[0]:
                fit, terms, parameter_terms = complete_fit, [retained], [template]
    step = {"operation": "refit_research_program",
            "question": "Retest the saved computation dependencies on new evidence",
            "program": fit["program"], "loss": fit["loss"]}
    steps = [step]
    if saved and len(parameter_terms) == len(terms):
        from .relation_uncertainty import _bind, _parameter_relations
        # Fitting removes constant columns and merges identical executable
        # terms. Follow the terms actually retained by that solve.
        mapped = {}
        for term, template in zip(terms, parameter_terms):
            mapped.setdefault(term, set()).add(template)
        if all(len(mapped[term]) == 1 for term in fit["fit_terms"]):
            actual_templates = [next(iter(mapped[term])) for term in fit["fit_terms"]]
            template = _sum_terms(actual_templates, fit["coefficients"][1:], fit["coefficients"][0])
            parameters = [item for item in saved["parameters"]
                          if _parameter_relations(template, item["symbol"])]
            if parameters:
                active = {item["symbol"] for item in parameters}
                fixed = {item["symbol"]: Program("const", value=item["radius"])
                         for item in saved["parameters"] if item["symbol"] not in active}
                contexts = []
                for context in saved.get("evidence_contexts", []):
                    if not active.intersection(context["parameters"]):
                        continue
                    context = deepcopy(context)
                    context["parameter_template"] = _bind(
                        Program.from_dict(context["parameter_template"]), fixed).to_dict()
                    context["parameters"] = {key: value for key, value in context["parameters"].items() if key in active}
                    contexts.append(context)
                steps.append({**deepcopy(saved), "program": fit["program"], "loss": fit["loss"],
                    "question": "Retest the surviving relation dependencies in the newly fitted complete response",
                    "parameter_template": template.to_dict(), "parameters": deepcopy(parameters),
                    "parameter_terms": [term.to_dict() for term in actual_templates],
                    "evidence_contexts": contexts})
    result = _entry(fit, steps, len(rows), budget.spent - start)
    result["research_program"]["reused_procedure"] = True
    return result
