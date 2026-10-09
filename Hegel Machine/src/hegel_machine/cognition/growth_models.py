"""Bounded residual discovery, empirical role binding, and executable scopes.

All selection rows are ordinary acquired research data. The caller keeps its
independent terminal evaluation separate. Charges count candidate-row passes,
including fitted columns and comparisons; they are not wall-clock operations.
"""

from dataclasses import replace
from itertools import combinations, islice, permutations
from statistics import mean

from .language import Definition, Program
from .synthesis import _fit_linear


def _mse(program, rows):
    return mean((program.predict(row) - row.target) ** 2 for row in rows)


def _nodes(program):
    yield program
    for child in program.children:
        yield from _nodes(child)


def bind_prior(program, feature_names, rows, budget):
    """Compare at most 96 injective role bindings on paid current evidence."""
    body = program.expanded()
    roles = sorted({name for name, _ in body.observed_variables() if name != "__target__"})
    names = sorted(feature_names)
    if len(roles) > len(names):
        return []
    mappings = permutations(names, len(roles))
    if set(roles) <= set(names):
        mappings = iter([tuple(roles), *(p for p in islice(mappings, 96) if p != tuple(roles))])
    results = []
    for values in islice(mappings, 96):
        if not budget.spend(len(rows)):
            break
        mapping = dict(zip(roles, values))
        rebound = program.remap_features(mapping)
        # Lexical calls rebind their arguments without expanding away learned
        # definitions. A body containing free feature names needs expansion.
        required = {name for name, _ in rebound.observed_variables() if name != "__target__"}
        if not required <= set(names):
            rebound = body.remap_features(mapping)
        results.append({"program": rebound, "mapping": mapping, "loss": _mse(rebound, rows)})
    return sorted(results, key=lambda item: (item["loss"], tuple(sorted(item["mapping"].items()))))


def abstract_program(program, identifier):
    """Turn observed feature leaves into lexical arguments of a callable body."""
    body = program.expanded()
    features = sorted({node for node, lag in body.observed_variables() if lag == 0})
    arguments = tuple(f"role_{index}" for index in range(len(features)))
    mapping = dict(zip(features, arguments))

    def abstract(node):
        if node.op == "feature":
            return Program("arg", name=mapping[node.name])
        return replace(node, children=tuple(abstract(child) for child in node.children))

    return Definition(identifier, arguments, abstract(body), source_concept_id=identifier)


def discover(rows, validation, budget, *, priors=(), tolerance=1e-6, max_depth=3):
    """Select a paid predictor, growing products and piecewise residual terms.

    Old programs compete directly before new fits. A scope candidate retains an
    old program on a region where it predicts training data, learns a threshold
    and a replacement on the other side, and competes on current validation.
    """
    rows, validation = list(rows), list(validation)
    if not rows or not validation:
        raise ValueError("discovery needs acquired fitting and validation observations")
    start, events, candidates = budget.spent, [], []
    names = sorted(set.intersection(*(set(row.features) for row in [*rows, *validation])))
    features = [Program("feature", name=name) for name in names]
    fallback = {"program": Program("const", value=0.), "loss": float("inf"),
        "training_loss": float("inf"), "terms": []}

    def rank(candidate):
        return (0. if candidate["loss"] <= tolerance else candidate["loss"], candidate["program"].size,
            candidate["loss"] + candidate["training_loss"])

    def record(program, terms, kind, training=rows, extra=None):
        charge = (len(terms) + 2) * len(training) + len(validation)
        if program is None:
            if not budget.spend(charge):
                return None
            program = _fit_linear(terms, training).simplified()
        else:
            charge = len(training) + len(validation)
            if not budget.spend(charge):
                return None
        candidate = {"program": program, "terms": list(terms), "operation": kind,
            "training_loss": _mse(program, training), "loss": _mse(program, validation)}
        candidates.append(candidate)
        events.append({"action": kind, "compute_cost": charge, "training_loss": candidate["training_loss"],
            "validation_loss": candidate["loss"], **(extra or {})})
        return candidate

    def finish(best):
        ranked = sorted(candidates, key=rank)
        executable_nodes = set(_nodes(best["program"]))
        nodes = set(_nodes(best["program"].expanded()))
        boundaries = []
        for node in nodes:
            name, threshold = None, None
            if node.op == "if":
                predicate = node.children[0]
                if predicate.op == "lt" and predicate.children[0].op == "feature" and predicate.children[1].op == "const":
                    name, threshold = predicate.children[0].name, predicate.children[1].value
            elif node.op == "max" and node.children[0] == Program("const", value=0.):
                shifted = node.children[1]
                if shifted.op == "feature":
                    name, threshold = shifted.name, 0.
                elif shifted.op == "sub" and shifted.children[0].op == "feature" and shifted.children[1].op == "const":
                    name, threshold = shifted.children[0].name, shifted.children[1].value
            if name is None:
                continue
            values = sorted({row.features[name] for row in rows})
            below, above = [v for v in values if v < threshold], [v for v in values if v >= threshold]
            boundaries.append({"feature": name, "threshold": threshold,
                "lower": max(below) if below else None, "upper": min(above) if above else None})
        boundaries.sort(key=lambda item: (item["feature"], item["threshold"]))
        return {"program": best["program"], "loss": best["loss"], "events": events,
            "alternatives": list(dict.fromkeys(item["program"] for item in ranked))[:8],
            "terms": best["terms"], "compute_cost": budget.spent - start,
            "used_priors": [prior for prior in supported_priors if prior in executable_nodes
                or (prior.op != "call" and prior.expanded() in nodes)],
            "scope_boundaries": boundaries,
            "fit_diagnostics": {"operation": best.get("operation", "prior_scope"),
                "training_rows": len(rows), "validation_rows": len(validation),
                "training_loss": best["training_loss"], "validation_loss": best["loss"],
                "fitted_term_count": len(best["terms"]),
                "term_expressions": [term.describe() for term in best["terms"]]}}

    def reuse(trial, action):
        nonlinear = any(node.op == "if" or (node.op == "mul" and all(child.op != "const" for child in node.children))
            for node in _nodes(trial["program"].expanded()))
        if nonlinear:
            simpler = record(None, features, "compare_simpler_prior")
            if simpler is not None and max(simpler["training_loss"], simpler["loss"]) <= tolerance \
                    and simpler["program"].size < trial["program"].size:
                events.append({"action": "simplify_supported_prior", "compute_cost": 0})
                return finish(simpler)
        events.append({"action": action, "compute_cost": 0})
        return finish(trial)

    supported_priors = []
    for prior in priors:
        required = {name for name, _ in prior.observed_variables() if name != "__target__"}
        if required <= set(names):
            trial = record(prior, [prior], "test_prior")
            if trial:
                supported_priors.append(prior)
                if max(trial["training_loss"], trial["loss"]) <= tolerance:
                    return reuse(trial, "reuse_prior")
        else:
            before = budget.spent
            bindings = bind_prior(prior, names, rows, budget)
            events.append({"action": "compare_role_bindings", "compute_cost": budget.spent - before,
                "bindings_tested": len(bindings)})
            for binding in bindings[:3]:
                trial = record(binding["program"], [binding["program"]], "validate_role_binding",
                    extra={"mapping": binding["mapping"]})
                if trial:
                    supported_priors.append(binding["program"])
                    if max(trial["training_loss"], trial["loss"]) <= tolerance:
                        return reuse(trial, "reuse_bound_prior")
    # A retained nonlinear computation can be one ingredient of the new law,
    # even when a newly observed coordinate explains its remaining error.
    prior_trials = sorted((candidate for candidate in candidates
        if candidate["program"] in supported_priors), key=lambda item: item["loss"])
    for candidate in prior_trials[:4]:
        terms = list(dict.fromkeys([candidate["program"], *features]))
        trial = record(None, terms, "fit_prior_residual")
        if trial is not None and max(trial["training_loss"], trial["loss"]) <= tolerance:
            return finish(trial)
    for terms in ([], features):
        record(None, terms, "fit_affine")
    best = min(candidates, key=rank) if candidates else fallback
    if best["loss"] <= tolerance:
        return finish(best)

    # Scope boundaries come only from the current controls and old-model errors.
    # Fit the new branch on its own side; validation compares the entire if.
    tried_scopes = set()

    def scopes(terms):
        for prior in supported_priors:
            if not budget.spend(len(rows)):
                return
            errors = [(prior.predict(row) - row.target) ** 2 for row in rows]
            events.append({"action": "measure_prior_scope_residual", "compute_cost": len(rows)})
            proposals = []
            for name in names:
                values = sorted({row.features[name] for row in rows})
                for left, right in zip(values, values[1:]):
                    threshold = (left + right) / 2
                    low = [i for i, row in enumerate(rows) if row.features[name] < threshold]
                    high = [i for i, row in enumerate(rows) if row.features[name] >= threshold]
                    if min(len(low), len(high)) < 3:
                        continue
                    for old, new, old_low in ((low, high, True), (high, low, False)):
                        if mean(errors[i] for i in old) <= tolerance and mean(errors[i] for i in new) > tolerance:
                            proposals.append((len(old), name, threshold, old_low, new))
            for _, name, threshold, old_low, indices in sorted(proposals, reverse=True)[:4]:
                key = (prior, name, threshold, old_low, tuple(terms))
                if key in tried_scopes:
                    continue
                tried_scopes.add(key)
                training = [rows[i] for i in indices]
                branch_terms = list(dict.fromkeys([prior, *terms]))
                charge = (len(branch_terms) + 1) * len(training) + len(rows) + len(validation)
                if not budget.spend(charge):
                    return
                replacement = _fit_linear(branch_terms, training).simplified()
                predicate = Program("lt", children=(Program("feature", name=name), Program("const", value=threshold)))
                branches = (prior, replacement) if old_low else (replacement, prior)
                program = Program("if", children=(predicate, *branches))
                candidate = {"program": program, "terms": branch_terms,
                    "training_loss": _mse(program, rows), "loss": _mse(program, validation)}
                candidates.append(candidate)
                events.append({"action": "learn_executable_scope", "feature": name, "threshold": threshold,
                    "old_branch": "low" if old_low else "high", "compute_cost": charge,
                    "training_loss": candidate["training_loss"], "validation_loss": candidate["loss"]})

    scopes(features)
    best = min(candidates, key=rank) if candidates else fallback
    if best["loss"] <= tolerance:
        return finish(best)

    def absolute_relations():
        nonlocal best
        if not budget.spend(len(rows)):
            return
        residual = [row.target - best["program"].predict(row) for row in rows]
        proposals = [Program("abs", children=(feature,)) for feature in features]
        proposals.extend(Program("abs", children=(Program("sub", children=(left, right)),))
            for left, right in combinations(features, 2))
        ranked = []
        for term in proposals:
            if not budget.spend(len(rows)):
                break
            column = [term.predict(row) for row in rows]
            center, residual_center = mean(column), mean(residual)
            variance = sum((value - center) ** 2 for value in column)
            covariance = sum((value - center) * (error - residual_center) for value, error in zip(column, residual))
            ranked.append((covariance * covariance / variance if variance > 1e-12 else 0., term))
        events.append({"action": "rank_absolute_relations", "compute_cost": (len(ranked) + 1) * len(rows),
            "candidate_count": len(ranked)})
        for _, term in sorted(ranked, key=lambda item: (-item[0], item[1].describe()))[:4]:
            terms = list(dict.fromkeys([*features, *best["terms"], term]))
            trial = record(None, terms, "grow_absolute_relation", extra={"term": term.to_dict()})
            if trial is not None and rank(trial) < rank(best):
                best = trial
            if best["loss"] <= tolerance:
                return

    def hinge_relations(base_terms):
        nonlocal best
        # Threshold locations come from acquired coordinates. Joint fits use
        # training loss to refine them; validation still chooses the response.
        for feature in features:
            values = sorted({row.features[feature.name] for row in rows})
            boundaries = [(left + right) / 2 for left, right in zip(values[2:-3], values[3:-2])]
            if not boundaries:
                continue
            count = min(7, len(boundaries))
            grid = sorted({boundaries[round(index * (len(boundaries) - 1) / max(1, count - 1))]
                for index in range(count)} | {boundaries[0] + (boundaries[-1] - boundaries[0]) * index / 12
                    for index in range(13)})
            fitted = {}

            def fit_threshold(threshold, operation):
                nonlocal best
                if threshold in fitted:
                    return fitted[threshold]
                term = Program("max", children=(Program("const", value=0.),
                    Program("sub", children=(feature, Program("const", value=threshold)))))
                trial = record(None, [*base_terms, term], operation,
                    extra={"feature": feature.name, "threshold": threshold, "term": term.to_dict()})
                if trial is not None:
                    fitted[threshold] = trial
                    if rank(trial) < rank(best):
                        best = trial
                return trial

            for threshold in grid:
                if fit_threshold(threshold, "search_hinge_threshold") is None:
                    return
            optimum = min(grid, key=lambda threshold: fitted[threshold]["training_loss"])
            index = grid.index(optimum)
            low, high = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
            for _ in range(24):
                local = min(fitted.values(), key=lambda trial: trial["training_loss"])
                if low == high or max(local["training_loss"], local["loss"]) <= tolerance * .001:
                    break
                left, right = low + (high - low) / 3, high - (high - low) / 3
                a = fit_threshold(left, "refine_hinge_threshold")
                b = fit_threshold(right, "refine_hinge_threshold")
                if a is None or b is None:
                    return
                if a["training_loss"] <= b["training_loss"]:
                    high = right
                else:
                    low = left
            local = min(fitted.values(), key=lambda trial: trial["training_loss"])
            if max(local["training_loss"], local["loss"]) <= tolerance * .01:
                # A sufficiently fitted hinge need not keep the polynomial
                # columns used while searching for its threshold.
                for term in list(local["terms"]):
                    reduced = [item for item in local["terms"] if item != term]
                    trial = record(None, reduced, "prune_supported_terms")
                    if trial is None:
                        break
                    if max(trial["training_loss"], trial["loss"]) <= tolerance * .01:
                        local = trial
                        if rank(local) < rank(best):
                            best = local
            if max(best["training_loss"], best["loss"]) <= tolerance:
                return

    monomials = [(feature, (index,)) for index, feature in enumerate(features)]
    polynomial_terms, polynomial_bases = list(features), []
    for degree in range(2, max_depth + 1):
        if not budget.spend(len(rows)):
            break
        residual = [row.target - best["program"].predict(row) for row in rows]
        events.append({"action": "measure_model_residual", "degree": degree, "compute_cost": len(rows)})
        expanded = {}
        for term, factors in monomials:
            for index, feature in enumerate(features):
                key = tuple(sorted((*factors, index)))
                expanded.setdefault(key, Program("mul", children=(term, feature)))
        ranked = []
        for factors, term in expanded.items():
            if not budget.spend(len(rows)):
                break
            column = [term.predict(row) for row in rows]
            center, residual_center = mean(column), mean(residual)
            variance = sum((value - center) ** 2 for value in column)
            correlation = sum((value - center) * (error - residual_center) for value, error in zip(column, residual))
            ranked.append((correlation * correlation / variance if variance > 1e-12 else 0., factors, term))
        events.append({"action": "rank_residual_coordinates", "degree": degree,
            "compute_cost": len(ranked) * len(rows), "candidate_count": len(ranked)})
        ranked.sort(key=lambda item: (-item[0], item[1]))
        additions = [item[2] for item in ranked[:6]]
        polynomial_terms = list(dict.fromkeys([*polynomial_terms, *additions]))
        polynomial_bases.append(list(polynomial_terms))
        selected = list(dict.fromkeys([*features, *best["terms"]]))
        for term in additions:
            selected = list(dict.fromkeys([*selected, term]))
            trial = record(None, selected, "grow_residual_composition", extra={"degree": degree})
            if trial is not None and rank(trial) < rank(best):
                best = trial
            if best["loss"] <= tolerance:
                return finish(best)
        scopes(selected[:8])
        best = min(candidates, key=rank) if candidates else fallback
        if best["loss"] <= tolerance:
            return finish(best)
        if degree == 2:
            absolute_relations()
            if best["loss"] <= tolerance:
                return finish(best)
        monomials = [(term, factors) for _, factors, term in ranked[:8]]
    if max_depth < 2:
        absolute_relations()
    if best["loss"] > tolerance:
        # Jointly approximate abs and hinge terms can agree on sparse samples
        # while inventing a kink in an unobserved interval. Also fit the simpler
        # polynomial explanations directly, before the mixed residual basis.
        bases = [*polynomial_bases, list(dict.fromkeys([*features, *best["terms"]]))]
        for base in bases:
            hinge_relations(base)
            if max(best["training_loss"], best["loss"]) <= tolerance:
                break
    return finish(best)
