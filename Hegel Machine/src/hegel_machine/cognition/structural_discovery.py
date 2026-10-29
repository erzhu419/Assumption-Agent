"""Fit structural explanations, design discriminating setups, and apply them."""

from dataclasses import dataclass, replace
from bisect import bisect_left, bisect_right
from itertools import combinations
from random import Random

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, loss
from .language import EvaluationContext, Program
from .models import Candidate, Observation, Probe, SearchBudget
from .structure_lab import ObjectSchema, StructureBatch
from .synthesis import _solve


def _feature(schema, item, role):
    return Program("feature", name=schema.fields[item][role])


def _sum(children):
    return Program("object_sum", children=tuple(children))


def _focused(schema, role):
    return _sum(Program("mul", children=(_feature(schema, item, "f"), _feature(schema, item, role)))
                for item in schema.objects)


def _neighbors(schema, radius):
    return Program("neighbor_sum", children=(Program("const", value=radius), *(
        _feature(schema, item, role) for item in schema.objects for role in ("f", "x", "q"))))


def _relation_distances(rows, schema, budget):
    # Sparse selected rows retain the preparation frames of temporal responses.
    names = tuple(schema.feature_names)
    frames = {tuple(frame[name] for name in names): frame
              for row in rows for frame in (row.features, *row.history[-4:])}
    distances, complete = [], True
    for left, right in combinations(schema.objects, 2):
        if not budget.spend(len(frames)):
            complete = False
            break
        # An edge with equal focus or equal quantity contributes no response.
        distances.extend(abs(frame[schema.fields[left]["x"]] - frame[schema.fields[right]["x"]])
                         for frame in frames.values() if
                         (frame[schema.fields[left]["f"]] - frame[schema.fields[right]["f"]]) *
                         (frame[schema.fields[left]["q"]] - frame[schema.fields[right]["q"]]) != 0.)
    return sorted(set(distances)), complete


def _conservation(transitions, schema, budget):
    count, width = len(transitions), len(schema.objects)
    if not count or not width or not budget.spend(count * (width + 2)):
        return None
    columns = [[row["after"][schema.fields[item]["q"]] - row["before"][schema.fields[item]["q"]]
                - row["before"][schema.fields[item]["u"]] for row in transitions] for item in schema.objects]
    gram = [[sum(x * y for x, y in zip(left, right)) for right in columns[1:]] for left in columns[1:]]
    rhs = [-sum(x * y for x, y in zip(column, columns[0])) for column in columns[1:]]
    for index in range(len(gram)):
        gram[index][index] += 1e-10
    weights = [1., *_solve(gram, rhs)]
    errors = [sum(weight * column[index] for weight, column in zip(weights, columns)) for index in range(count)]
    error = sum(value * value for value in errors) / count
    quantity = _sum_terms([_feature(schema, item, "q") for item in schema.objects], weights)
    drive = _sum_terms([_feature(schema, item, "u") for item in schema.objects], weights)
    return {"kind": "conservation", "status": "proposed" if error < 1e-8 else "refuted",
            "program": quantity.to_dict(), "drive_program": drive.to_dict(), "weights": weights,
            "fit_error": error, "tested_transitions": count,
            "counterexamples": [] if error < 1e-8 else [transitions[max(range(count), key=lambda index: abs(errors[index]))]]}


@dataclass
class StructuralDiscovery:
    candidates: list[Candidate]
    fits: list[dict]
    hypotheses: list[dict]
    schema: ObjectSchema
    compute_cost: int
    radius_interval: tuple[float, float] | None = None
    search_complete: bool = True

    def to_dict(self):
        return {"candidates": [{"candidate_id": candidate.candidate_id, "program": candidate.program.to_dict(),
                                "operation": candidate.operation, "rationale": candidate.rationale}
                               for candidate in self.candidates], "fits": self.fits, "hypotheses": self.hypotheses,
                "schema": self.schema.to_dict(), "compute_cost": self.compute_cost,
                "radius_interval": list(self.radius_interval) if self.radius_interval else None,
                "search_complete": self.search_complete}

    @classmethod
    def from_dict(cls, payload):
        return cls([Candidate(row["candidate_id"], Program.from_dict(row["program"]), row["operation"], row["rationale"])
                    for row in payload["candidates"]], payload["fits"], payload["hypotheses"],
                   ObjectSchema.from_dict(payload["schema"]), payload["compute_cost"],
                   tuple(payload["radius_interval"]) if payload.get("radius_interval") else None,
                   payload.get("search_complete", True))


def _canonical_response(term):
    children = tuple(_canonical_response(child) for child in term.children)
    if term.op == "object_sum":
        children = tuple(sorted(children, key=lambda child: child.describe()))
    return replace(term, children=children)


def _response_blocks(rows, schema, response_terms=()):
    groups, reductions = [], []
    if response_terms:
        # Reopen the timing of the retained measured reductions. A range
        # residual does not by itself justify adding unrelated sensor fields.
        current = []
        for term in response_terms:
            while term.op in {"window_sum", "event_delta"}:
                term = term.children[0]
            term = _canonical_response(term)
            if term not in current:
                current.append(term)
        # The ordinary conservation candidate already proposes measured
        # quantity and drive totals; retain them when a source omitted one.
        for role in ("q", "u"):
            total = _canonical_response(_sum(_feature(schema, item, role) for item in schema.objects))
            if total not in current:
                current.append(total)
        reductions.append(current)
    else:
        for focused in (False, True):
            reductions.append([(_focused(schema, role) if focused else
                _sum(_feature(schema, item, role) for item in schema.objects)) for role in ("q", "u", "x")])
    for current in reductions:
        blocks = [current]
        if any(row.history for row in rows):
            events = [Program("event_delta", lag=1, children=(term,)) for term in current]
            blocks.extend((events, [*current, *events]))
            if response_terms:
                blocks.extend([Program("window_sum", lag=width, children=(term,))
                               for term in terms] for width in (2, 3, 4)
                              for terms in (current, events))
            else:
                blocks.extend([Program("window_sum", lag=width, children=(term,))
                               for term in [*current, *events]] for width in (2, 3, 4))
        groups.append(blocks)
    return groups


def discover_structure(observations, schema, *, budget, transitions=(), radius_candidates=None,
                       families=None, response_terms=None):
    """Jointly revise a measured response basis and the range of its missing interaction."""
    rows, start = list(observations), budget.spent
    discovery = _discover_structure(rows, schema, budget=budget, transitions=transitions,
        radius_candidates=radius_candidates, families=families, response_terms=response_terms,
        response_blocks=_response_blocks(rows, schema, response_terms)
                        if response_terms and families is None else ())
    if (response_terms or families is not None or not rows or
            any(fit["training_loss"] < 1e-8 for fit in discovery.fits)):
        return discovery
    # An unexplained response can combine measured reductions and a relation.
    # Select the reductions from paid data before fitting them jointly with
    # each candidate range; an instantaneous range winner is not a prerequisite.
    from .research_programs import _Fitter, _rank
    fitter, seeds = _Fitter(rows, budget, .5), []
    for blocks in _response_blocks(rows, schema):
        fitted = [value for terms in blocks if (value := fitter.fit(terms)) is not None]
        if fitted:
            seeds.append(min(fitted, key=_rank))
    seeds.sort(key=_rank)
    alternatives = [discovery]
    for seed in seeds:
        revised = _discover_structure(rows, schema, budget=budget, transitions=transitions,
            radius_candidates=radius_candidates, response_terms=seed["terms"])
        alternatives.append(revised)
        if any(fit["training_loss"] < 1e-8 for fit in revised.fits):
            break
    ranked = sorted(((candidate, fit, result) for result in alternatives
        for candidate, fit in zip(result.candidates, result.fits)),
        key=lambda item: (0. if item[1]["training_loss"] < 1e-8 else item[1]["training_loss"],
                          item[1]["effective_dimension"], item[0].program.size))[:8]
    if ranked:
        discovery = ranked[0][2]
        discovery = StructuralDiscovery([item[0] for item in ranked], [item[1] for item in ranked],
            discovery.hypotheses, schema, budget.spent - start, ranked[0][1]["radius_interval"],
            all(result.search_complete for result in alternatives))
    discovery.compute_cost = budget.spent - start
    return discovery


def _discover_structure(observations, schema, *, budget, transitions=(), radius_candidates=None,
                        families=None, response_terms=None, response_blocks=()):
    """Return executable scalar candidates and falsifiable structural proposals.

    All column construction, fitting and candidate prediction passes spend the
    ordinary candidate-row budget. The schema contributes only public field
    roles. It supplies neither an interaction radius nor a target mechanism.
    """
    start, rows = budget.spent, list(observations)
    if not rows or not schema.objects:
        return StructuralDiscovery([], [], [], schema, 0)
    count, cache, records, complete = len(rows), {}, [], True
    response_bases = [tuple(response_terms)] if response_terms else []
    covering_range = False
    targets = [row.target for row in rows]
    target_mean = sum(targets) / count
    tolerance = max(1e-10, sum((value - target_mean) ** 2 for value in targets) / count * 1e-8)
    total_q = _sum(_feature(schema, item, "q") for item in schema.objects)
    total_u = _sum(_feature(schema, item, "u") for item in schema.objects)
    focal_q, focal_u, focal_x = (_focused(schema, role) for role in ("q", "u", "x"))
    labeled = [Program("mul", children=(_feature(schema, item, "f"), _feature(schema, item, role)))
               for item in schema.objects for role in ("q", "x")]

    def fit(terms, family, radius=None, window=None, response_basis=None):
        nonlocal complete
        if families is not None and family not in families:
            return None
        terms = tuple(dict.fromkeys(terms))
        keys = {term: _canonical_response(term) if response_blocks else term for term in terms}
        missing = [term for term in terms if keys[term] not in cache]
        if (len(missing) + 2) * count > budget.remaining:
            complete = False
            return None
        for term in missing:
            budget.spend(count)
            cache[keys[term]] = [term.predict(row) for row in rows]
        budget.spend(count)
        coefficients = _linear_fit([cache[keys[term]] for term in terms], targets)
        program = _sum_terms(terms, coefficients[1:], coefficients[0])
        budget.spend(count)
        error = loss(program, rows, 0., .5)
        dimension = sum(abs(coefficient) > 1e-7 for coefficient in coefficients[1:])
        record = {"family": family, "training_loss": error, "effective_dimension": dimension,
                  "raw_dimension": len(schema.feature_names), "radius": radius, "window": window,
                  "response_basis": response_basis,
                  "fit_terms": terms,
                  "coefficients": list(coefficients),
                  "candidate_id": f"structure:{family}:{len(records)}", "program": program}
        if response_basis is not None:
            record["response_terms"] = [term.to_dict() for term in response_bases[response_basis]]
        elif response_terms:
            record["response_terms"] = [term.to_dict() for term in response_terms]
        records.append(record)
        return record

    if not response_terms:
        fit((total_q, total_u), "conserved_total")
        fit((focal_q, focal_u, focal_x), "focal_coordinates")
        fit((*labeled, focal_u, total_q), "object_specific")
    if transitions and len(transitions) * (len(schema.objects) + 2) > budget.remaining:
        complete = False
    conservation = _conservation(transitions, schema, budget) if not response_terms else None
    if conservation and conservation["status"] == "proposed":
        fit((Program.from_dict(conservation["program"]), Program.from_dict(conservation["drive_program"])),
            "conserved_combination")
    distances, distance_complete = _relation_distances(rows, schema, budget)
    complete = complete and distance_complete
    boundaries = [(left + right) / 2 for left, right in zip(distances, distances[1:])]
    if radius_candidates is not None:
        proposed = sorted(set(radius_candidates))
    elif boundaries:
        proposed = sorted({0., distances[-1] + 1e-6, *(boundaries[round(index * (len(boundaries) - 1) / 15)]
                          for index in range(16))})
    else:
        proposed = []

    def fit_radius(radius, leading=None, response_indices=None):
        neighbor = _neighbors(schema, radius)
        models = [((focal_q, focal_u, neighbor), "local_exchange", None, None),
                  ((focal_q, focal_u, neighbor, total_q), "local_and_global", None, None),
                  ((*labeled, focal_u, neighbor, total_q), "object_specific_exchange", None, None),
                  ((Program("event_delta", lag=1, children=(neighbor,)),), "event_change", None, None)]
        models.extend(((Program("window_sum", lag=window, children=(neighbor,)),),
                       "window_exchange", window, None) for window in (2, 3, 4))

        def joint_models(basis):
            terms = response_bases[basis]
            event = Program("event_delta", lag=1, children=(neighbor,))
            joint = [((*terms, neighbor), "joint_current", None, basis)]
            if any(row.history for row in rows):
                joint.append(((*terms, neighbor, event), "joint_event", None, basis))
                for width in (2, 3, 4):
                    for term, family in ((neighbor, "joint_window"), (event, "joint_window_event")):
                        joint.append(((*terms, neighbor, Program("window_sum", lag=width, children=(term,))),
                                      family, width, basis))
            return joint

        # With a measured response basis, coarse coverage asks which range
        # completes that response. Compare simpler standalone explanations
        # at the first edge and again at a successful edge, rather than
        # relearning all seven unconditioned families at every coarse point.
        if response_indices is not None or (covering_range and response_blocks):
            models = []
        models.extend(model for basis in (range(len(response_bases)) if response_indices is None
                                         else response_indices) for model in joint_models(basis))
        fitted = []
        def compare(proposals):
            for terms, family, window, basis in proposals:
                if leading is not None and (family, window, basis) != leading:
                    continue
                record = fit(terms, family, radius, window, basis)
                if record is not None:
                    fitted.append(record)
        compare(models)
        return fitted

    if radius_candidates is None and proposed:
        # Cover the range before spending on its resolution. A fixed ascending
        # grid used to exhaust small budgets on low radii, leaving no compute
        # for the boundary suggested by the observed response.
        probe_models = fit_radius(proposed[len(proposed) // 2])
        broad_start = budget.spent
        broad_limit = budget.remaining // 2
        remaining = [radius for radius in proposed
                     if not probe_models or radius != probe_models[0]["radius"]]
        visited = [record["radius"] for record in probe_models]
        covering_range = True
        while remaining and budget.spent - broad_start < broad_limit:
            radius = max(remaining, key=lambda value: min(
                abs(value - previous) for previous in visited)) if visited else remaining[0]
            remaining.remove(radius)
            before_fit = budget.spent
            fit_radius(radius)
            visited.append(radius)
            if budget.spent == before_fit:
                break
        covering_range = False
        complete = complete and not remaining
    else:
        for radius in proposed:
            fit_radius(radius)
    local_records = [record for record in records if record["radius"] is not None]
    if response_blocks and local_records:
        prior = min(local_records, key=lambda record: record["training_loss"])
        if prior["training_loss"] > tolerance:
            # A midpoint mismatch can be a range error. Cover the measured
            # range first, then compare changed responses at its best paid
            # edge before spending the remaining work on finer boundaries.
            from .research_programs import _Fitter, _rank
            fitter = _Fitter(rows, budget, .5)
            fitter.columns = cache
            seeds, first_new = [], len(response_bases)
            for blocks in response_blocks:
                fits = [result for block in blocks if (result := fitter.fit(block)) is not None]
                if fits:
                    seeds.append(min(fits, key=_rank))
            for seed in sorted(seeds, key=_rank):
                terms = tuple(seed["terms"])
                if terms and not any(set(terms) == {_canonical_response(term) for term in known}
                                     for known in response_bases):
                    response_bases.append(terms)
            if len(response_bases) > first_new:
                local_records.extend(fit_radius(prior["radius"],
                    response_indices=range(first_new, len(response_bases))))
    if local_records and radius_candidates is None:
        best = min(local_records, key=lambda record: record["training_loss"])
        # Refine the two untested cells adjoining the best observed response.
        # Walking sixteen consecutive distances can miss a boundary when a
        # long trajectory put hundreds of distinct distances in a coarse cell.
        for _ in range(32):
            if best["training_loss"] <= tolerance:
                break
            # Refitting coefficients makes range loss non-monotonic. Once the
            # best cell is exhausted, continue another measured neighborhood;
            # its slightly worse edge may precede the consistent explanation.
            selected_family = (best["family"], best["window"], best["response_basis"])
            neighbors = sorted((record for record in records
                if (record["family"], record["window"], record["response_basis"]) == selected_family),
                key=lambda record: record["training_loss"])
            # Another response's fit did not test this response at that edge.
            tested = {record["radius"] for record in neighbors}
            cell = None
            for record in neighbors:
                center = record["radius"]
                low = max((value for value in tested if value < center), default=0.)
                high = min((value for value in tested if value > center), default=distances[-1] + 1e-6)
                cells = [(bisect_right(boundaries, low), bisect_left(boundaries, center)),
                         (bisect_right(boundaries, center), bisect_left(boundaries, high))]
                left, right = max(cells, key=lambda item: item[1] - item[0])
                if left < right:
                    cell = (left, right)
                    break
            if cell is None:
                break
            left, right = cell
            radius = boundaries[(left + right - 1) // 2]
            refined = fit_radius(radius, leading=selected_family)
            if not refined:
                break
            best = min([best, *refined], key=lambda record: record["training_loss"])
            if best["training_loss"] <= tolerance:
                # A flexible family can locate the edge before a simpler law
                # does. Compare their dimensions once at that successful edge.
                fit_radius(best["radius"])
                break
        if best["training_loss"] <= tolerance:
            # Meeting the response error goal does not identify its relation
            # cell. A weak observed edge can be diluted by many inactive rows.
            # Compare its immediate neighboring cells with the same response
            # before turning the selected cell into a later search domain.
            family = (best["family"], best["window"], best["response_basis"])
            cell = bisect_left(distances, best["radius"])
            tested = {bisect_left(distances, record["radius"]) for record in records
                if record["radius"] is not None and
                (record["family"], record["window"], record["response_basis"]) == family}
            for index in (cell - 2, cell):
                if 0 <= index < len(boundaries) and index + 1 not in tested:
                    fit_radius(boundaries[index], leading=family)
    if not records:
        return StructuralDiscovery([], [], [conservation] if conservation else [], schema,
                                   budget.spent - start, search_complete=complete)
    best_loss = min(record["training_loss"] for record in records)
    records.sort(key=lambda record: (record["training_loss"] > best_loss + tolerance,
        record["effective_dimension"] if record["training_loss"] <= best_loss + tolerance else record["training_loss"],
        record["program"].size, record["training_loss"]))
    candidates, selected, seen = [], [], set()
    for record in records:
        if record["program"] in seen:
            continue
        seen.add(record["program"])
        candidates.append(Candidate(record["candidate_id"], record["program"], "representation",
                                    f"Use observed {record['family'].replace('_', ' ')} structure"))
        fit_record = {key: value for key, value in record.items() if key not in ("program", "fit_terms")}
        fit_record["fit_terms"] = [term.to_dict() for term in record["fit_terms"]]
        radius = record["radius"]
        fit_record["radius_interval"] = None if radius is None else (
            max((distance for distance in distances if distance < radius), default=0.),
            min((distance for distance in distances if distance >= radius), default=max(distances, default=0.)))
        selected.append(fit_record)
        if len(candidates) == 8:
            break
    hypotheses = ([conservation] if conservation else []) + [
        {"kind": kind, "status": "proposed", "counterexamples": []} for kind in ("permutation", "translation")]
    if len(response_bases) > 1:
        previous_loss = min(record["training_loss"] for record in records if record["response_basis"] in (None, 0))
        hypotheses.append({"kind": "response_basis_revision", "source": "inherited",
            "reason": "response_candidates_compared_at_shared_ranges",
            "previous_loss": previous_loss, "best_loss": best_loss, "counterexamples": [],
            "status": "fit_improved" if best_loss < previous_loss else "proposed"})
    radius = selected[0]["radius"]
    interval = None
    if radius is not None:
        interval = selected[0]["radius_interval"]
        hypotheses.append({"kind": "locality", "status": "proposed", "radius": radius, "counterexamples": []})
    return StructuralDiscovery(candidates, selected, hypotheses, schema, budget.spent - start,
                               interval, complete)


def _observed_focus_settings(schema, paired_observations):
    """Bind object membership to controls actually witnessed by an instrument."""
    settings = {}
    for command, measured in paired_observations:
        values = {item: measured.features.get(fields["f"], 0.) for item, fields in schema.fields.items()}
        active = [item for item, value in values.items() if value == 1.]
        if len(active) == 1 and sum(values.values()) == 1. and schema.focus_control in command.features:
            settings.setdefault(active[0], []).append(command.features[schema.focus_control])
    return {item: sorted(values)[len(values) // 2] for item, values in settings.items()}


def distinguishing_probes(discovery, spec, *, seed=71, length=6, paired_observations=()):
    """Construct whole paired trajectories so event/window semantics agree."""
    schema, rng = discovery.schema, Random(seed)
    controls, object_ids = schema.controls, list(schema.controls)
    focus_settings = _observed_focus_settings(schema, paired_observations)
    can_permute = all(item in focus_settings for item in object_ids)
    bases = []
    focused_objects = []
    for _ in range(length):
        point = {name: low + (.25 + .5 * rng.random()) * (high - low)
                 for name, (low, high) in spec.feature_bounds.items()}
        focused = rng.choice(object_ids) if can_permute else None
        if focused is not None:
            point[schema.focus_control] = focus_settings[focused]
        focused_objects.append(focused)
        bases.append(point)
    translated, permuted = [], []
    shift = .15 * min(spec.feature_bounds[fields["x"]][1] - spec.feature_bounds[fields["x"]][0]
                      for fields in controls.values())
    for base, focused in zip(bases, focused_objects):
        translated.append({name: value + shift if name in {fields["x"] for fields in controls.values()} else value
                           for name, value in base.items()})
        if not can_permute:
            continue
        changed = dict(base)
        for index, item in enumerate(object_ids):
            destination = object_ids[(index + 1) % len(object_ids)]
            for role in ("q", "x", "u"):
                changed[controls[destination][role]] = base[controls[item][role]]
        destination = object_ids[(object_ids.index(focused) + 1) % len(object_ids)]
        changed[schema.focus_control] = focus_settings[destination]
        permuted.append(changed)
    plans = [{"kind": kind, "baseline": Probe(f"structure_{kind}_base", tuple(bases)),
              "probe": Probe(f"structure_{kind}_changed", tuple(changed))}
             for kind, changed in (("translation", translated), ("permutation", permuted)) if changed]
    known_focus = next((item for item in object_ids if item in focus_settings), None)
    if discovery.radius_interval is not None and known_focus is not None and len(object_ids) > 1:
        far_base, far_changed = [], []
        for _ in range(length):
            base = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
            base[schema.focus_control] = focus_settings[known_focus]
            for index, item in enumerate(object_ids):
                low, high = spec.feature_bounds[controls[item]["x"]]
                base[controls[item]["x"]] = low + .05 * (high - low) if item == known_focus else high - .05 * (high - low)
                base[controls[item]["q"]] = rng.uniform(-.3, .3)
            changed = dict(base)
            name = controls[next(item for item in reversed(object_ids) if item != known_focus)]["q"]
            low, high = spec.feature_bounds[name]
            changed[name] = base[name] + .25 * (high - low)
            far_base.append(base)
            far_changed.append(changed)
        plans.append({"kind": "locality", "baseline": Probe("structure_far_base", tuple(far_base)),
                      "probe": Probe("structure_far_changed", tuple(far_changed))})
    return plans


def confirm_structure(discovery, paired_batches, *, budget):
    """Use new controlled outcomes to support a proposal or retain a counterexample."""
    hypotheses = [{**item, "counterexamples": list(item["counterexamples"])} for item in discovery.hypotheses]
    for experiment in paired_batches:
        base, changed = experiment["baseline"], experiment["batch"]
        count = min(len(base.observations), len(changed.observations))
        if not budget.spend(count):
            break
        errors = [(left.target - right.target) ** 2 for left, right in zip(base.observations, changed.observations)]
        error = sum(errors) / len(errors)
        hypothesis = next((item for item in hypotheses if item["kind"] == experiment["kind"]), None)
        if hypothesis is not None:
            hypothesis.update(status="supported" if error < 1e-8 else "refuted", prospective_error=error,
                              tested_rows=count)
            if error >= 1e-8:
                index = max(range(count), key=errors.__getitem__)
                hypothesis["counterexamples"].append({"before": base.observations[index].features,
                    "after": changed.observations[index].features, "before_target": base.observations[index].target,
                    "after_target": changed.observations[index].target})
        conservation = next((item for item in hypotheses if item["kind"] == "conservation"), None)
        if conservation is not None and budget.spend(3 * len(changed.transitions)):
            program, drive = Program.from_dict(conservation["program"]), Program.from_dict(conservation["drive_program"])
            residuals = []
            for row in changed.transitions:
                before, after = Observation(row["before"], 0.), Observation(row["after"], 0.)
                residuals.append(program.predict(after) - program.predict(before) - drive.predict(before))
            measured = sum(value * value for value in residuals) / len(residuals)
            if measured >= 1e-8:
                conservation["status"] = "refuted"
                conservation["counterexamples"].append(changed.transitions[max(range(len(residuals)), key=lambda index: abs(residuals[index]))])
            elif conservation["status"] != "refuted":
                conservation["status"] = "supported"
            conservation["prospective_error"] = measured
    return hypotheses


def project_batch(batch, schema):
    """Coarsen already paid object readings without buying the same data again."""
    source = batch.schema
    def project(features):
        result = {}
        for item in schema.objects:
            groups = [key for key in source.objects if set(source.members[key]).issubset(schema.members[item])]
            for role in ("q", "u", "f"):
                result[schema.fields[item][role]] = sum(features[source.fields[group][role]] for group in groups)
            total = sum(len(source.members[group]) for group in groups)
            result[schema.fields[item]["x"]] = sum(features[source.fields[group]["x"]] * len(source.members[group])
                                                     for group in groups) / total
        return result
    observations = [Observation(project(row.features), row.target, tuple({**project(past), "__target__": past["__target__"]}
                                                                         for past in row.history)) for row in batch.observations]
    transitions = [{"before": project(row["before"]), "after": project(row["after"])} for row in batch.transitions]
    return StructureBatch(observations, transitions, schema, 0, 0)


def boundary_probe(discovery, spec, *, index=0, length=5, paired_observations=()):
    """Intervene inside the unresolved interaction interval, using public bounds."""
    if discovery.radius_interval is None or len(discovery.schema.controls) < 2:
        return None
    low_radius, high_radius = discovery.radius_interval
    if high_radius - low_radius < 1e-6:
        return None
    radius = (low_radius + high_radius) / 2
    schema, sequence = discovery.schema, []
    objects = list(schema.controls)
    focus_settings = _observed_focus_settings(schema, paired_observations)
    known_focus = next((item for item in objects[:2] if item in focus_settings), None)
    focus_low, focus_high = spec.feature_bounds[schema.focus_control]
    selector_settings = ([focus_settings[known_focus]] if known_focus is not None else
        [focus_low + (focus_high - focus_low) * item / (len(objects) - 1) for item in range(len(objects))])
    position_bounds = [spec.feature_bounds[schema.controls[item]["x"]] for item in objects]
    left = max(bounds[0] for bounds in position_bounds)
    right = min(bounds[1] for bounds in position_bounds)
    if radius >= right - left:
        return None
    for step in range(length):
        point = {name: (low + high) / 2 for name, (low, high) in spec.feature_bounds.items()}
        # Anonymous object order does not identify a selector setting. Reuse
        # witnessed membership, or cover its public domain within this probe.
        point[schema.focus_control] = selector_settings[step % len(selector_settings)]
        for number, item in enumerate(objects):
            fields = schema.controls[item]
            point[fields["x"]] = left if number == 0 else left + radius if number == 1 else right
            q_low, q_high = spec.feature_bounds[fields["q"]]
            contrast = .3 if step % 2 == 0 else .15
            fraction = .5 - contrast if number == 0 else .5 + contrast if number == 1 else .5
            point[fields["q"]] = q_low + fraction * (q_high - q_low)
        sequence.append(point)
    return Probe(f"structure_boundary_{index}", tuple(sequence))


class StructuredEnvironment:
    """Expose the measurement configuration selected by structural research."""

    def __init__(self, laboratory, *, objects=None, scale=1):
        self.laboratory, self.objects, self.scale = laboratory, objects, scale
        self.spec = laboratory.spec_for(objects, scale)
        self.object_schema = laboratory.schema_for(objects, scale)

    def observe(self, probe):
        return self.laboratory.acquire(probe, objects=self.objects, scale=self.scale).observations


def run_structural_research(laboratory, *, config=None, library=None, measurement_budget=4096):
    """Discover, falsify, reconfigure measurements, and continue the ordinary agent.

    Fine pilot measurements are projected into competing views without buying
    them again. New experiments execute the chosen measurement configuration.
    The final workspace retains every acquired trajectory in that view.
    """
    from dataclasses import replace
    from time import perf_counter
    from .controller import CognitiveAgent
    from .memory import KnowledgeLibrary
    from .models import ResearchConfig
    from .workspace import ResearchWorkspace

    started = perf_counter()
    config = config or ResearchConfig(observation_budget=192, compute_budget=160000, max_rounds=3)
    library = library if library is not None else KnowledgeLibrary()
    initial_costs = dict(laboratory.costs)
    pilot_length = min(24, config.observation_budget // 4, measurement_budget // (9 * len(laboratory.object_ids)))
    if pilot_length < 8:
        raise ValueError("object research needs two pilot trajectories and a prospective comparison")
    pilot_probes = [laboratory.random_probe(config.seed + 991 * index, length=pilot_length,
                                           name=f"structure_pilot_{index}") for index in range(2)]
    pilots = [laboratory.acquire(probe) for probe in pilot_probes]
    # Initial fine readings support all these projections. Future acquisition
    # cost differs, and a subset may lose a causally relevant object.
    views = [(None, scale) for scale in dict.fromkeys((1, 2, len(laboratory.object_ids)))]
    views.append((laboratory.object_ids[:max(1, len(laboratory.object_ids) // 2)], 1))
    preparation = SearchBudget(max(1, config.compute_budget - max(16000, config.compute_budget // 5)))
    selection_limit = min(preparation.remaining, config.compute_budget // 2)
    assessed = []
    fallback = initial_prediction(pilots[0].observations, .5)
    baseline_program = Program("const", value=fallback)
    if not preparation.spend(pilot_length):
        raise ValueError("computation budget cannot compare object measurements")
    baseline_loss = loss(baseline_program, pilots[1].observations, fallback, .5)
    for index, (objects, scale) in enumerate(views):
        schema = laboratory.schema_for(objects, scale)
        training, validation = [project_batch(batch, schema) for batch in pilots]
        local_limit = min(preparation.remaining, selection_limit // len(views))
        local = SearchBudget(max(0, local_limit - 8 * pilot_length))
        discovery = discover_structure(training.observations, schema, budget=local, transitions=training.transitions)
        preparation.spend(local.spent)
        ranked = []
        for candidate, fit in zip(discovery.candidates, discovery.fits):
            if not preparation.spend(pilot_length):
                break
            measured = loss(candidate.program, validation.observations, fallback, .5)
            ranked.append((measured, fit["effective_dimension"], candidate.program.size, candidate, fit))
        if not ranked:
            continue
        best = min(ranked, key=lambda item: item[:3])
        assessed.append({"objects": objects, "scale": scale, "schema": schema, "discovery": discovery,
                         "candidate": best[3], "fit": best[4], "validation_loss": best[0],
                         "cost_per_row": 3 * len(schema.objects), "compute_cost": local.spent + pilot_length * len(ranked)})
    if not assessed:
        raise ValueError("computation budget cannot fit any object representation")
    best_loss = min(item["validation_loss"] for item in assessed)
    tolerance = max(1e-8, min(config.minimum_improvement, .005 * baseline_loss))
    chosen = min((item for item in assessed if item["validation_loss"] <= best_loss + tolerance),
                 key=lambda item: (item["cost_per_row"], item["validation_loss"]))
    objects, scale, schema = chosen["objects"], chosen["scale"], chosen["schema"]
    environment = StructuredEnvironment(laboratory, objects=objects, scale=scale)
    discovery = chosen["discovery"]
    # The prospective winner, including its actual radius, determines the next
    # experiment. Do not query the training winner when validation changed it.
    winner_index = discovery.candidates.index(chosen["candidate"])
    discovery.candidates.insert(0, discovery.candidates.pop(winner_index))
    discovery.fits.insert(0, discovery.fits.pop(winner_index))
    radius = discovery.fits[0]["radius"]
    discovery.radius_interval = discovery.fits[0]["radius_interval"]
    projected = [project_batch(batch, schema) for batch in pilots]
    rows = [row for batch in projected for row in batch.observations]
    transitions = [row for batch in projected for row in batch.transitions]
    queried, experiments = list(pilot_probes), []
    # A validation winner can still contradict its training trajectory. Its
    # radius cell is then a tested candidate, not a surviving graph boundary.
    # Reconcile all paid pilot evidence before narrowing that boundary.
    if radius is not None:
        local = SearchBudget(min(preparation.remaining, len(rows) * 400))
        consolidated = discover_structure(rows, schema, budget=local, transitions=transitions,
            families=(chosen["fit"]["family"],),
            response_terms=[Program.from_dict(term) for term in chosen["fit"].get("response_terms", [])])
        preparation.spend(local.spent)
        if consolidated.candidates:
            discovery = consolidated
            radius = discovery.fits[0]["radius"]
        if discovery.fits[0]["training_loss"] > max(1e-8, config.minimum_improvement * .01):
            discovery.radius_interval = None
    prior_records = library.structural_priors
    prior_assessments = []
    recalled = []
    for prior in [record for record in prior_records if record.get("kind") != "mechanism"][-4:]:
        prior_schema = ObjectSchema.from_dict(prior["schema"])
        if len(prior_schema.objects) != len(schema.objects):
            continue
        mapping = {prior_schema.fields[left][role]: schema.fields[right][role]
                   for left, right in zip(prior_schema.objects, schema.objects) for role in ("q", "x", "u", "f")}
        old = Program.from_dict(prior["program"]).remap_features(mapping)
        if not preparation.spend(len(rows)):
            break
        residuals = [(old.predict(row) - row.target) ** 2 for row in rows]
        error = sum(residuals) / len(residuals)
        status = "retained" if error <= chosen["validation_loss"] + tolerance else "revised"
        prior_assessments.append({"record_id": prior["record_id"], "status": status, "loss": error,
            "retained_source_support": prior["support_rows"], "program": old.to_dict(),
            "counterexample": None if status == "retained" else {
                "features": rows[max(range(len(rows)), key=residuals.__getitem__)].features,
                "reason": "previous executable structural explanation fails acquired outcomes"}})
        hypotheses = []
        for hypothesis in prior["hypotheses"]:
            hypothesis = {**hypothesis, "counterexamples": list(hypothesis["counterexamples"])}
            for key in ("program", "drive_program"):
                if key in hypothesis:
                    hypothesis[key] = Program.from_dict(hypothesis[key]).remap_features(mapping).to_dict()
            hypotheses.append(hypothesis)
        recalled.append((prior_assessments[-1], StructuralDiscovery([], [], hypotheses, schema, 0)))
        if status == "retained":
            discovery.candidates.insert(0, Candidate(f"{prior['record_id']}:reuse", old, "representation"))
            discovery.fits.insert(0, {**chosen["fit"], "family": "prior_reuse", "training_loss": error,
                                     "candidate_id": f"{prior['record_id']}:reuse"})

    def available(length):
        observations_spent = laboratory.costs["observation_cost"] - initial_costs["observation_cost"]
        measurements_spent = laboratory.costs["measurement_cost"] - initial_costs["measurement_cost"]
        return (observations_spent + length + 8 <= config.observation_budget and
                measurements_spent + (length + 8) * chosen["cost_per_row"] <= measurement_budget)

    def acquire(probe):
        batch = laboratory.acquire(probe, objects=objects, scale=scale)
        rows.extend(batch.observations)
        transitions.extend(batch.transitions)
        queried.append(probe)
        return batch

    pairs = []
    pilot_pairs = [(Observation(point, row.target), row) for probe, batch in zip(pilot_probes, projected)
                   for point, row in zip(probe.sequence, batch.observations)]
    for plan in distinguishing_probes(discovery, environment.spec, seed=config.seed + 123, length=4,
                                      paired_observations=pilot_pairs):
        if not available(plan["baseline"].cost + plan["probe"].cost):
            break
        base, changed = acquire(plan["baseline"]), acquire(plan["probe"])
        pairs.append({"kind": plan["kind"], "baseline": base, "batch": changed})
        experiments.append({"kind": plan["kind"], "baseline": plan["baseline"].probe_id,
                            "probe": plan["probe"].probe_id, "observations": base.observation_cost + changed.observation_cost,
                            "measurement_cost": base.measurement_cost + changed.measurement_cost})
    confirmed_hypotheses = confirm_structure(discovery, pairs, budget=preparation)
    for assessment, proposal in recalled:
        assessment["hypotheses_in_new_context"] = confirm_structure(proposal, pairs, budget=preparation)
    # A paid intervention inside the surviving graph boundary shrinks the set
    # of possible interaction laws. Refits keep the complete bought evidence.
    refinements = []
    for index in range(8):
        probe = boundary_probe(discovery, environment.spec, index=index, paired_observations=pilot_pairs)
        if probe is None or not available(probe.cost):
            break
        left, right = discovery.radius_interval
        family = discovery.fits[0]["family"]
        if family == "prior_reuse":
            family = chosen["fit"]["family"]
        # Refine the empirically selected explanation's uncertain graph. The
        # later ordinary-agent comparison still admits different explanations.
        local_limit = min(preparation.remaining, (len(rows) + probe.cost) * 55)
        if local_limit < (len(rows) + probe.cost) * 45:
            break
        batch = acquire(probe)
        local = SearchBudget(local_limit)
        radii = [left + fraction * (right - left) for fraction in (.25, .5, .75)]
        revised = discover_structure(rows, schema, budget=local, transitions=transitions,
            radius_candidates=radii, families=(family,),
            response_terms=[Program.from_dict(term) for term in discovery.fits[0].get("response_terms", [])])
        preparation.spend(local.spent)
        if not revised.candidates:
            break
        refinements.append({"probe_id": probe.probe_id, "before_interval": [left, right],
                            "after_interval": revised.radius_interval, "compute_cost": local.spent,
                            "measurement_cost": batch.measurement_cost,
                            "training_loss": revised.fits[0]["training_loss"]})
        discovery = revised
    discovery.hypotheses = confirmed_hypotheses
    # New non-boundary counterexamples must also participate in the final fit.
    residual_loss = None
    if preparation.spend(len(rows)):
        residual_loss = loss(discovery.candidates[0].program, rows, fallback, .5)
    if not refinements and len(rows) > 2 * pilot_length and residual_loss is not None and residual_loss > 1e-8:
        local = SearchBudget(min(preparation.remaining, max(0, len(rows) * 120)))
        radii = [radius] if radius is not None else []
        revised = discover_structure(rows, schema, budget=local, transitions=transitions, radius_candidates=radii,
            response_terms=[Program.from_dict(term) for term in discovery.fits[0].get("response_terms", [])])
        preparation.spend(local.spent)
        if revised.candidates:
            revised.hypotheses = confirmed_hypotheses
            discovery = revised
    candidate = discovery.candidates[0]
    if not preparation.spend(len(rows)):
        raise ValueError("computation budget cannot initialize the acquired structural workspace")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows,
        reference=projected[1].observations, queried=queried, incumbent=candidate, evidence_champion=candidate,
        bank=discovery.candidates[:8], fallback=fallback,
        initial_loss=loss(baseline_program, rows, fallback, .5))
    costs = {name: laboratory.costs[name] - initial_costs[name] for name in initial_costs}
    remaining = min(config.observation_budget - costs["observation_cost"],
                    (measurement_budget - costs["measurement_cost"]) // chosen["cost_per_row"])
    episode = CognitiveAgent(replace(config, observation_budget=remaining,
                                    compute_budget=config.compute_budget - preparation.spent), library).run(
                                        environment, research_state=workspace.to_dict())
    record = {"record_id": f"structural-prior:{len(prior_records)}", "schema": schema.to_dict(),
              "program": episode.program.to_dict(), "hypotheses": discovery.hypotheses,
              "support_rows": len(episode.observations), "context": dict(environment.spec.context),
              "revises": [item["record_id"] for item in prior_assessments if item["status"] == "revised"]}
    library.structural_priors.append(record)
    costs = {name: laboratory.costs[name] - initial_costs[name] for name in initial_costs}
    return {"task_id": environment.spec.task_id, "selected_objects": list(objects or laboratory.object_ids),
        "selected_scale": scale, "selected_schema": schema.to_dict(), "measurement_budget": measurement_budget,
        "selection": [{"objects": list(item["objects"] or laboratory.object_ids), "scale": item["scale"],
            "cost_per_row": item["cost_per_row"], "validation_loss": item["validation_loss"],
            "effective_dimension": item["fit"]["effective_dimension"], "compute_cost": item["compute_cost"],
            "family": item["fit"]["family"]} for item in assessed],
        "measurement_reconfiguration": {"before_scale": 1, "after_scale": scale,
            "before_cost_per_row": 3 * len(laboratory.object_ids), "after_cost_per_row": chosen["cost_per_row"],
            "acquisitions_after_selection": len(laboratory.acquisitions) - initial_costs["queries"] - 2},
        "experiments": experiments, "boundary_refinements": refinements, "prior_assessments": prior_assessments,
        "discovery": discovery.to_dict(), "structural_prior": record, "episode": episode.to_dict(),
        "costs": {**costs, "compute_cost": preparation.spent + episode.compute_cost},
        "wall_time_seconds": perf_counter() - started}
