"""Infer object roles from anonymous measurements and controlled differences.

Names are opaque identifiers. Moving quantities are detected in before/after
measurements; stable inputs are associated by their measured effects. Position
membership needs mixed interventions, and focus membership needs an informative
response comparison. Ambiguous assignments remain candidates rather than being
filled from field order.
"""

from collections import Counter
from itertools import combinations

from .continuous_state import _linear_fit


def _different(left, right):
    return abs(left - right) > 1e-8 * max(1., abs(left), abs(right))


def _history_keys(records, controls):
    locations = {(row["probe_id"], row["position"]): row for row in records}
    result = []
    for row in records:
        if "history" in row:
            prefix = row["history"]
        else:
            prefix = [locations.get((row["probe_id"], position))
                      for position in range(row["position"])]
            if any(item is None for item in prefix):
                result.append(None)
                continue
            prefix = [item["controls"] for item in prefix]
        result.append(tuple(tuple(point[name] for name in controls) for point in prefix))
    return result


def _pairs(records, controls, fields, histories, budget):
    effects = {}
    for control in controls:
        if not budget.spend(len(records)):
            break
        others = [name for name in controls if name != control]
        frequency = Counter(row["controls"][control] for row in records)
        groups = {}
        for index, row in enumerate(records):
            groups.setdefault(tuple(row["controls"][name] for name in others), []).append(index)
        pairs = []
        for indices in groups.values():
            values = {}
            for index in indices:
                values.setdefault(records[index]["controls"][control], []).append(index)
            if len(values) < 2:
                continue
            # A shared observed baseline makes mixed comparisons invariant to
            # record order without evaluating every pair of control levels.
            ordered = sorted(values, key=lambda value: (-frequency[value], value))
            reference = values[ordered[0]]
            for value in ordered[1:]:
                following = values[value]
                matching = {}
                for left in reference:
                    history = histories[left]
                    if history is None or history in matching:
                        continue
                    right = next((index for index in following if histories[index] == history), None)
                    if right is not None:
                        matching[history] = (left, right)
                # Retain each measured response context. Choosing one arbitrary
                # match can let a reset erase an already purchased history test.
                for left, right in matching.values() or [(reference[0], following[0])]:
                    if not budget.spend(2 * len(fields) + 1):
                        break
                    a, b = records[left], records[right]
                    delta_before = {name: b["before"][name] - a["before"][name] for name in fields}
                    delta_after = {name: b["after"][name] - a["after"][name] for name in fields}
                    before = [name for name in fields if _different(a["before"][name], b["before"][name])]
                    after = [name for name in fields if _different(a["after"][name], b["after"][name])]
                    pairs.append({"left": left, "right": right, "before_fields": before,
                        "after_fields": after, "delta_before": delta_before, "delta_after": delta_after,
                        "target_delta": b["target"] - a["target"] if matching else None})
        effects[control] = {"before_fields": sorted({name for pair in pairs for name in pair["before_fields"]}),
                            "after_fields": sorted({name for pair in pairs for name in pair["after_fields"]}),
                            "pairs": pairs}
    return effects


def _propagate(candidates, quantities):
    """Use measured one-to-one assignments, never an arbitrary permutation."""
    result = {name: set(values) for name, values in candidates.items()}
    if len(result) != len(quantities):
        return result
    while True:
        counts = Counter(next(iter(values)) for values in result.values() if len(values) == 1)
        fixed = {quantity for quantity, count in counts.items() if count == 1}
        changed = False
        for name, values in result.items():
            if len(values) > 1 and values - fixed and values - fixed != values:
                result[name] = values - fixed
                changed = True
        if not changed:
            return result


def _membership_contrasts(rows, controls, histories, effects, excitation, selector, binary, budget):
    """Find selector-dependent intervention effects at matched contexts."""
    evidence = {field: [] for field in binary}
    if selector is None:
        return evidence
    selector_index = controls.index(selector)
    for control in excitation:
        groups = {}
        for pair in effects[control]["pairs"]:
            if pair["target_delta"] is None:
                continue
            left, right = rows[pair["left"]], rows[pair["right"]]
            values = left["controls"][control], right["controls"][control]
            if values[0] == values[1]:
                continue
            history = histories[pair["left"]]
            key = (tuple(sorted(values)),
                tuple(left["controls"][name] for name in controls if name not in (control, selector)),
                tuple(tuple(value for index, value in enumerate(point) if index != selector_index)
                      for point in history))
            delta = pair["target_delta"] * (1 if values[1] > values[0] else -1)
            groups.setdefault(key, []).append((pair, delta))
        for alternatives in groups.values():
            for (first, first_delta), (second, second_delta) in combinations(alternatives, 2):
                if not budget.spend(1) or not _different(first_delta, second_delta):
                    continue
                left, right = rows[first["left"]], rows[second["left"]]
                for field in binary:
                    if left["before"][field] != right["before"][field]:
                        evidence[field].append({"control": control,
                            "pairs": [[first["left"], first["right"]],
                                      [second["left"], second["right"]]],
                            "response_effects": [first_delta, second_delta]})
    return evidence


def _selective_direct_membership(rows, selector, contrasts, drives, quantities):
    """Test whether a selector exposes exactly one independently driven response.

Unlike a regression on correlated before/after quantities, this local-readout
claim must explain both active and inactive selector states. A global response
or an effect on multiple independently driven objects leaves it unresolved.
    """
    evidence, selected, unselected = [], set(), set()
    for contrast in contrasts:
        drive = drives.get(contrast["control"])
        if not drive or drive.get("quantity") is None:
            continue
        pairs = contrast["pairs"]
        states = [rows[pair[0]]["before"][selector] for pair in pairs]
        if set(states) != {0., 1.}:
            continue
        active = contrast["response_effects"][states.index(1.)]
        inactive = contrast["response_effects"][states.index(0.)]
        quantity = drive["quantity"]
        active_effect, inactive_effect = _different(active, 0.), _different(inactive, 0.)
        evidence.append({"quantity": quantity, "control": contrast["control"],
                         "pairs": pairs, "active_effect": active, "inactive_effect": inactive})
        if active_effect and not inactive_effect:
            selected.add(quantity)
        elif inactive_effect and not active_effect:
            unselected.add(quantity)
        else:
            return {"status": "proposed", "quantities": sorted(quantities), "evidence": evidence}
    supported = (len(selected) == 1 and not selected & unselected
                 and selected | unselected == set(quantities))
    return {"status": "supported" if supported else "proposed",
            "quantities": sorted(selected if supported else quantities), "evidence": evidence}


def _background_comparisons(rows, controls, effects, quantity_controls, stable_controls, budget):
    """Compare measured quantity responses across one stable control change."""
    quantities, comparisons = list(quantity_controls), []
    for excited, quantity_control in quantity_controls.items():
        for left, right in combinations(effects.get(quantity_control, {}).get("pairs", []), 2):
            a, b = rows[left["left"]], rows[right["left"]]
            a_end, b_end = rows[left["right"]], rows[right["right"]]
            if sorted((a["controls"][quantity_control], a_end["controls"][quantity_control])) != sorted(
                    (b["controls"][quantity_control], b_end["controls"][quantity_control])):
                continue
            changed = [control for control in controls if control != quantity_control
                       and _different(a["controls"][control], b["controls"][control])]
            if len(changed) != 1 or changed[0] not in stable_controls:
                continue
            denominators = left["delta_before"][excited], right["delta_before"][excited]
            if min(abs(value) for value in denominators) < 1e-10 or not budget.spend(len(quantities)):
                continue
            responses = [{quantity: pair["delta_after"][quantity] / denominator for quantity in quantities}
                         for pair, denominator in zip((left, right), denominators)]
            affected = [quantity for quantity in quantities if _different(
                        responses[0][quantity], responses[1][quantity])]
            comparisons.append({"control": changed[0], "quantity_control": quantity_control,
                "excited_quantity": excited, "affected_quantities": affected,
                "pairs": [[left["left"], left["right"]], [right["left"], right["right"]]],
                "responses": responses})
    return comparisons


def _direct_pairs(rows, pairs, quantities):
    """Equal measured quantities suppress difference-based interactions."""
    selected = []
    for pair in pairs:
        left, right = rows[pair["left"]], rows[pair["right"]]
        values = [left["before"][name] for name in quantities]
        if (values and not _different(min(values), max(values)) and all(
                not _different(left["before"][name], right["before"][name]) for name in quantities)):
            selected.append(pair)
    return selected


def _competing_roles(rows, effects, quantity_controls, stable_controls, comparisons):
    """Keep additive and relational explanations separate until an intervention distinguishes them."""
    quantities = set(quantity_controls)
    candidates, hypotheses, witnesses = {}, [], []
    for control, field in stable_controls.items():
        pairs = effects[control]["pairs"]
        usable = [pair for pair in pairs if abs(pair["delta_before"][field]) > 1e-10]
        slopes = [{name: pair["delta_after"][name] / pair["delta_before"][field]
                   for name in quantities} for pair in usable]
        affected = {name for slope in slopes for name, value in slope.items() if abs(value) > 1e-8}
        additive = (len(slopes) >= 2 and len(affected) == 1 and all(
            not _different(min(slope[name] for slope in slopes), max(slope[name] for slope in slopes))
            for name in quantities))
        tested = [item for item in comparisons if item["control"] == control]
        interacting = [item for item in tested if item["affected_quantities"]]
        covered = {item["excited_quantity"] for item in tested}
        direct = _direct_pairs(rows, pairs, quantities)
        direct_fields = {name for pair in direct for name in quantities if abs(pair["delta_after"][name]) > 1e-8}
        if interacting:
            roles = ["x"]
            kind = ("relational_with_direct_effect" if direct_fields else "relational_only") if direct else None
        elif additive and covered == quantities:
            roles, kind = ["u"], "pure_additive"
        else:
            roles, kind = (["u", "x"] if additive else ["x"]), None
        candidates[field] = {"control": control, "roles": roles,
            "quantity_candidates": sorted(affected if additive else quantities),
            "background_quantities": sorted(covered), "dependency_kind": kind,
            "additive_supported": additive,
            "support_pairs": len(usable), "direct_fields": sorted(direct_fields),
            "background_comparisons": tested}
        for alternative in ("pure_additive", "relational_only", "relational_with_direct_effect"):
            refuted = ((alternative == "pure_additive" and bool(interacting)) or
                (alternative == "relational_only" and bool(direct_fields)) or
                (alternative == "relational_with_direct_effect" and bool(direct) and not direct_fields))
            status = "supported" if alternative == kind else "refuted" if refuted else "proposed"
            hypotheses.append({"hypothesis_id": f"{field}:{alternative}", "field": field,
                "control": control, "kind": alternative, "status": status,
                "background_quantities": sorted(covered), "direct_fields": sorted(direct_fields),
                "counterexamples": ([item["pairs"] for item in interacting[:1]] if alternative == "pure_additive"
                                    else [[[pair["left"], pair["right"]]] for pair in direct[:1]])
                                    if status == "refuted" else []})
        if kind and tested:
            selected = (interacting or tested)[0]
            indices = list(dict.fromkeys(index for pair in selected["pairs"] for index in pair))
            if direct:
                baseline = min(direct, key=lambda pair: sum(pair[side] not in indices for side in ("left", "right")))
                indices = list(dict.fromkeys([*indices, baseline["left"], baseline["right"]]))
            witnesses.append({"field": field, "control": control, "kind": kind,
                "quantity_candidates": candidates[field]["quantity_candidates"],
                "sequence": [dict(rows[index]["controls"]) for index in indices],
                "support_pairs": selected["pairs"], "direct_fields": sorted(direct_fields)})
    return candidates, hypotheses, witnesses


def validate_object_roles(prior, records, budget):
    """Retest previous role definitions on newly purchased transition witnesses.

    A changed target alone does not invalidate an object role. Changes in a
    conditional dependency or the presence of a direct effect do; changing only
    the strength of an otherwise identical dependency does not.
    """
    rows, start = list(records), budget.spent
    result = {"status": "insufficient", "counterexamples": [], "unresolved": [],
              "tested_hypotheses": [], "compute_cost": 0}
    previous = [item for item in prior.get("role_hypotheses", []) if item["status"] == "supported"]
    if not rows or not previous:
        result["unresolved"] = ["no_retained_role_definition_or_new_observation"]
        return result
    controls = prior["raw_control_features"]
    fields = sorted(set.intersection(*(set(row["before"]) & set(row["after"]) for row in rows)))
    quantity_controls = prior["quantity_controls"]
    stable_controls = {item["control"]: field for field, item in prior["role_candidates"].items()}
    histories = _history_keys(rows, controls)
    effects = _pairs(rows, controls, fields, histories, budget)
    available = {control: field for control, field in stable_controls.items() if control in effects}
    comparisons = _background_comparisons(rows, controls, effects, quantity_controls, available, budget)
    current, hypotheses, _ = _competing_roles(rows, effects, quantity_controls, available, comparisons)
    proposed = {item["hypothesis_id"]: item for item in hypotheses}
    for old in previous:
        identifier, field = old["hypothesis_id"], old["field"]
        hypothesis, candidate = proposed.get(identifier), current.get(field)
        if hypothesis is None or candidate is None:
            result["unresolved"].append(identifier)
            continue
        same_members = (candidate["quantity_candidates"] ==
                        prior["role_candidates"][field]["quantity_candidates"])
        additive_confirmed = (old["kind"] == "pure_additive" and candidate["additive_supported"]
            and bool(candidate["background_comparisons"]) and same_members)
        supported = hypothesis["status"] == "supported" or additive_confirmed
        refuted = (hypothesis["status"] == "refuted" or
                   (old["kind"] == "pure_additive" and candidate["additive_supported"] and not same_members))
        if refuted:
            indices = list(dict.fromkeys(index for item in candidate["background_comparisons"][:1]
                                        for pair in item["pairs"] for index in pair))
            result["counterexamples"].append({"field": field, "hypothesis_id": identifier,
                "previous_kind": old["kind"], "current_kind": candidate["dependency_kind"],
                "direct_fields": candidate["direct_fields"],
                "records": [{key: rows[index][key] for key in ("controls", "before", "after")}
                            for index in indices]})
        elif supported:
            result["tested_hypotheses"].append(identifier)
        else:
            result["unresolved"].append(identifier)
    result["status"] = ("counterexample" if result["counterexamples"] else
                        "insufficient" if result["unresolved"] else "supported")
    result["compute_cost"] = budget.spent - start
    return result


def infer_object_model(spec, records, budget, prior=None):
    """Return a complete public schema only when every required role is identified.

    Records have controls, before, after, target, probe_id and position; optional
    history contains earlier control dictionaries. Response differences are used
    only at equal observed prefixes. A prior assignment can persist while current
    candidate sets still permit it, but contradictory interventions are reported.
    """
    rows, start = list(records), budget.spent
    controls = list(spec.feature_bounds)
    fields = sorted(set.intersection(*(set(row["before"]) & set(row["after"]) for row in rows))) if rows else []
    output = {"status": "partial", "schema": None, "complete": False, "groups": [],
        "raw_control_features": list(controls), "raw_sensor_features": fields,
        "dynamic_fields": [], "quantity_controls": {}, "control_effects": {},
        "drive_links": {}, "position_candidates": {}, "focus_candidates": {},
        "role_candidates": {}, "role_hypotheses": [], "membership_hypotheses": [], "dependency_witnesses": [],
        "focus_control": None, "dependencies": [], "contradictions": [],
        "unresolved": [], "support_rows": len(rows), "compute_cost": 0}
    if not rows or not fields or not budget.spend(len(rows) * len(fields)):
        output["unresolved"] = ["insufficient_observation_or_compute_budget"]
        return output
    dynamic = [name for name in fields if any(_different(row["before"][name], row["after"][name]) for row in rows)]
    stable = set(fields) - set(dynamic)
    binary = {name for name in stable if all(row["before"][name] in (0., 1.) for row in rows)}
    output["dynamic_fields"] = dynamic
    histories = _history_keys(rows, controls)
    effects = _pairs(rows, controls, fields, histories, budget)
    output["control_effects"] = effects
    for effect in effects.values():
        effect["after_fields"] = [name for name in effect["after_fields"] if name in dynamic]
    quantities = set(dynamic)
    quantity_controls = {}
    for quantity in dynamic:
        owners = [control for control, effect in effects.items()
                  if effect["before_fields"] == [quantity]]
        if len(owners) == 1:
            quantity_controls[quantity] = owners[0]
    output["quantity_controls"] = quantity_controls
    field_controls = {field: [control for control, effect in effects.items()
                             if field in effect["before_fields"]] for field in stable}
    focus_controls = [control for control, effect in effects.items()
                      if effect["before_fields"] and set(effect["before_fields"]) <= binary
                      and not effect["after_fields"]]
    if len(focus_controls) == 1:
        output["focus_control"] = focus_controls[0]

    stable_controls = {owners[0]: field for field, owners in field_controls.items()
                       if field not in binary and len(owners) == 1
                       and effects[owners[0]]["before_fields"] == [field]}
    comparisons = _background_comparisons(rows, controls, effects, quantity_controls,
                                         stable_controls, budget)
    roles, hypotheses, witnesses = _competing_roles(rows, effects, quantity_controls, stable_controls, comparisons)
    output.update(role_candidates=roles, role_hypotheses=hypotheses, dependency_witnesses=witnesses)
    drives = {item["control"]: {"field": field, "quantity_candidates": item["quantity_candidates"],
        "quantity": item["quantity_candidates"][0] if item["dependency_kind"] == "pure_additive" else None,
        "support_pairs": item["support_pairs"]}
        for field, item in roles.items() if "u" in item["roles"]}
    output["drive_links"] = drives
    positions = {field: {"control": item["control"], "quantities": list(dynamic), "support_pairs": 0}
                 for field, item in roles.items() if item["roles"] == ["x"]}
    position_by_control = {item["control"]: field for field, item in positions.items()}
    dependencies = []
    for comparison in comparisons:
        control, excited = comparison["control"], comparison["excited_quantity"]
        if control not in position_by_control:
            continue
        affected = set(comparison["affected_quantities"])
        others = affected - {excited}
        if not others:
            continue
        candidates = {excited} if len(others) >= 2 else {excited, *others}
        field = position_by_control[control]
        positions[field]["quantities"] = sorted(set(positions[field]["quantities"]) & candidates)
        positions[field]["support_pairs"] += 1
        dependencies.append({**comparison, "candidate_quantities": sorted(candidates)})
    output["dependencies"] = dependencies
    for field, candidate in positions.items():
        role = roles[field]
        direct = set(role["direct_fields"]) & quantities
        if role["dependency_kind"] == "relational_with_direct_effect" and len(direct) == 1:
            # Local direct effects provide an anchor even when a two-location
            # distance is symmetric. This is a revisable locality hypothesis,
            # distinct from the purely relational membership explanation.
            candidate["quantities"] = sorted(set(candidate["quantities"]) & direct)
            candidate["direct_anchor"] = {"kind": "local_direct_effect", "status": "supported",
                "quantities": sorted(direct), "control": candidate["control"],
                "witness_sequences": [witness["sequence"] for witness in witnesses if witness["field"] == field]}
            output["membership_hypotheses"].append({"field": field, "role": "x", **candidate["direct_anchor"]})
    propagated = _propagate({name: item["quantities"] for name, item in positions.items()}, quantities)
    for name, values in propagated.items():
        positions[name]["quantities"] = sorted(values)
    output["position_candidates"] = positions

    drive_fields = {}
    for quantity in dynamic:
        linked = [link["field"] for link in drives.values() if link["quantity"] == quantity]
        if len(linked) == 1:
            drive_fields[quantity] = linked[0]
    excitation = {control: quantity for quantity, control in quantity_controls.items()}
    excitation.update({control: link["quantity"] for control, link in drives.items() if link["quantity"]})
    contrasts = _membership_contrasts(rows, controls, histories, effects, excitation,
                                     output["focus_control"], binary, budget)
    focus_candidates = {}
    for selector in sorted(binary):
        samples, coverage = [], set()
        for control, quantity in excitation.items():
            for pair in effects[control]["pairs"]:
                if pair["target_delta"] is None:
                    continue
                left, right = rows[pair["left"]], rows[pair["right"]]
                if left["before"][selector] == right["before"][selector] == 1.:
                    samples.append(pair)
                    coverage.add(quantity)
        accepted, losses, response_contexts = set(dynamic), {}, []
        informative_depths = sorted({len(histories[item["pairs"][0][0]])
                                    for item in contrasts[selector]})
        for depth in informative_depths:
            contextual = [sample for sample in samples if len(histories[sample["left"]]) == depth]
            if not contextual or not dynamic:
                continue
            targets = [sample["target_delta"] for sample in contextual]
            context_losses = {}
            tolerance = 1e-10 * max(1., sum(value * value for value in targets) / len(targets))
            for quantity in dynamic:
                columns = [[sample[side].get(field, 0.) for sample in contextual]
                           for side, field in (("delta_after", quantity), ("delta_before", quantity),
                               ("delta_before", drive_fields.get(quantity, "")))]
                if not budget.spend(2 * (len(columns) + 2) * len(contextual)):
                    break
                # Reversing a purchased contrast reverses both its measured
                # inputs and response. Fit both orientations of those same
                # pairs, so a free intercept cannot explain missing effects.
                # Incomplete coverage can rule out a candidate already; any
                # observationally equivalent assignments remain in the set.
                coefficients = _linear_fit([[*column, *(-value for value in column)]
                                             for column in columns],
                                           [*targets, *(-value for value in targets)])
                predictions = [sum(value * column[index]
                    for value, column in zip(coefficients[1:], columns)) for index in range(len(contextual))]
                context_losses[quantity] = sum((value - target) ** 2 for value, target in zip(predictions, targets)) / len(targets)
            supported = {quantity for quantity, error in context_losses.items() if error <= tolerance}
            response_contexts.append({"history_rows": depth, "support_pairs": len(contextual),
                "fit_losses": context_losses, "quantities": sorted(supported),
                "fully_assessed": len(context_losses) == len(dynamic)})
            if len(context_losses) == len(dynamic) and supported:
                accepted &= supported
                for quantity, error in context_losses.items():
                    losses[quantity] = max(losses.get(quantity, 0.), error)
        direct_membership = _selective_direct_membership(rows, selector, contrasts[selector], drives, quantities)
        if direct_membership["status"] == "supported":
            accepted &= set(direct_membership["quantities"])
        output["membership_hypotheses"].append({"field": selector, "role": "f",
            "kind": "selective_direct_response", **direct_membership})
        focus_candidates[selector] = {"quantities": sorted(accepted), "support_pairs": len(samples),
            "covered_quantities": sorted(coverage), "fit_losses": losses,
            "response_contexts": response_contexts,
            "distinguishing_contrasts": contrasts[selector], "direct_response": direct_membership}
    propagated = _propagate({name: item["quantities"] for name, item in focus_candidates.items()}, quantities)
    for name, values in propagated.items():
        focus_candidates[name]["quantities"] = sorted(values)
    output["focus_candidates"] = focus_candidates

    if prior:
        previous_roles = {item["hypothesis_id"]: item for item in prior.get("role_hypotheses", [])
                          if item["status"] == "supported"}
        for hypothesis in hypotheses:
            if hypothesis["status"] == "refuted" and hypothesis["hypothesis_id"] in previous_roles:
                output["contradictions"].append({"field": hypothesis["field"],
                    "role_definition": hypothesis["kind"], "counterexamples": hypothesis["counterexamples"]})
        for group in prior.get("groups", []):
            quantity = group["q"]
            for role, candidates in (("x", positions), ("f", focus_candidates)):
                field = group.get(role)
                if field not in candidates:
                    continue
                previous_anchor = prior.get("position_candidates", {}).get(field, {}).get("direct_anchor")
                if (role == "x" and previous_anchor and not candidates[field].get("direct_anchor")
                        and roles[field]["dependency_kind"] == "relational_only"):
                    output["contradictions"].append({"field": field, "role": role,
                        "previous_quantity": quantity, "candidate_quantities": candidates[field]["quantities"],
                        "withdrawn_membership_basis": "local_direct_effect"})
                    continue
                if quantity in candidates[field]["quantities"]:
                    candidates[field]["quantities"] = [quantity]
                else:
                    output["contradictions"].append({"field": field, "role": role,
                        "previous_quantity": quantity, "candidate_quantities": candidates[field]["quantities"]})
    for index, quantity in enumerate(dynamic):
        group = {"object_id": f"induced_{index}", "q": quantity, "controls": {}}
        if quantity in quantity_controls:
            group["controls"]["q"] = quantity_controls[quantity]
        if quantity in drive_fields:
            group["u"] = drive_fields[quantity]
            group["controls"]["u"] = next(control for control, link in drives.items()
                                             if link["field"] == group["u"])
        for role, candidates in (("x", positions), ("f", focus_candidates)):
            matched = [field for field, item in candidates.items() if item["quantities"] == [quantity]]
            if len(matched) == 1:
                group[role] = matched[0]
                if role == "x":
                    group["controls"][role] = candidates[matched[0]]["control"]
        output["groups"].append(group)
    complete = (bool(dynamic) and len(binary) == len(dynamic) and output["focus_control"] is not None
                and all(set(group) >= {"q", "u", "x", "f"} and set(group["controls"]) >= {"q", "u", "x"}
                        for group in output["groups"]))
    if complete:
        groups = output["groups"]
        output["schema"] = {"objects": [group["object_id"] for group in groups],
            "fields": {group["object_id"]: {role: group[role] for role in ("q", "x", "u", "f")} for group in groups},
            "members": {group["object_id"]: [group["object_id"]] for group in groups},
            "controls": {group["object_id"]: group["controls"] for group in groups},
            "focus_control": output["focus_control"]}
        output["status"], output["complete"] = "identified", True
    else:
        output["unresolved"] = [f"{group['object_id']}:{role}" for group in output["groups"]
            for role in ("q_control", "u", "x", "f")
            if ("q" not in group["controls"] if role == "q_control" else role not in group)]
        if output["focus_control"] is None:
            output["unresolved"].append("focus_control")
    output["compute_cost"] = budget.spent - start
    return output
