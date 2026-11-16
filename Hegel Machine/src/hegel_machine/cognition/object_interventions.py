"""Choose real interventions that distinguish anonymous object dependencies.

Only public control domains, purchased records and competing dependency claims
enter this planner. It returns a probe; the research loop purchases its rows and
the object inducer interprets their results.
"""

from .models import Probe


def _missing_states(points, records):
    observed = [record["controls"] for record in records]
    result = []
    for point in points:
        if point not in observed and point not in result:
            result.append(point)
    return result


def _observed_response(sequence, records):
    """Response contrasts need the same complete prefix, including resets."""
    final = len(sequence) - 1
    for record in records:
        if record["position"] != final or record["controls"] != sequence[-1]:
            continue
        prefix = sorted((other for other in records if other["probe_id"] == record["probe_id"]
                         and other["position"] < final), key=lambda other: other["position"])
        if [other["controls"] for other in prefix] == list(sequence[:-1]):
            return True
    return False


def _position_square(spec, baseline, moved, excitation):
    changed = max(spec.feature_bounds[excitation], key=lambda level: abs(level - baseline[excitation]))
    return [dict(baseline), {**baseline, excitation: changed}, dict(moved), {**moved, excitation: changed}]


def _represented_position_owners(schemas, field):
    return [next(group["q"] for group in schema["fields"].values() if group["x"] == field)
            for schema in schemas]


def plan_paid_position_contrast(spec, model, records, budget, *, schemas=None, remaining_rows=None):
    """Quote one reset factorial from paid geometry and competing position owners.

    The old measured interaction rule supplies conditional, qualitative predictions.
    A finite list of response assignments directs effort, never restricts the
    admissible owners of the measured field.
    """
    from .object_induction import _different, position_owner_predictions

    start = budget.spent
    def result(status, **data):
        return {"status": status, "purpose": "distinguish_position_dependency",
                "compute_cost": budget.spent - start, **data}

    positions = model.get("position_candidates", {})
    quantity_controls = model.get("quantity_controls", {})
    quantities = model.get("dynamic_fields", ())
    if len(positions) < 3 or len(quantities) < 3 or not quantity_controls:
        return result("insufficient_position_distinction")
    if schemas is None:
        schemas = model.get("schema_alternatives", {}).get("schemas", [])
    if not budget.spend(len(records)):
        return result("position_planning_compute_exhausted")
    resets = [r for r in records if r["position"] == 0 and not r.get("history")]
    background = None
    for row in resets:
        if not budget.spend(len(positions)):
            return result("position_planning_compute_exhausted")
        if all(field in row["before"] for field in positions):
            values = sorted(row["before"][field] for field in positions)
            if all(_different(a, b) for a, b in zip(values, values[1:])):
                background = row
                break
    if background is None:
        return result("no_paid_separated_reset")

    # Minimize the worst number of competing assignments surviving an
    # informative single-foreign-quantity change. A flat experiment always
    # retains all owners. Ties keep the measured field / excitation order.
    best, split = None, None
    for field, candidate in positions.items():
        domain = candidate["quantities"]
        if len(domain) < 2:
            continue
        represented = _represented_position_owners(schemas, field)
        if any(owner not in domain for owner in represented):
            raise ValueError("A supplied assignment is inconsistent with the current position domain")
        owners = represented or domain
        if len(set(owners)) < 2:
            continue
        for excited in quantity_controls:
            if not budget.spend(1 + len(owners) * len(quantities)):
                return result("position_planning_compute_exhausted")
            predictions = position_owner_predictions(quantities, excited, domain)
            survivors = [sum(other in predictions[q]["allowed_foreign_quantities"] for q in owners)
                         for other in quantities if other != excited]
            # Zero surviving local rivals is an incompatibility, not a split.
            if not any(0 < count < len(owners) for count in survivors):
                continue
            criterion = (max(survivors), sum(n*n for n in survivors))
            if split is None or criterion < split:
                split = criterion
                best = field, excited, represented, predictions
    if best is None:
        return result("no_competing_position_predictions")
    field, quantity, represented, predictions = best
    position = positions[field]["control"]
    excitation = quantity_controls[quantity]
    base = background["controls"]
    level = max(spec.feature_bounds[position], key=lambda value: abs(value - base[position]))
    corners = _position_square(spec, base, {**base, position: level}, excitation)
    if corners[0] == corners[1] or corners[0] == corners[2]:
        return result("no_distinct_control_levels")
    missing = [i for i, corner in enumerate(corners) if not any(row["controls"] == corner for row in resets)]
    status = ("complete_paid_position_experiment" if not missing else
              "complete_position_experiment_unfunded" if remaining_rows is not None and len(missing) > remaining_rows
              else "proposed_position_experiment")
    return result(status, field=field, position_control=position, excited_quantity=quantity,
                  quantity_control=excitation, candidate_owners=list(positions[field]["quantities"]),
                  alternative_owners=represented, owner_predictions=predictions, matched_prefix_rows=0,
                  baseline_ref={"probe_id": background["probe_id"], "position": 0},
                  corners=corners, missing_corner_indices=missing, required_rows=len(missing))


def plan_position_owner_followup(spec, model, records, budget, *, previous_plan, previous_result,
                                 schemas, remaining_rows=2):
    """Choose a remaining-owner test on the original two measured reset backgrounds.

    Each owner must permit a distinct nonempty set of cross-quantity effects.
    A flat response leaves both alive. Reuse a complete paid square first,
    then the cheapest missing endpoints, in measured excitation order.
    """
    from .object_induction import position_owner_predictions

    start = budget.spent
    field = previous_plan["field"]

    def result(status, **data):
        return {"status": status, "purpose": "distinguish_position_dependency",
                "field": field, "compute_cost": budget.spent - start, **data}

    position = model["position_candidates"][field]
    owners = list(position["quantities"])
    if (previous_result["status"] != "position_domain_narrowed"
            or previous_result["field"] != field or previous_result["source_model_id"] != model.get("model_id")
            or set(previous_result["retained_owners"]) != set(owners) or len(owners) != 2
            or position["control"] != previous_plan["position_control"]):
        return result("no_unresolved_measured_owner_pair")
    represented = _represented_position_owners(schemas, field)
    if set(represented) != set(owners):
        return result("incomplete_position_owner_representation")

    base, _, moved, _ = previous_plan["corners"]
    ref0, _, refx, _ = previous_result["corner_refs"]
    if ref0 != previous_plan["baseline_ref"]:
        return result("unavailable_paid_position_background")
    if not budget.spend(len(records)):
        return result("position_planning_compute_exhausted")
    resets = [row for row in records if row["position"] == 0 and not row.get("history")]
    if not budget.spend(len(resets) * (len(base) + 1)):
        return result("position_planning_compute_exhausted")
    by_id = {row["probe_id"]: row for row in resets}
    anchors = [by_id.get(ref["probe_id"]) for ref in (ref0, refx)]
    if (not all(anchors) or [row["controls"] for row in anchors] != [base, moved]):
        return result("unavailable_paid_position_background")

    def point_key(point):
        return tuple(point[c] for c in base)

    by_controls = {point_key(row["controls"]): row for row in resets}
    quantities = model["dynamic_fields"]
    best = None
    for excited, control in model["quantity_controls"].items():
        if not budget.spend(5 + len(owners) * len(quantities)):
            return result("position_planning_compute_exhausted")
        forecasts = position_owner_predictions(quantities, excited, owners)
        left, right = (set(forecasts[q]["allowed_foreign_quantities"]) for q in owners)
        if not left or not right or left & right:
            continue
        corners = _position_square(spec, base, moved, control)
        if corners[0] == corners[1] or corners == previous_plan["corners"]:
            continue
        paid = [anchors[0], by_controls.get(point_key(corners[1])),
                anchors[1], by_controls.get(point_key(corners[3]))]
        missing = [i for i, row in enumerate(paid) if row is None]
        if best is None or len(missing) < len(best["missing_corner_indices"]):
            best = {"excited_quantity": excited, "quantity_control": control,
                    "owner_predictions": forecasts, "corners": corners,
                    "missing_corner_indices": missing, "required_rows": len(missing),
                    "paid_corner_refs": [None if row is None else {"probe_id": row["probe_id"], "position": 0}
                                         for row in paid]}
    if best is None:
        return result("no_disjoint_position_predictions")
    needed = best["required_rows"]
    status = ("complete_paid_position_experiment" if needed == 0 else
              "complete_position_experiment_unfunded" if needed > remaining_rows else
              "proposed_position_experiment")
    return result(status, position_control=position["control"], candidate_owners=owners,
                  alternative_owners=represented, matched_prefix_rows=0, baseline_ref=ref0,
                  moved_ref=refx, **best)


def plan_generalized_position_followup(spec, model, records, budget, *, previous_plan,
                                       previous_result, schemas=None, remaining_rows=2):
    """Generalize position-owner follow-up from ``k == 2`` to any ``2 <= k <= N`` owners.

    When ``len(owners) == 2``, delegates directly to ``plan_position_owner_followup``
    so historical ``k=2`` receipts remain byte-identical. For ``k >= 3`` surviving
    owners on the shared reset backgrounds ``(base, moved)``, chooses the untried
    excitation minimizing worst-case surviving owner count (breaking ties by
    missing corners and partition entropy) under ``SearchBudget``.
    """
    from .object_induction import position_owner_predictions

    field = previous_plan["field"]
    position = model.get("position_candidates", {}).get(field, {})
    owners = list(position.get("quantities", ()))
    if len(owners) == 2 and schemas is not None:
        return plan_position_owner_followup(
            spec, model, records, budget,
            previous_plan=previous_plan, previous_result=previous_result,
            schemas=schemas, remaining_rows=remaining_rows,
        )

    start = budget.spent

    def result(status, **data):
        return {"status": status, "purpose": "distinguish_position_dependency",
                "field": field, "compute_cost": budget.spent - start, **data}

    if (previous_result.get("status") != "position_domain_narrowed"
            or previous_result.get("field") != field
            or previous_result.get("source_model_id") != model.get("model_id")
            or set(previous_result.get("retained_owners", ())) != set(owners)
            or len(owners) < 2
            or position.get("control") != previous_plan.get("position_control")):
        return result("no_unresolved_measured_owner_set")
    represented = _represented_position_owners(schemas, field) if schemas else list(owners)
    if schemas and not set(represented) <= set(owners):
        return result("incomplete_position_owner_representation")

    base, _, moved, _ = previous_plan["corners"]
    ref0, _, refx, _ = previous_result["corner_refs"]
    if ref0 != previous_plan["baseline_ref"]:
        return result("unavailable_paid_position_background")
    if not budget.spend(len(records)):
        return result("position_planning_compute_exhausted")
    resets = [row for row in records if row["position"] == 0 and not row.get("history")]
    if not budget.spend(len(resets) * (len(base) + 1)):
        return result("position_planning_compute_exhausted")
    by_id = {row["probe_id"]: row for row in resets}
    anchors = [by_id.get(ref["probe_id"]) for ref in (ref0, refx)]
    if not all(anchors) or [row["controls"] for row in anchors] != [base, moved]:
        return result("unavailable_paid_position_background")

    def point_key(point):
        return tuple(point[c] for c in base)

    by_controls = {point_key(row["controls"]): row for row in resets}
    quantities = model["dynamic_fields"]
    best, best_key = None, None
    for excited, control in model["quantity_controls"].items():
        if not budget.spend(5 + len(owners) * len(quantities)):
            return result("position_planning_compute_exhausted")
        forecasts = position_owner_predictions(quantities, excited, owners)
        survivors = [
            sum(other in forecasts[q]["allowed_foreign_quantities"] for q in owners)
            for other in quantities if other != excited
        ]
        if not any(0 < count < len(owners) for count in survivors):
            continue
        corners = _position_square(spec, base, moved, control)
        if corners[0] == corners[1] or corners == previous_plan["corners"]:
            continue
        paid = [anchors[0], by_controls.get(point_key(corners[1])),
                anchors[1], by_controls.get(point_key(corners[3]))]
        missing = [i for i, row in enumerate(paid) if row is None]
        criterion = (max(survivors), len(missing), sum(n * n for n in survivors))
        if best_key is None or criterion < best_key:
            best_key = criterion
            best = {"excited_quantity": excited, "quantity_control": control,
                    "owner_predictions": forecasts, "corners": corners,
                    "missing_corner_indices": missing, "required_rows": len(missing),
                    "paid_corner_refs": [None if row is None else {"probe_id": row["probe_id"], "position": 0}
                                         for row in paid]}
    if best is None:
        return result("no_splitting_position_predictions")
    needed = best["required_rows"]
    status = ("complete_paid_position_experiment" if needed == 0 else
              "complete_position_experiment_unfunded" if remaining_rows is not None and needed > remaining_rows
              else "proposed_position_experiment")
    return result(status, position_control=position["control"], candidate_owners=owners,
                  alternative_owners=represented or owners, matched_prefix_rows=0,
                  baseline_ref=ref0, moved_ref=refx, **best)


def propose_object_probe(spec, model, records, serial, *, methods=None, remaining_rows=None):
    """Return the next unfunded distinction as ``(Probe | None, metadata)``.

    A centered single-control scan supplies causal sensor bindings. Thereafter
    only unresolved position and membership assignments motivate experiments.
    Position questions reuse already acquired corners of a two-control design.
    Membership questions use reset or identical-prefix response comparisons.
    Nothing here assigns a role from a control name, order or domain width.
    """
    bounds = spec.feature_bounds
    center = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    model = model or {}

    def proposal(points, purpose, **details):
        if not points:
            return None
        limit = spec.max_probe_length
        if remaining_rows is not None:
            limit = min(limit, remaining_rows)
        # Sensor contrasts can accumulate across purchased setups. A response
        # contrast needs its entire matched prefix: truncating it would ask a
        # different question and could turn a history effect into a silent reset.
        if limit <= 0 or (purpose == "distinguish_membership_dependency" and len(points) > limit):
            return None, {"purpose": purpose, "observation_rows": 0,
                "required_rows": len(points), **details}
        sequence = tuple(points[:limit])
        return Probe(f"object_intervention:{serial}", sequence), {
            "purpose": purpose, "observation_rows": len(sequence), **details}

    if model.get("complete"):
        return None, {"purpose": "object_dependencies_resolved"}

    # The common context is deliberately uninformative about object labels.
    # Independent before/after responses supply the role candidates instead.
    single = [center]
    single.extend({**center, control: value} for control, interval in bounds.items() for value in interval)
    missing = _missing_states(single, records)
    if missing:
        return proposal(missing, "isolate_control_effects",
                        controls=[name for name in bounds if any(point[name] != center[name] for point in missing)])

    quantities = model.get("quantity_controls", {})
    dynamic = model.get("dynamic_fields", [])
    positions = model.get("position_candidates", {})
    role_questions, points = [], []
    for field, candidate in model.get("role_candidates", {}).items():
        if candidate.get("dependency_kind") is not None:
            continue
        control = candidate["control"]
        tested = set(candidate.get("background_quantities", []))
        for quantity, excitation in quantities.items():
            if quantity in tested:
                continue
            moved = {**center, control: bounds[control][1]}
            excited = {**center, excitation: bounds[excitation][1]}
            joint = {**moved, excitation: bounds[excitation][1]}
            missing = _missing_states([center, moved, excited, joint], records)
            points.extend(point for point in missing if point not in points)
            if missing:
                role_questions.append({"field": field, "control": control,
                    "quantity": quantity, "quantity_control": excitation,
                    "competing_roles": candidate["roles"]})
    if points:
        return proposal(points, "distinguish_role_definition", questions=role_questions,
                        competing_explanations=["pure_additive", "relational_only", "relational_with_direct_effect"])
    # With two exchange sites, the two location assignments describe the same
    # distance. More repetitions cannot distinguish that symmetry.
    if len(dynamic) >= 3:
        for field, candidate in positions.items():
            if len(candidate["quantities"]) <= 1:
                continue
            control = candidate["control"]
            order = [name for name in candidate["quantities"] if name in quantities]
            order.extend(name for name in quantities if name not in order)
            for quantity in order:
                excitation = quantities[quantity]
                moved = {**center, control: bounds[control][1]}
                excited = {**center, excitation: bounds[excitation][1]}
                joint = {**moved, excitation: bounds[excitation][1]}
                missing = _missing_states([center, moved, excited, joint], records)
                if missing:
                    return proposal(missing, "distinguish_position_dependency", field=field,
                        position_control=control, quantity_control=excitation,
                        excited_quantity=quantity, competing_quantities=list(candidate["quantities"]))

    focus = model.get("focus_candidates", {})
    unresolved = {field: candidate for field, candidate in focus.items()
                  if len(candidate["quantities"]) > 1}
    selector = model.get("focus_control")
    if not unresolved or selector is None:
        return None, {"purpose": "remaining_dependencies_not_distinguished"}

    selected_values = {}
    for field in focus:
        active = next((record for record in records if record["before"].get(field) == 1.), None)
        if active is not None:
            selected_values[field] = active["controls"][selector]
    if len(selected_values) < len(focus):
        low, high = bounds[selector]
        points = [{**center, selector: low + (high - low) * (index + .5) / len(focus)}
                  for index in range(len(focus))]
        missing = _missing_states(points, records)
        if missing:
            return proposal(missing, "observe_missing_membership_states", selector_control=selector,
                            unresolved_fields=[field for field in focus if field not in selected_values])

    if methods is not None:
        return methods.propose(spec, model, records, serial, selected_values, remaining_rows=remaining_rows)

    drives = {link["quantity"]: control for control, link in model.get("drive_links", {}).items()
              if link.get("quantity") is not None}
    for role, controls, prefix_length in (("direct_response", drives, 0),
                                          ("transition_response", quantities, 1)):
        for quantity, control in controls.items():
            if not any(quantity in candidate["quantities"] or
                       quantity not in candidate.get("covered_quantities", [])
                       for candidate in unresolved.values()):
                continue
            # Compare a control's effect across selector states. A total
            # readout gives equal effects and therefore no member assignment.
            for field, value in selected_values.items():
                baseline = {**center, selector: value}
                changed = {**baseline, control: bounds[control][1]}
                prefix = (baseline,) if prefix_length else ()
                for sequence in ((*prefix, baseline), (*prefix, changed)):
                    if not _observed_response(sequence, records):
                        return proposal(sequence, "distinguish_membership_dependency",
                            response_question=role, selector_control=selector,
                            selected_field=field, intervention_control=control,
                            quantity_candidate=quantity, matched_prefix_rows=prefix_length,
                            competing_fields=[name for name, candidate in unresolved.items()
                                              if quantity in candidate["quantities"]])

    return None, {"purpose": "response_does_not_distinguish_remaining_memberships",
                  "unresolved_fields": list(unresolved)}
