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


def propose_object_probe(spec, model, records, serial):
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
        sequence = tuple(points[:spec.max_probe_length])
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
