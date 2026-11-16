"""Keep evidence-permitted object assignments as executable alternatives.

The object inducer may identify a role without identifying which quantity owns
it.  A finite set of one-to-one assignments can still support a shared law.
Enumerating those assignments never turns a partial object model into a unique
schema; consumers must retain the alternatives and establish program equivalence
separately.
"""

from copy import deepcopy


def _assignment_inputs(model):
    """Only fields read by the ordered assignment search, including controls."""
    return {
        "groups": [{key: deepcopy(group[key]) for key in
                    ("object_id", "q", "u", "f", "x", "controls") if key in group}
                   for group in model.get("groups", [])],
        "focus_control": model.get("focus_control"),
        "positions": {name: [item.get("control"), sorted(item.get("quantities", ()))]
                      for name, item in model.get("position_candidates", {}).items()},
        "foci": {name: sorted(item.get("quantities", ()))
                 for name, item in model.get("focus_candidates", {}).items()},
    }


def restore_paid_schema_alternatives(model, returned_parent, actions):
    """Attach the original ordinary enumeration domain to an old restored model.

    The separately returned parent and its actual induction action establish
    the producer. Callers supply a mutable recovery view, leaving both original
    artifacts intact. Unverified legacy results use fresh enumeration.
    """
    paid = model.get("schema_alternatives", {})
    identity = model.get("model_id")
    if (not identity or not paid.get("schemas") or paid.get("source_model_id") != identity
            or returned_parent.get("model_id") != identity
            or paid != returned_parent.get("schema_alternatives")
            or model.get("reused") is not False or _assignment_inputs(model) != _assignment_inputs(returned_parent)):
        return False
    if not any(action.get("action") == "retain_object_ambiguity"
               and action.get("object_model_id") == identity
               and action.get("rows") == returned_parent.get("acquisition_rows") == model.get("acquisition_rows")
               and action.get("groups") == model["groups"]
               and action.get("unresolved") == model.get("unresolved") == paid.get("unresolved")
               for action in actions):
        return False
    model["schema_alternatives"] = {**paid, "ordinary_assignment_inputs": _assignment_inputs(returned_parent)}
    return True


def _position_owner(schema, field):
    return next(item["q"] for item in schema["fields"].values() if item["x"] == field)


def enumerate_object_schemas(model, budget=None, *, max_schemas=16, preferred_schemas=()):
    """Enumerate complete schemas allowed by the current membership evidence.

Quantity and drive bindings must already be identified.  Position and focus
memberships range only over the inducer's candidates, respecting established
assignments. ``exhaustive`` refers to this evidence-permitted finite set, not to
    every possible ontology. A truncated set cannot authorize an equivalence claim.
    Saved schemas receive the first slots only when their bindings remain allowed
    by the current evidence; they do not identify the current object model.
"""
    if max_schemas < 1:
        raise ValueError("max_schemas must be positive")
    start = budget.spent if budget is not None else 0
    groups = model.get("groups", [])
    result = {"schemas": [], "exhaustive": False, "status": "incomplete_roles",
              "unresolved": list(model.get("unresolved", [])),
              "source_model_id": model.get("model_id"), "compute_cost": 0}
    contrast = model.get("position_assignment_contrast", {})
    field_of_interest = contrast.get("field")
    if field_of_interest is None:
        # Old returns kept the four measured row references and control
        # directly on the dependency. Recover its field on ordinary replay.
        last_square = next((dep for dep in reversed(model.get("dependencies", ()))
                            if len(dep.get("corner_refs", ())) == 4 and "candidate_quantities" in dep), None)
        if last_square:
            field_of_interest = next((name for name, item in model.get("position_candidates", {}).items()
                                      if item.get("control") == last_square["control"]
                                      and len(item["quantities"]) == 2), None)
    ordinary = bool(result["source_model_id"]) and not preferred_schemas
    if ordinary:
        inputs = _assignment_inputs(model)
        result["ordinary_assignment_inputs"] = inputs
        paid = model.get("schema_alternatives", {})
        prefix = paid.get("schemas", [])
        if (paid.get("source_model_id") == result["source_model_id"]
                and paid.get("ordinary_assignment_inputs") == inputs and prefix
                and paid.get("position_owner_coverage", {}).get("field") == field_of_interest
                and (len(prefix) >= max_schemas or paid["exhaustive"])):
            coverage = paid.get("position_owner_coverage")
            if coverage:
                selected = prefix[:max_schemas]
                result["position_owner_coverage"] = {**coverage,
                    "represented_owners": list(dict.fromkeys(_position_owner(s, field_of_interest) for s in selected)),
                    "pairs": [pair for pair in coverage["pairs"] if max(pair["indices"]) < len(selected)]}
            return {**result, "status": "enumerated", "schemas": deepcopy(prefix[:max_schemas]),
                    "exhaustive": paid["exhaustive"] and len(prefix) <= max_schemas,
                    "reused_schema_count": min(max_schemas, len(prefix))}
    if not groups or model.get("focus_control") is None or any(
            not {"q", "u"} <= set(group)
            or not {"q", "u"} <= set(group.get("controls", {})) for group in groups):
        return result

    quantities = {group["q"] for group in groups}
    domains, controls = {}, {}
    for role, candidate_key in (("x", "position_candidates"), ("f", "focus_candidates")):
        candidates = deepcopy(model.get(candidate_key, {}))
        for group in groups:
            if role in group:
                field = group[role]
                if field not in candidates:
                    candidates[field] = {"quantities": [group["q"]]}
                    if role == "x" and "x" in group["controls"]:
                        candidates[field]["control"] = group["controls"]["x"]
                else:
                    candidates[field]["quantities"] = [quantity for quantity in
                        candidates[field]["quantities"] if quantity == group["q"]]
        if len(candidates) != len(groups) or (role == "x" and any(
                "control" not in item for item in candidates.values())):
            return result
        for field, candidate in candidates.items():
            domains[role, field] = quantities.intersection(candidate["quantities"])
            if role == "x":
                controls[field] = candidate["control"]

    owners = sorted(domains.get(("x", field_of_interest), ()))
    contrast_active = len(owners) == 2
    paired, represented = [], []
    exhausted_budget = False

    def schema_from_assignment(assigned):
        fields = {group["object_id"]: {"q": group["q"], "u": group["u"],
            "x": assigned["x", group["q"]], "f": assigned["f", group["q"]]}
            for group in groups}
        return {"objects": [group["object_id"] for group in groups], "fields": fields,
            "members": {group["object_id"]: [group["object_id"]] for group in groups},
            "controls": {group["object_id"]: {"q": group["controls"]["q"],
                "u": group["controls"]["u"], "x": controls[fields[group["object_id"]]["x"]]}
                for group in groups}, "focus_control": model["focus_control"]}

    def remember(assigned, candidate=None, *, reserve_other=False):
        nonlocal exhausted_budget
        if candidate is None:
            candidate = schema_from_assignment(assigned)
        selected = result["schemas"]
        if candidate in selected or len(selected) == max_schemas:
            return
        if not contrast_active:
            selected.append(candidate)
            return
        owner = _position_owner(candidate, field_of_interest)
        if reserve_other and len(selected) == max_schemas - 1 and represented == [owner]:
            return
        selected.append(candidate)
        if owner not in represented:
            represented.append(owner)
        # The reciprocal x swap keeps both bijections, every focus, and all
        # other bindings fixed. The displaced field must allow its new owner.
        if len(selected) >= max_schemas:
            return
        other = next(q for q in owners if q != owner)
        displaced = assigned["x", other]
        if budget is not None and not budget.spend(2):
            exhausted_budget = True
            return
        if owner not in domains["x", displaced]:
            return
        swapped = {**assigned, ("x", owner): displaced, ("x", other): field_of_interest}
        mate = schema_from_assignment(swapped)
        if mate not in selected:
            paired.append({"indices": [len(selected)-1, len(selected)], "owners": [owner, other]})
            selected.append(mate)
            if other not in represented:
                represented.append(other)

    def seeds():
        yield from preferred_schemas
        if contrast_active:
            for seed in contrast.get("paid_anchors", ()):
                if {q for q, _, _ in seed} == quantities and all(x in controls for _, x, _ in seed):
                    yield schema_from_assignment({(role, q): value for q, x, f in seed
                                                  for role, value in (("x", x), ("f", f))})

    seen_preferred = []
    for schema in seeds():
        if schema in seen_preferred:
            continue
        seen_preferred.append(schema)
        # One current-role membership check per object, including rejected
        # proposals. This is not a fitted response or a new role confirmation.
        if budget is not None and not budget.spend(len(groups)):
            exhausted_budget = True
            break
        fields, prior_controls = schema.get("fields", {}), schema.get("controls", {})
        by_quantity = {item.get("q"): (obj, item) for obj, item in fields.items()}
        if (schema.get("focus_control") != model["focus_control"]
                or len(fields) != len(groups) or set(by_quantity) != quantities):
            continue
        assigned, valid = {}, True
        for group in groups:
            obj, prior = by_quantity[group["q"]]
            prior_control = prior_controls.get(obj, {})
            if (prior.get("u") != group["u"]
                    or any(prior_control.get(role) != group["controls"][role] for role in ("q", "u"))):
                valid = False
                break
            for role in ("x", "f"):
                field = prior.get(role)
                if group["q"] not in domains.get((role, field), ()):
                    valid = False
                    break
                assigned[role, group["q"]] = field
            if not valid or prior_control.get("x") != controls.get(prior.get("x")):
                valid = False
                break
        if not valid or any(len({field for (kind, _), field in assigned.items() if kind == role})
                            != len(groups) for role in ("x", "f")):
            continue
        remember(assigned, reserve_other=True)
        if len(result["schemas"]) == max_schemas or exhausted_budget:
            break

    # Branch on the most constrained field. Every completed assignment is a
    # bijection for each role, so a field cannot be copied into multiple objects.
    ordered = sorted(domains, key=lambda key: (len(domains[key]), key))

    def assignments(index, assignment, used, order):
        nonlocal exhausted_budget
        if index == len(order):
            yield dict(assignment)
            return
        role, field = order[index]
        for quantity in sorted(domains[role, field] - used[role]):
            if budget is not None and not budget.spend(1):
                exhausted_budget = True
                return
            assignment[role, quantity] = field
            used[role].add(quantity)
            yield from assignments(index + 1, assignment, used, order)
            used[role].remove(quantity)
            del assignment[role, quantity]
            if exhausted_budget:
                return

    # If a reciprocal swap is illegal, direct the same constrained search
    # to the missing owner rather than spending its slot on a duplicate.
    if contrast_active and max_schemas >= len(owners):
        remaining_fields = [key for key in ordered if key != ("x", field_of_interest)]
        for owner in owners:
            if owner in represented or exhausted_budget or len(result["schemas"]) == max_schemas:
                continue
            if budget is not None and not budget.spend(1):
                exhausted_budget = True
                break
            fixed = {("x", owner): field_of_interest}
            example = next(assignments(0, fixed, {"x": {owner}, "f": set()}, remaining_fields), None)
            if example is not None:
                remember(example)
    truncated = False
    for assigned in (() if exhausted_budget else assignments(0, {}, {"x": set(), "f": set()}, ordered)):
        candidate = schema_from_assignment(assigned)
        if candidate in result["schemas"]:
            continue
        if len(result["schemas"]) == max_schemas:
            truncated = True
            break
        remember(assigned, candidate)
        if exhausted_budget:
            break
    if contrast_active:
        result["position_owner_coverage"] = {"field": field_of_interest, "owners": owners,
            "represented_owners": represented, "pairs": paired}
    result["exhaustive"] = not (truncated or exhausted_budget)
    result["status"] = ("enumerated" if result["schemas"] else
                        "search_exhausted" if exhausted_budget else "inconsistent")
    result["compute_cost"] = budget.spent - start if budget is not None else 0
    return result
