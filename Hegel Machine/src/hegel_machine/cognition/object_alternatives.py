"""Keep evidence-permitted object assignments as executable alternatives.

The object inducer may identify a role without identifying which quantity owns
it.  A finite set of one-to-one assignments can still support a shared law.
Enumerating those assignments never turns a partial object model into a unique
schema; consumers must retain the alternatives and establish program equivalence
separately.
"""

from copy import deepcopy


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

    seen_preferred = []
    for schema in preferred_schemas:
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
        candidate = schema_from_assignment(assigned)
        if candidate not in result["schemas"]:
            result["schemas"].append(candidate)
        if len(result["schemas"]) == max_schemas:
            break

    # Branch on the most constrained field. Every completed assignment is a
    # bijection for each role, so a field cannot be copied into multiple objects.
    ordered = sorted(domains, key=lambda key: (len(domains[key]), key))
    assignment, used = {}, {"x": set(), "f": set()}

    def assignments(index):
        nonlocal exhausted_budget
        if index == len(ordered):
            yield dict(assignment)
            return
        role, field = ordered[index]
        for quantity in sorted(domains[role, field] - used[role]):
            if budget is not None and not budget.spend(1):
                exhausted_budget = True
                return
            assignment[role, quantity] = field
            used[role].add(quantity)
            yield from assignments(index + 1)
            used[role].remove(quantity)
            del assignment[role, quantity]
            if exhausted_budget:
                return

    truncated = False
    for assigned in (() if exhausted_budget else assignments(0)):
        candidate = schema_from_assignment(assigned)
        if candidate in result["schemas"]:
            continue
        if len(result["schemas"]) == max_schemas:
            truncated = True
            break
        result["schemas"].append(candidate)
    result["exhaustive"] = not (truncated or exhausted_budget)
    result["status"] = ("enumerated" if result["schemas"] else
                        "search_exhausted" if exhausted_budget else "inconsistent")
    result["compute_cost"] = budget.spent - start if budget is not None else 0
    return result
