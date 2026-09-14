"""Keep evidence-permitted object assignments as executable alternatives.

The object inducer may identify a role without identifying which quantity owns
it.  A finite set of one-to-one assignments can still support a shared law.
Enumerating those assignments never turns a partial object model into a unique
schema; consumers must retain the alternatives and establish program equivalence
separately.
"""

from copy import deepcopy


def enumerate_object_schemas(model, budget=None, *, max_schemas=16):
    """Enumerate complete schemas allowed by the current membership evidence.

Quantity and drive bindings must already be identified.  Position and focus
memberships range only over the inducer's candidates, respecting established
assignments. ``exhaustive`` refers to this evidence-permitted finite set, not to
every possible ontology. A truncated set cannot authorize an equivalence claim.
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

    # Branch on the most constrained field. Every completed assignment is a
    # bijection for each role, so a field cannot be copied into multiple objects.
    ordered = sorted(domains, key=lambda key: (len(domains[key]), key))
    assignment, used = {}, {"x": set(), "f": set()}
    exhausted_budget = False

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
    for assigned in assignments(0):
        if len(result["schemas"]) == max_schemas:
            truncated = True
            break
        fields = {group["object_id"]: {"q": group["q"], "u": group["u"],
            "x": assigned["x", group["q"]], "f": assigned["f", group["q"]]}
            for group in groups}
        result["schemas"].append({"objects": [group["object_id"] for group in groups],
            "fields": fields,
            "members": {group["object_id"]: [group["object_id"]] for group in groups},
            "controls": {group["object_id"]: {"q": group["controls"]["q"],
                "u": group["controls"]["u"], "x": controls[fields[group["object_id"]]["x"]]}
                for group in groups},
            "focus_control": model["focus_control"]})
    result["exhaustive"] = not (truncated or exhausted_budget)
    result["status"] = ("enumerated" if result["schemas"] else
                        "search_exhausted" if exhausted_budget else "inconsistent")
    result["compute_cost"] = budget.spent - start if budget is not None else 0
    return result
