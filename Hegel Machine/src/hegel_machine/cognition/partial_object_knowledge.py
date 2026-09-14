"""Use identified object roles while other relations remain unresolved."""

import json

from .language import Program
from .research_programs import _entry, _Fitter, _rank


def partial_object_schema(model):
    """Describe only established roles for binding a learned computation.

    This sparse metadata does not assert that object induction is complete.
    In particular, it must not authorize a search requiring focal membership.
    """
    groups = model.get("groups", [])
    if not groups:
        return None
    return {"objects": [group["object_id"] for group in groups],
        "fields": {group["object_id"]: {role: group[role] for role in ("q", "u", "x", "f")
                                       if role in group} for group in groups},
        "members": {group["object_id"]: [group["object_id"]] for group in groups},
        "controls": {group["object_id"]: dict(group.get("controls", {})) for group in groups},
        "focus_control": model.get("focus_control")}


def propose_partial_object_claims(spec, training, budget, *, max_candidates=16):
    """Fit role reductions and temporal compositions from a partial object graph.

    An unidentified membership never becomes a focal coordinate. Reductions
    carry their actual object coverage, and are hypotheses about the response,
    not declarations of conservation. The caller compares them on later data.
    """
    model = spec.context.get("object_model")
    if not model or not training:
        return []
    model = json.loads(model) if isinstance(model, str) else model
    groups = model.get("groups", [])
    roles, coverage, coordinates = [], {}, []
    for role in ("q", "u", "x"):
        identified = [group for group in groups
                      if group.get(role) in spec.feature_names]
        if not identified:
            continue
        terms = [Program("feature", name=group[role]) for group in identified]
        roles.append(Program("object_sum", children=tuple(terms)))
        coordinates.extend(terms)
        coverage[role] = [group["object_id"] for group in identified]
    if not roles:
        return []
    start, fitter = budget.spent, _Fitter(training, budget, spec.risk_quantile)
    blocks = [("role_totals", roles), ("object_linear_combination", coordinates)]
    if any(row.history for row in training):
        differences = [Program("event_delta", lag=1, children=(term,)) for term in roles]
        blocks.extend((("event_totals", differences), ("current_and_event_totals", [*roles, *differences])))
        for width in (2, 3, 4):
            blocks.append((f"window_{width}_totals", [Program("window_sum", lag=width,
                children=(term,)) for term in [*roles, *differences]]))
        for decay in (.2, .5, .8, .95):
            blocks.append((f"accumulated_totals_{decay}", [Program("accumulate", value=decay,
                children=(term, Program("const", value=0.))) for term in roles]))
    fitted = []
    for operation, terms in blocks:
        fit = fitter.fit(terms)
        if fit is not None:
            fitted.append((fit, operation))
    fitted.sort(key=lambda item: _rank(item[0]))
    entries, seen = [], set()
    for fit, operation in fitted:
        if fit["program"] in seen:
            continue
        seen.add(fit["program"])
        step = {"operation": operation, "question": "Test a composition of the identified object roles",
                "program": fit["program"], "loss": fit["loss"]}
        entry = _entry(fit, [step], len(training), budget.spent - start, len(entries))
        entry.update(kind=f"partial_object:{operation}",
            partial_object_knowledge={"source_model_id": model.get("model_id"),
                "role_coverage": coverage, "unresolved": list(model.get("unresolved", [])),
                "used_fields": sorted(fit["program"].required_features())})
        entries.append(entry)
        if len(entries) >= max_candidates:
            break
    return entries
