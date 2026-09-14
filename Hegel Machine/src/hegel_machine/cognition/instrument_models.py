"""Learn executable instrument calibrations from purchased paired observations.

Public field roles nominate control coordinates. Their response, coefficients
and interval boundaries must fit the measured values. The task's response target
is never used. These are training proposals; the controller must test substituting
them for actual instruments on fresh trajectories before relying on them.
"""

from itertools import combinations_with_replacement, product
from copy import deepcopy
import json
from math import ceil, floor

from .continuous_state import _linear_fit, _sum_terms
from .language import Program
from .models import Observation
from .structure_lab import ObjectSchema


def _schema(spec):
    payload = spec.context.get("object_schema")
    if payload is None:
        return None
    return ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)


def _product(terms):
    result = terms[0]
    for term in terms[1:]:
        result = Program("mul", children=(result, term))
    return result


def _sum(programs):
    return programs[0] if len(programs) == 1 else Program("object_sum", children=tuple(programs))


def _fit_numeric(rows, target, names, budget):
    count, columns = len(rows), {}
    best = None
    raw = [Program("feature", name=name) for name in names]
    scale = max(1., sum(value * value for value in target) / count)
    tolerance = 1e-10 * scale
    for degree in range(4):
        terms = [term for power in range(1, degree + 1)
                 for group in combinations_with_replacement(raw, power)
                 for term in [_product(group)]]
        missing = [term for term in terms if term not in columns]
        if (len(missing) + 2) * count > budget.remaining:
            break
        for term in missing:
            budget.spend(count)
            columns[term] = [term.predict(row) for row in rows]
        budget.spend(count)
        coefficients = _linear_fit([columns[term] for term in terms], target)
        program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
        budget.spend(count)
        error = sum((program.predict(row) - value) ** 2 for row, value in zip(rows, target)) / count
        proposal = {"program": program, "fit_loss": error, "kind": "polynomial",
                    "degree": degree, "tolerance": tolerance}
        if best is None or error < best["fit_loss"]:
            best = proposal
        if error <= tolerance:
            return proposal
    return best


def _fit_dynamic(rows, target, names, budget, max_history):
    """Test finite control history only after a static map has failed."""
    depth_limit = min(max_history, max((len(row.history) for row in rows), default=0))
    if not depth_limit:
        return None
    count, columns, best = len(rows), {}, None
    tolerance = 1e-10 * max(1., sum(value * value for value in target) / count)
    current = [Program("feature", name=name) for name in names]
    historical = {
        lag: [Program("if", children=(Program("available", lag=lag),
              Program("lag", name=name, lag=lag), Program("const", value=0.))) for name in names]
        for lag in range(1, depth_limit + 1)}
    # A reset is first explained by the same law with absent past controls.
    # Independent startup coordinates can interpolate a lone reset observation
    # while changing that law on the next reset. Give ordinary finite-memory
    # polynomials their chance before introducing a separate startup response.
    for separate_startup in (False, True):
        for complexity in range(2, depth_limit + 4):
            for degree in range(1, 4):
                depth = complexity - degree
                if not 1 <= depth <= depth_limit:
                    continue
                raw = [*current, *(term for lag in range(1, depth + 1) for term in historical[lag])]
                if separate_startup:
                    raw.extend(Program("available", lag=lag) for lag in range(1, depth + 1))
                terms = [term for power in range(1, degree + 1)
                         for group in combinations_with_replacement(raw, power)
                         for term in [_product(group)]]
                missing = [term for term in terms if term not in columns]
                if (len(missing) + 2) * count > budget.remaining:
                    continue
                for term in missing:
                    budget.spend(count)
                    columns[term] = [term.predict(row) for row in rows]
                budget.spend(count)
                coefficients = _linear_fit([columns[term] for term in terms], target)
                program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
                budget.spend(count)
                error = sum((program.predict(row) - value) ** 2 for row, value in zip(rows, target)) / count
                proposal = {"program": program, "fit_loss": error, "kind": "finite_history_polynomial",
                            "degree": degree, "history_depth": depth, "tolerance": tolerance}
                if best is None or error < best["fit_loss"]:
                    best = proposal
                if error <= tolerance:
                    return proposal
    return best


def _cut_options(left, right, grid, *, lower):
    """Both open and closed cuts are hypotheses; observed endpoints choose them."""
    if left is None or right is None:
        return [None]
    candidates = [value for value in grid if left <= value <= right]
    candidates.extend([(left + right) / 2, left, right])
    result = []
    for value in dict.fromkeys(candidates):
        # For a lower cut: left is observed false, right observed true.
        # For an upper cut: left is observed true, right observed false.
        for inclusive in (True, False):
            accepts_left = left >= value if lower and inclusive else left > value if lower else left <= value if inclusive else left < value
            accepts_right = right >= value if lower and inclusive else right > value if lower else right <= value if inclusive else right < value
            if (not accepts_left and accepts_right) if lower else (accepts_left and not accepts_right):
                result.append((value, inclusive))
        # Compare two distinct cut choices, preferring the public numerical grid
        # when it separates the purchased observations as well as a midpoint.
        if len(result) >= 2:
            break
    return result[:2]


def _interval(term, low, high):
    conditions = []
    if low is not None:
        value, inclusive = low
        constant = Program("const", value=value)
        conditions.append(Program("ge", children=(term, constant)) if inclusive
                          else Program("lt", children=(constant, term)))
    if high is not None:
        value, inclusive = high
        constant = Program("const", value=value)
        conditions.append(Program("ge", children=(constant, term)) if inclusive
                          else Program("lt", children=(term, constant)))
    if not conditions:
        return Program("const", value=1.)
    return conditions[0] if len(conditions) == 1 else Program("and", children=tuple(conditions))


def _fit_membership(rows, targets, control, bounds, budget):
    count = len(rows)
    if not set(targets) <= {0., 1.} or not budget.spend(count):
        return None
    ordered = sorted(zip((row.features[control] for row in rows), targets))
    if any(left[0] == right[0] and left[1] != right[1] for left, right in zip(ordered, ordered[1:])):
        return None
    groups, start = [], None
    for index, (_, positive) in enumerate(ordered):
        if positive and start is None:
            start = index
        if start is not None and (not positive or index == len(ordered) - 1):
            end = index if positive else index - 1
            groups.append((start, end))
            start = None
    if not groups:
        candidates = [Program("const", value=0.)]
    else:
        low, high = bounds
        # Half-grid candidates use public bounds, not object labels or the
        # apparatus's rounding implementation. Measured membership chooses cuts.
        grid = [index / 2 for index in range(ceil(2 * low), floor(2 * high) + 1)]
        term = Program("feature", name=control)
        alternatives = []
        for first, last in groups:
            lower = _cut_options(ordered[first - 1][0] if first else None,
                                 ordered[first][0], grid, lower=True)
            upper = _cut_options(ordered[last][0],
                                 ordered[last + 1][0] if last + 1 < len(ordered) else None,
                                 grid, lower=False)
            alternatives.append([_interval(term, left, right) for left, right in product(lower, upper)])
        candidates = []
        # Boundary choices are compared as complete measurable predicates.
        for offset in range(min(4, max(map(len, alternatives)))):
            parts = [choices[offset % len(choices)] for choices in alternatives]
            program = parts[0]
            for part in parts[1:]:
                program = Program("or", children=(program, part))
            candidates.append(program)
    best = None
    for program in dict.fromkeys(candidates):
        if not budget.spend(count):
            break
        error = sum((program.predict(row) - value) ** 2 for row, value in zip(rows, targets)) / count
        proposal = {"program": program, "fit_loss": error, "kind": "interval_membership",
                    "intervals": len(groups), "tolerance": 1e-10}
        if best is None or (error, program.size) < (best["fit_loss"], best["program"].size):
            best = proposal
    return best


def fit_instrument_model(command_spec, measured_spec, paired_rows, budget, *, max_history=4):
    """Fit each available measured field, returning a possibly partial JSON model.

    Every evaluated column, complete fit, prediction comparison and membership
    diagnosis consumes one row pass. No response targets or evaluation rows are
    read. Missing or unexplained fields are excluded from the executable binding.
    """
    pairs = list(paired_rows)
    schema, start = _schema(measured_spec), budget.spent
    object_model = measured_spec.context.get("object_model") if schema is None else None
    object_model = json.loads(object_model) if isinstance(object_model, str) else object_model
    payload = {"kind": "command_to_instrument", "status": "requires_fresh_validation",
        "source_schema": schema.to_dict() if schema else None,
        "command_features": list(command_spec.feature_names),
        "command_bounds": {name: list(bounds) for name, bounds in command_spec.feature_bounds.items()},
        "fields": {}, "unmodeled_fields": {}, "support_rows": len(pairs)}
    roles = {name: (group, role) for group in schema.objects
             for role, name in schema.fields[group].items()} if schema else {}
    partial_controls = {}
    if object_model:
        for group in object_model.get("groups", []):
            for role in ("q", "u", "x", "f"):
                name = group.get(role)
                if name is not None:
                    roles[name] = (group["object_id"], role)
                    if role in group.get("controls", {}):
                        partial_controls[name] = [group["controls"][role]]
        effects = object_model.get("control_effects", {})
        focus = object_model.get("focus_control")
        for name in measured_spec.feature_names:
            if focus and name in effects.get(focus, {}).get("before_fields", []):
                roles[name] = (roles.get(name, (None, None))[0], "f")
                partial_controls[name] = [focus]
            elif name not in partial_controls:
                partial_controls[name] = [control for control, effect in effects.items()
                                          if name in effect.get("before_fields", [])]
        payload["source_object_model"] = {"model_id": object_model.get("model_id"),
            "groups": deepcopy(object_model.get("groups", [])), "focus_control": focus,
            "field_controls": deepcopy(partial_controls),
            "unresolved": list(object_model.get("unresolved", []))}
    for name in measured_spec.feature_names:
        group, role = roles.get(name, (None, None))
        if schema and role == "f":
            controls = [schema.focus_control]
        elif schema and role is not None:
            controls = [schema.controls[member][role] for member in schema.members[group]
                        if role in schema.controls[member]]
        elif object_model:
            controls = partial_controls.get(name, [])
        else:
            controls = list(command_spec.feature_names)
        if not controls:
            payload["unmodeled_fields"][name] = {"reason": "unresolved_control_dependency"}
            continue
        applicable = [(command, measured) for command, measured in pairs
                      if name in measured.features and controls and
                      set(controls) <= set(command.features)]
        if len(applicable) < 6:
            payload["unmodeled_fields"][name] = {"reason": "insufficient_paired_measurements",
                                                "support_rows": len(applicable)}
            continue
        before = budget.spent
        # Calibration sees actual control histories, with each probe's reset
        # preserved. Response targets and measured-history values are omitted.
        rows = [Observation({control: command.features[control] for control in controls}, 0.,
                tuple({control: past[control] for control in controls if control in past}
                      for past in command.history))
                for command, _ in applicable]
        targets = [measured.features[name] for _, measured in applicable]
        if role == "f":
            limits = command_spec.feature_bounds.get(controls[0])
            fitted = _fit_membership(rows, targets, controls[0], limits, budget) if limits else None
        else:
            fitted = _fit_numeric(rows, targets, controls, budget)
        if fitted is None or fitted["fit_loss"] > fitted["tolerance"]:
            dynamic = _fit_dynamic(rows, targets, controls, budget, max_history)
            if dynamic is not None and (fitted is None or dynamic["fit_loss"] < fitted["fit_loss"]):
                fitted = dynamic
        cost = budget.spent - before
        metadata = {"role": role, "members": list(schema.members[group]) if schema else [],
                    "support_rows": len(rows), "compute_cost": cost}
        if fitted is not None and fitted["fit_loss"] <= fitted["tolerance"]:
            payload["fields"][name] = {**metadata, **fitted, "program": fitted["program"].to_dict()}
        else:
            payload["unmodeled_fields"][name] = {**metadata,
                "reason": "unexplained_with_available_budget",
                "fit_loss": fitted["fit_loss"] if fitted else None}
    payload["compute_cost"] = budget.spent - start
    return payload


def instrument_binding(payload):
    """Return only fields whose calibration explained their purchased values."""
    return {name: Program.from_dict(field["program"]) for name, field in payload["fields"].items()}


def rebind_instrument_model(payload, current_measured_spec):
    """Bind measured roles and controls, including fine-to-coarse aggregation.

    A group cannot be split into unobserved individual instruments. A coarser
    requested field is available only when learned groups cover all its members.
    """
    current = _schema(current_measured_spec)
    previous = payload.get("source_schema")
    binding = instrument_binding(payload)
    if previous is None or current is None:
        return {name: program for name, program in binding.items()
                if name in current_measured_spec.feature_names}
    source = ObjectSchema.from_dict(previous)
    controls = {old: current.controls[member][role]
                for member, fields in source.controls.items() if member in current.controls
                for role, old in fields.items() if role in current.controls[member]}
    controls[source.focus_control] = current.focus_control
    rebound = {}
    for group in current.objects:
        wanted = set(current.members[group])
        for role, name in current.fields[group].items():
            covered, parts = set(), []
            for candidate in sorted(source.objects, key=lambda item: -len(source.members[item])):
                members = set(source.members[candidate])
                old_name = source.fields[candidate].get(role)
                if (old_name not in binding or not members <= wanted or members & covered):
                    continue
                program = binding[old_name].remap_features(controls)
                if not program.required_features() <= set(current_measured_spec.feature_bounds):
                    continue
                if role == "x" and members != wanted:
                    program = Program("mul", children=(
                        Program("const", value=len(members) / len(wanted)), program))
                parts.append(program)
                covered.update(members)
            if covered == wanted and parts:
                rebound[name] = _sum(parts).simplified()
    return rebound
