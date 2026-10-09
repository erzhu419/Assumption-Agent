"""Use a paid fixed-background trajectory to distinguish condition and duration.

These low-dimensional fits describe this measured background. Their supported
structure and boundary intervals nominate the next full-response experiment;
they are not a response law for changing the controllable inputs.
"""

from collections import Counter
from dataclasses import replace

from .conditional_coordinates import _masked
from .language import Program
from .models import SearchBudget
from .research_programs import _entry, _Fitter, _rank


def propose_observed_condition_duration(spec, rows, budget, max_candidates=8, *, tolerance=1e-7):
    """Return standard claims with ``observed_condition_evidence`` for refitting.

    Only actually varying, noncommand measurements are candidate conditions.
    Each returned program is restricted to its purchased constant background;
    the attached evidence is what a subsequent full multivariable fit reuses.
    """
    rows = list(rows)
    if len(rows) < 8 or max_candidates < 1:
        return []
    names = set.intersection(*(set(row.features) for row in rows)) - {"__target__"}
    controlled = set(spec.feature_bounds) & names
    if any(len({row.features[name] for row in rows}) > 1 for name in controlled):
        return []
    fields = [name for name in spec.feature_names if name in names
        and name not in controlled and len({row.features[name] for row in rows}) > 1]
    if not fields:
        return []
    local = SearchBudget(min(40000, budget.remaining))
    count = len(rows)
    fitter = _Fitter(rows, local, spec.risk_quantile)
    upper = max(1, max(len(row.history) + 1 for row in rows))
    widths = list(range(1, upper + 1))
    measured, trials, plain, intervals = {}, [], [], {}
    bases = {}
    for name in fields:
        gate = Program("feature", name=name)
        column = fitter.column(gate)
        if column is None or not local.spend(count):
            continue
        occurrences = Counter(column)
        distinct, seen, brackets = sorted(occurrences), 0, []
        for left, right in zip(distinct, distinct[1:]):
            seen += occurrences[left]
            if seen >= 3 and count - seen >= 3:
                brackets.append((left, right))
        if not brackets:
            continue
        intervals[name] = brackets
        bases[name] = (Program("const", value=1.), gate,
            Program("mul", children=(gate, gate)))

    def evaluate(name, index, width, placement):
        key = name, index, width, placement
        if key in measured:
            return measured[key]
        left, right = intervals[name][index]
        condition = Program("ge", children=(Program("feature", name=name),
            Program("const", value=(left + right) / 2.)))
        terms = [_masked(term, condition, side, width, placement)
            for term in bases[name] for side in (True, False)]
        before = local.spent
        fit = fitter.fit(terms)
        if fit is None:
            return None
        record = {"control": name, "threshold": (left + right) / 2.,
            "threshold_interval": [left, right], "threshold_index": index,
            "width": width, "placement": placement, "loss": fit["loss"],
            "compute_cost": local.spent - before}
        trials.append(record)
        measured[key] = fit, record
        return fit, record

    # Every supported duration is compared with both temporal placements.
    # A common smooth response remains a deletion hypothesis, never a screen
    # that can exclude the width of a conditional response.
    for width in widths:
        for name in intervals:
            before = local.spent
            terms = [Program("window_sum", lag=width, children=(term,))
                if width > 1 else term for term in bases[name]]
            fit = fitter.fit(terms)
            if fit is not None:
                plain.append((fit, {"control": name, "width": width,
                    "loss": fit["loss"], "compute_cost": local.spent - before}))
            # The edge brackets already have three measured responses on
            # either side. Dropping them can make a well exposed condition
            # look like a different duration before boundary refinement runs.
            coarse = sorted({round(fraction * (len(intervals[name]) - 1))
                for fraction in (0., .5, 1.)})
            for index in coarse:
                for placement in ("inside_window", "outside_window"):
                    evaluate(name, index, width, placement)

    # Refine several leading structures rather than letting a wrong boundary
    # at one duration eliminate its competitors. The next round uses the actual
    # newly compared losses; all paid coarse alternatives remain available.
    fit_price = count * 8
    while measured and local.remaining >= fit_price:
        best_by_structure = {}
        for key, item in measured.items():
            family = key[0], key[2], key[3]
            if family not in best_by_structure or _rank(item[0]) < _rank(best_by_structure[family][1][0]):
                best_by_structure[family] = key, item
        leaders = sorted(best_by_structure.values(), key=lambda item: _rank(item[1][0]))[:4]
        if leaders[0][1][0]["loss"] <= 1e-12:
            break
        before = local.spent
        for key, _ in leaders:
            name, _, width, placement = key
            for _ in range(2):
                current = min((other for other in measured if other[0] == name
                    and other[2:] == key[2:]), key=lambda other: _rank(measured[other][0]))
                index = current[1]
                tested = sorted({0, len(intervals[name]) - 1, *(other[1] for other in measured
                    if other[0] == name and other[2:] == key[2:])})
                position = tested.index(index)
                adjacent = tested[max(0, position - 1):position + 2]
                proposals = list(dict.fromkeys([index - 1, index + 1,
                    *((left + right) // 2 for left, right in zip(adjacent, adjacent[1:])
                      if right - left > 1)]))
                for proposed in proposals:
                    if local.remaining < fit_price:
                        break
                    if 0 <= proposed < len(intervals[name]):
                        evaluate(name, proposed, width, placement)
        # A revised boundary changes the evidence for duration too. Propose
        # that measured bracket to the other window hypotheses, rather than
        # freezing their poorer fit at an earlier coarse boundary. A nearby
        # duration is cheap to discriminate first; the other supported widths
        # still follow in increasing expression complexity.
        best = min(measured, key=lambda key: _rank(measured[key][0]))
        name, index, best_width, _ = best
        revisit_widths = list(dict.fromkeys([best_width - 1, best_width + 1, *widths]))
        for width in revisit_widths:
            if width not in widths or local.remaining < fit_price:
                continue
            for placement in ("inside_window", "outside_window"):
                if local.remaining < fit_price:
                    break
                evaluate(name, index, width, placement)
            if min(fit["loss"] for fit, _ in measured.values()) <= 1e-12:
                break
        if local.spent == before:
            break

    budget.spend(local.spent)
    background = {name: rows[0].features[name] for name in sorted(controlled)}
    options = []
    for fit, record in plain:
        options.append((fit, record, "smooth_duration"))
    options.extend((fit, record, "conditional_duration") for fit, record in measured.values())
    # Compare all actually fitted explanations at the strongest duration.
    # Several cheaper windows must not displace its smooth deletion or poorer
    # gate placement from the nominations passed to the full response fit.
    groups = {}
    for option in options:
        groups.setdefault((option[1]["control"], option[1]["width"]), []).append(option)
    ordered = sorted(groups.values(), key=lambda group: min(_rank(item[0]) for item in group))
    ranked = [item for group in ordered for item in sorted(group, key=lambda item: _rank(item[0]))]
    selected, seen = [], set()
    for fit, record, status in ranked:
        family = record["control"], record["width"], record.get("placement"), status
        if family in seen:
            continue
        seen.add(family)
        smooth_loss = min((common["loss"] for common, trial in plain
                          if trial["control"] == record["control"]), default=float("inf"))
        supported = fit["loss"] <= tolerance
        gain = smooth_loss - fit["loss"]
        condition_identified = status == "conditional_duration" and supported and gain > tolerance
        evidence_status = ("unresolved_duration" if not supported else
            "unidentified_condition_duration" if status == "conditional_duration" and not condition_identified
            else status)
        evidence = {"status": evidence_status,
            "proposed_structure": status, "control": record["control"],
            "width": record["width"],
            "width_candidates": [record["width"]],
            "placement_candidates": [record["placement"]] if "placement" in record else [],
            "threshold_intervals": [record["threshold_interval"]] if "threshold_interval" in record else [],
            "gate_coordinate": Program("feature", name=record["control"]).to_dict(),
            "background": background, "fit_rows": count, "loss": fit["loss"],
            "support_scope": "fixed_control_background",
            "local_program": fit["program"].to_dict(),
            "response_supported": supported,
            "condition_identified": condition_identified,
            "conditional_loss": fit["loss"], "smooth_loss": smooth_loss,
            "gain": gain,
            "parameter_trials": trials, "duration_trials": [item for _, item in plain]}
        if "placement" in record:
            evidence.update(placement=record["placement"], threshold=record["threshold"],
                            threshold_interval=record["threshold_interval"])
        step = {"operation": "discover_observed_condition_duration",
            "question": "Distinguish temporal condition structure on a purchased fixed background",
            "program": fit["program"], "loss": fit["loss"],
            "width": record["width"], **({"placement": record["placement"],
                "threshold_interval": record["threshold_interval"],
                "gate_coordinate": evidence["gate_coordinate"]}
                if "placement" in record else {}),
            "background": background, "fit_rows": count,
            "observed_condition_evidence": evidence}
        entry = _entry(fit, [step], count, local.spent, len(selected))
        entry["candidate"] = replace(entry["candidate"],
            candidate_id=f"observed_condition_duration:{len(selected)}",
            rationale="A fixed-background measurement nominates condition structure for full response refitting")
        entry["observed_condition_evidence"] = evidence
        selected.append(entry)
        if len(selected) >= max_candidates:
            break
    return selected
