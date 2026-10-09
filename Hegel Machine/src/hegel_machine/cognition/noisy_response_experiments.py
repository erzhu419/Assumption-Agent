"""Purchase distinguishable trajectories before choosing a noisy response refit."""

from statistics import mean, stdev

from .competing_object_experiments import _forecast, _probe_pool
from .evaluation import predict
from .models import Probe
from .probing import construct_probes


def _sequence_key(sequence):
    return tuple(tuple(sorted(point.items())) for point in sequence)


def plan_noisy_response_probe(spec, entries, budget, *, serial=0, length=24,
                              seed=7, used=(), observation_remaining=None,
                              reserve_rows=0, minimum_improvement=1e-7):
    """Choose maximum complete-trajectory squared prediction disagreement.

    These are existing public-control experiments. All programs are already
    frozen, including their fitted startup fallback. The caller purchases the
    chosen sequence and reserves fresh confirmation outside this comparison.
    """
    start = budget.spent
    entries = list(entries)
    result = {"probe": None, "predicted_disagreement": None, "hypotheses": [],
              "evaluated_probes": 0, "compute_cost": 0, "reason": None}

    def finish(reason):
        result.update(reason=reason, compute_cost=budget.spent - start)
        return result

    if len(entries) < 2 or len({entry["candidate"].program for entry in entries}) < 2:
        return finish("no_response_competition")
    if not spec.feature_bounds:
        return finish("no_public_controls")
    controls = set(spec.feature_bounds)
    programs = [entry["candidate"].program for entry in entries]
    if any({name for name, _ in program.observed_variables()} - controls
           for program in programs):
        return finish("unforecastable_response_inputs")
    count = min(spec.max_probe_length,
                max(4, length, max(program.history_depth for program in programs) + 2))
    if observation_remaining is not None and count + reserve_rows > observation_remaining:
        return finish("confirmation_reserve_unfunded")
    price = count * (len(entries) + 1)
    if budget.remaining < price:
        return finish("insufficient_budget")

    # The unified planner already uses these alternating endpoint sequences;
    # the shared pool and constructor add maintained and varied histories.
    bounds = tuple(spec.feature_bounds.items())
    pool = [tuple({name: limits[(step + axis) % 2]
                   for axis, (name, limits) in enumerate(bounds)} for step in range(count)),
            tuple({name: limits[(step // 3 + axis) % 2]
                   for axis, (name, limits) in enumerate(bounds)} for step in range(count))]
    pool.extend(_probe_pool(spec, serial, count))
    pool.extend(probe.sequence for probe in construct_probes(spec,
        [entry["candidate"] for entry in entries], round_number=serial,
        seed=seed, max_history=count - 2))
    pool.extend(probe.sequence for probe in spec.available_probes if probe.cost == count)
    seen = {_sequence_key(probe.sequence) for probe in used}
    best = None
    for sequence in pool:
        key = _sequence_key(sequence)
        if key in seen:
            continue
        seen.add(key)
        if not budget.spend(price):
            break
        predictions = [[entry.get("fallback", 0.) if value is None else value
                        for value in _forecast(entry["candidate"].program, sequence)]
                       for entry in entries]
        disagreement = mean((max(values) - min(values)) ** 2
                            for values in zip(*predictions))
        result["evaluated_probes"] += 1
        if best is None or disagreement > best[0]:
            best = disagreement, sequence, predictions
    if best is None:
        return finish("no_unqueried_control_trajectory")
    disagreement, sequence, predictions = best
    result["predicted_disagreement"] = disagreement
    if disagreement <= minimum_improvement:
        return finish("no_material_predicted_distinction")
    result.update(probe=Probe(f"noisy_response:{serial}", sequence),
        hypotheses=[{"candidate_id": entry["candidate"].candidate_id,
                     "program": entry["candidate"].program.to_dict(),
                     "fallback": entry.get("fallback", 0.), "predictions": prediction}
                    for entry, prediction in zip(entries, predictions)])
    return finish("predicted_response_distinction")


def assess_noisy_response_probe(parent, proposed, rows, budget, *, fallback=0.,
                                minimum_improvement=1e-7):
    """Compare two frozen responses using only the newly paid outcome rows.

    Positive gain supports the proposal. An interval crossing zero leaves the
    response competition unresolved; a tiny empirical loss difference does not
    turn an already supported response into a measured counterexample.
    """
    rows = list(rows)
    start = budget.spent
    result = {"gain": None, "margin": None, "lower_bound": None, "upper_bound": None,
              "supported": False, "refuted": False, "parent_loss": None,
              "candidate_loss": None, "observation_rows": len(rows), "compute_cost": 0}

    def finish(reason):
        result.update(reason=reason, compute_cost=budget.spent - start)
        return result

    if len(rows) < 2:
        return finish("insufficient_comparison_rows")
    if not budget.spend(3 * len(rows)):
        return finish("insufficient_budget")
    parent_losses = [(row.target - predict(parent["candidate"].program, row,
                      parent.get("fallback", fallback))) ** 2 for row in rows]
    candidate_losses = [(row.target - predict(proposed["candidate"].program, row,
                         proposed.get("fallback", fallback))) ** 2 for row in rows]
    gains = [before - after for before, after in zip(parent_losses, candidate_losses)]
    gain, margin = mean(gains), 3 * stdev(gains) / len(gains) ** .5
    supported, refuted = gain - margin > minimum_improvement, gain + margin < -minimum_improvement
    result.update(gain=gain, margin=margin, lower_bound=gain - margin,
        upper_bound=gain + margin, supported=supported, refuted=refuted,
        parent_loss=mean(parent_losses), candidate_loss=mean(candidate_losses))
    return finish("supported_response_gain" if supported else
                  "supported_incumbent_gain" if refuted else "response_gain_inconclusive")
