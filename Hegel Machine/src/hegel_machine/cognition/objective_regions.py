"""Propose local experiments around competing goal minima and a paid winner."""

from itertools import combinations, product, zip_longest

from .decision_experiments import _predict, objective_pool


def propose_decision_regions(spec, model, rows, budget):
    """Spend predictions on decision-relevant proposals, before any new label.

    An uncertain interaction model may propose an experiment; its adoption still
    depends on future evidence. The full symmetric local pool keeps directions
    that an unvalidated explanation predicts poorly: those experiments may
    expose its mistake rather than confirm its preferred direction.
    """
    models = [model]
    if model.get("interaction_challenger") is not None:
        models.append(model["interaction_challenger"])
    names = spec.feature_names
    pairs = tuple(combinations(names, 2))

    def key(point):
        return tuple(point[name] for name in names)

    anchors = {}
    for explanation in models:
        # objective_pool prices the joint stationary-point solve. Its first
        # point is the model minimum; its global sweeps are not experiment offers.
        solve_cost = len(names) ** 3 if explanation.get("interactions") else 0
        if budget.remaining >= solve_cost + len(models):
            point = objective_pool(spec, explanation, budget)[0]
            anchors.setdefault(key(point), dict(point))
    bound = float(spec.context["constraint_upper_bound"])
    feasible = [row for row in rows if row.get("observation_status") != "disputed"
                and row["observation"]["target"] <= bound]
    if feasible:
        winner = min(feasible, key=lambda row: row["context"]["objective"])
        point = winner["observation"]["features"]
        anchors.setdefault(key(point), dict(point))

    predictions = {}

    def score(point):
        point_key = key(point)
        if point_key not in predictions:
            if not budget.spend(len(models)):
                return None
            predictions[point_key] = min(_predict(spec, explanation, point)
                                         for explanation in models)
        return predictions[point_key]

    result = {}
    for point_key, point in anchors.items():
        if score(point) is None:
            break
        result[point_key] = point

    local = {}
    exhausted = False
    for anchor in result.values():
        def shifted(coordinates, signs):
            point = dict(anchor)
            for name, sign in zip(coordinates, signs):
                low, high = spec.feature_bounds[name]
                point[name] = min(high, max(low, point[name] + sign * 4 * (high - low) / 32))
            if all(point[name] != anchor[name] for name in coordinates):
                return point
            return None

        groups = [[shifted((name,), (sign,)) for name in names for sign in (-1, 1)],
                  *[[shifted(pair, signs) for signs in product((-1, 1), repeat=2)]
                    for pair in pairs]]
        # Interleave axes and coordinate pairs so a tight budget still gets
        # a chance to propose a joint experiment before exhausting the axes.
        for batch in zip_longest(*groups):
            for point in batch:
                if point is None:
                    continue
                value = score(point)
                if value is None:
                    exhausted = True
                    break
                local.setdefault(key(point), (value, point))
            if exhausted:
                break
        if exhausted:
            break
    for _, point in sorted(local.values(), key=lambda item: item[0]):
        result.setdefault(key(point), point)
    return list(result.values())
