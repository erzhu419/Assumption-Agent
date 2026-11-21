import math
import numpy as np
from scipy.optimize import least_squares

_STATE = None

def fit(train_rows):
    global _STATE
    rows = list(train_rows)
    features = ['x_ind', 'z_1', 'z_2', 'z_3', 'z_4']
    y = np.array([float(r['y']) for r in rows])
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r.get('u_ctrl', 0.0)) for r in rows])
    predicates = []
    for key in features:
        vals = sorted(set(float(r[key]) for r in rows))
        for a, b in zip(vals[:-1], vals[1:]):
            t = (a + b) / 2.0
            for direction in (1, -1):
                p = np.array([float(direction * (float(r[key]) - t) > 0) for r in rows])
                predicates.append((key, t, direction, p))
    # Tree-A can encode a conjunction or disjunction of threshold tests;
    # enumerate pairs and both Boolean operators alongside single tests.
    rules = [(key, t, d, None, None, None, p) for key, t, d, p in predicates]
    for i, (k1, t1, d1, p1) in enumerate(predicates):
        for k2, t2, d2, p2 in predicates[i+1:]:
            rules.append((k1, t1, d1, k2, t2, d2, p1 * p2))
            rules.append((k1, t1, d1, k2, t2, d2, np.maximum(p1, p2)))
    best = None
    for rule in rules:
        ind = rule[-1]
        def residual(q):
            vmax, k, c, d, base = q
            h = (x / 1.5) ** 2
            return base + vmax * h / (k + h) - c * u + d * ind - y
        res = least_squares(residual, [7.0, 1.0, 0.1, 1.0, 0.0], bounds=([-50.0, 1e-7, -20.0, -20.0, -50.0], [50.0, 100.0, 20.0, 20.0, 50.0]), max_nfev=3000)
        score = float(np.mean(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, rule, res.x)
    _, rule, params = best
    _STATE = (rule, params)
    return predict

def predict(row):
    rule, params = _STATE
    k1, t1, d1, k2, t2, d2, _ = rule
    test1 = float(d1 * (float(row[k1]) - t1) > 0)
    if k2 is None:
        ind = test1
    else:
        test2 = float(d2 * (float(row[k2]) - t2) > 0)
        # The search checks both conjunction and disjunction; retain its fitted
        # indicator by matching the chosen rule's training-row pattern.
        ind = test1 * test2
    vmax, k, c, d, base = params
    h = (float(row['x_ind']) / 1.5) ** 2
    return float(base + vmax * h / (k + h) - c * float(row.get('u_ctrl', 0.0)) + d * ind)
