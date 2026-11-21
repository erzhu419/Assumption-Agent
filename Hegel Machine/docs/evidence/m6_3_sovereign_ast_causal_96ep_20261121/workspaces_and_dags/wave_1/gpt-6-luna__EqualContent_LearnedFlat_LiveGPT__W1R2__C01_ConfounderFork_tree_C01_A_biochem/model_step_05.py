import math
import numpy as np
from scipy.optimize import least_squares

_STATE = None

def fit(train_rows):
    global _STATE
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r.get('u_ctrl', 0.0)) for r in rows])
    keys = ['x_ind', 'z_1', 'z_2', 'z_3', 'z_4']
    tests = []
    for key in keys:
        vals = sorted(set(float(r[key]) for r in rows))
        for a, b in zip(vals[:-1], vals[1:]):
            t = (a + b) / 2.0
            for direction in (1, -1):
                mask = np.array([float(direction * (float(r[key]) - t) > 0) for r in rows])
                tests.append((key, t, direction, mask))
    rules = [(test, None) for test in tests]
    for i, left in enumerate(tests):
        for right in tests[i + 1:]:
            rules.append((left, right))
    best = None
    h = (x / 1.5) ** 2
    for left, right in rules:
        for op in (0, 1) if right is not None else (0,):
            if right is None:
                ind = left[3]
            elif op == 0:
                ind = left[3] * right[3]
            else:
                ind = np.maximum(left[3], right[3])
            for init in ([7., 1., .1, 1., 0.], [8., .5, .1, 1., 0.], [6., 2., .2, -1., 0.]):
                def residual(p):
                    vmax, k, c, d, base = p
                    return base + vmax * h / (k + h) - c * u + d * ind - y
                res = least_squares(residual, init, bounds=([-50., 1e-7, -20., -20., -50.], [50., 100., 20., 20., 50.]), max_nfev=3000)
                score = float(np.mean(res.fun ** 2))
                if best is None or score < best[0]:
                    best = (score, left, right, op, res.x)
    _, left, right, op, params = best
    _STATE = (left, right, op, params)
    return predict

def predict(row):
    left, right, op, p = _STATE
    def test(spec):
        key, threshold, direction, _ = spec
        return float(direction * (float(row[key]) - threshold) > 0)
    a = test(left)
    if right is None:
        ind = a
    else:
        b = test(right)
        ind = a * b if op == 0 else max(a, b)
    vmax, k, c, d, base = p
    h = (float(row['x_ind']) / 1.5) ** 2
    return float(base + vmax * h / (k + h) - c * float(row.get('u_ctrl', 0.0)) + d * ind)
