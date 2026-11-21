import math
import numpy as np
from scipy.optimize import least_squares


def _value(row, p, branch):
    vmax, rho, c = p
    u = max(1e-12, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'sqrt_plus':
        return base + c * math.sqrt(u)
    if branch == 'sqrt_minus':
        return base - c * math.sqrt(u)
    if branch == 'linear_leak':
        return base - c * u
    return base


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    candidates = ['sqrt_plus', 'sqrt_minus', 'linear_leak', 'none']
    best = None
    for branch in candidates:
        npar = 2 if branch == 'none' else 3
        def residual(p):
            pp = [p[0], p[1], p[2] if npar == 3 else 0.0]
            return np.asarray([_value(r, pp, branch) for r in rows]) - y
        starts = ([3.0, 0.5, 0.5] if npar == 3 else [3.0, 0.5])
        result = least_squares(residual, starts, max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, branch, result.x)
    return {'branch': best[1], 'params': [float(v) for v in best[2]]}


def predict(row):
    # Parameters are populated from fit's returned model by the evaluator.
    raise RuntimeError('predict requires fitted parameters')