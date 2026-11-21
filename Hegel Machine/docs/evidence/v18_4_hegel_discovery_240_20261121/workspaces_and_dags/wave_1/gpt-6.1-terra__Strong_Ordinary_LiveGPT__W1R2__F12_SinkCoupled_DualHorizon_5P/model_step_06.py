import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return u, s, h

def _calc(p, u, s, h, kind, shape, branch):
    vmax, rho, af, slow, tf, ts = p[:6]
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * u / (1.2 + u) * sink
    if branch:
        base = base + p[6] * np.sqrt(np.maximum(u, 0.0))
    fast = -np.expm1(-h / tf)
    slow_response = -np.expm1(-np.power(np.maximum(h, 0.0) / ts, shape))
    if kind == 0:
        return base + u * (af * fast - slow * slow_response)
    if kind == 1:
        return base + u * sink * (af * fast - slow * slow_response)
    if kind == 2:
        return base + u * (af * fast * sink - slow * slow_response)
    return base + u * (af * fast - slow * slow_response * (1.0 + rho * s / (1.0 + s)))

def fit(train_rows):
    global _model
    u, s, h = _features(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for kind in range(4):
        for shape in (1.0, 2.0):
            for branch in (False, True):
                lower = [0.0, 0.0, -10.0, 0.0, 0.02, 0.2]
                upper = [20.0, 1.5, 10.0, 20.0, 30.0, 100.0]
                if branch:
                    lower.append(-10.0)
                    upper.append(10.0)
                for tf, ts in ((0.5, 10.0), (2.0, 20.0), (5.0, 5.0)):
                    p0 = [3.0, 0.7, 0.8, 1.5, tf, ts] + ([0.0] if branch else [])
                    result = least_squares(lambda p: _calc(p, u, s, h, kind, shape, branch) - y, p0, bounds=(lower, upper), max_nfev=3000)
                    error = float(np.mean(result.fun ** 2))
                    if best is None or error < best[0]:
                        best = (error, result.x, kind, shape, branch)
    _model = best[1:]

def predict(row):
    p, kind, shape, branch = _model
    u, s, h = _features([row])
    return float(_calc(p, u, s, h, kind, shape, branch)[0])
