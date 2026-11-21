import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _value(row, p, kind):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    v, rho, c, af, tf, a_s, ts = p
    sink = 1.0 - rho * s / (1.0 + s)
    base = v * sink * u / (1.2 + u) + c * math.sqrt(max(0.0, u))
    fast = -math.expm1(-h / tf)
    slow = -math.expm1(-h / ts)
    if kind == 0:
        return base + u * (af * fast - a_s * slow)
    if kind == 1:
        return base + u * sink * (af * fast - a_s * slow)
    if kind == 2:
        return base + u * (af * sink * fast - a_s * slow)
    if kind == 3:
        return base + u * (af * fast - a_s * sink * slow)
    if kind == 4:
        return base * (1.0 + af * fast) - u * a_s * slow
    return base + u * (af * fast - a_s * (h * h / (ts * ts + h * h)))

def fit(train_rows):
    global _params, _kind
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lower = [0.0, 0.0, -10.0, 0.0, 0.03, 0.0, 0.2]
    upper = [30.0, 1.5, 10.0, 30.0, 20.0, 30.0, 200.0]
    best = None
    for kind in range(6):
        def residual(p):
            return np.array([_value(r, p, kind) for r in rows]) - y
        for tf, ts in [(0.4, 8.0), (1.0, 15.0), (3.0, 25.0), (0.15, 3.0)]:
            p0 = [3.0, 0.7, 0.1, 1.0, tf, 2.0, ts]
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=5000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, kind, result.x)
    _kind, _params = best[1], best[2]

def predict(row):
    return float(_value(row, _params, _kind))
