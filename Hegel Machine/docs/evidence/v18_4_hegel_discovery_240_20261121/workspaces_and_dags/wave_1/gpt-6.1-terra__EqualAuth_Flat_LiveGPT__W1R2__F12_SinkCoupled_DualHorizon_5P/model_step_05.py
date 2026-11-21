import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_family = None

def _value(row, p, family):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    h = float(row['H'])
    v, rho, c, af, slow, tf, ts = p
    sink = 1.0 - rho * g / (1.0 + g)
    base = v * sink * u / (1.2 + u) + c * math.sqrt(max(u, 0.0))
    fast = -math.expm1(-h / tf)
    slow_response = -math.expm1(-h / ts)
    if family == 0:
        return base * (1.0 + af * fast - slow * slow_response)
    if family == 1:
        return base + u * sink * (af * fast - slow * slow_response)
    if family == 2:
        return base + u * (af * fast - slow * (1.0 + rho * g) * slow_response)
    if family == 3:
        return base + u * (af * sink * fast - slow * slow_response)
    return base + u * (af * fast - slow * sink * slow_response)

def fit(train_rows):
    global _params, _family
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lower = [0.0, 0.0, -5.0, -5.0, -5.0, 0.02, 0.5]
    upper = [20.0, 2.0, 5.0, 10.0, 10.0, 10.0, 100.0]
    best = None
    for family in range(5):
        def residual(p):
            return np.array([_value(r, p, family) for r in rows]) - y
        for tf, ts in [(0.3, 8.0), (1.0, 15.0), (3.0, 30.0)]:
            start = [2.0, 0.7, 0.3, 1.0, 2.0, tf, ts]
            result = least_squares(residual, start, bounds=(lower, upper), max_nfev=4000)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, family, result.x)
    _, _family, _params = best

def predict(row):
    return float(_value(row, _params, _family))
