import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c, af, slow, tf, delta, coupling = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    sink = s / (1.0 + s)
    base = vmax * u / (1.2 + u) * (1.0 - rho * sink) + c * math.sqrt(max(u, 0.0))
    fast_response = -math.expm1(-h / tf)
    slow_response = -math.expm1(-h / (tf + delta))
    return base + u * (af * fast_response - slow * (1.0 + coupling * sink) * slow_response)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    lower = [0.0, 0.0, -10.0, -10.0, 0.0, 0.03, 0.03, -1.0]
    upper = [30.0, 1.5, 10.0, 20.0, 30.0, 20.0, 100.0, 5.0]
    best = None
    for tf in (0.5, 2.0, 5.0):
        for delta in (5.0, 15.0, 35.0):
            p0 = [3.0, 0.7, 0.2, 1.0, 2.0, tf, delta, 0.2]
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=5000)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, result.x)
    _params = best[1]

def predict(row):
    return float(_value(row, _params))
