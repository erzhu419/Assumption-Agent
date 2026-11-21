import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c, amp, threshold, exponent = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u)
    return base - amp * max(0.0, s - threshold) ** exponent

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    bounds = ([0.0, 0.69, -10.0, 0.0, 0.0, 0.25], [30.0, 0.72, 10.0, 100.0, 3.5, 5.0])
    best = None
    for threshold in (1.8, 2.2, 2.4, 2.6, 2.8):
        for exponent in (1.0, 2.0, 3.0):
            p0 = [5.0, 0.70, 0.7, 3.0, threshold, exponent]
            result = least_squares(residual, p0, bounds=bounds, max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))