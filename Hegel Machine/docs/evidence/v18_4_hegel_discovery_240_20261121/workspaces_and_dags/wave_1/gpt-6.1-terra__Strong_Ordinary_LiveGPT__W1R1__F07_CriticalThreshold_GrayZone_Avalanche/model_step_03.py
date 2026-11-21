import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    vmax, rho, c_sqrt, strength, threshold, exponent = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(u)
    avalanche = strength * u * max(0.0, s - threshold) ** exponent
    return base - avalanche

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    lower = [0.0, 0.0, -10.0, 0.0, 0.0, 0.3]
    upper = [50.0, 2.0, 10.0, 50.0, 3.5, 4.0]
    best = None
    for threshold in (1.8, 2.1, 2.3, 2.5):
        for exponent in (1.0, 2.0):
            result = least_squares(residual, [5.0, 0.7, 1.0, 1.5, threshold, exponent], bounds=(lower, upper), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_value(_params, row))