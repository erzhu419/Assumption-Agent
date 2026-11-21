import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_sqrt, threshold, strength, exponent = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(u)
    return base - strength * max(0.0, s - threshold) ** exponent

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    lower = [0.0, 0.0, -20.0, 0.0, 0.0, 0.5]
    upper = [100.0, 2.0, 20.0, 3.57, 100.0, 5.0]
    best = None
    for threshold in (1.8, 2.2, 2.5, 2.7):
        for exponent in (1.0, 2.0, 3.0):
            result = least_squares(residual, [8.0, 0.7, 1.0, threshold, 4.0, exponent], bounds=(lower, upper), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))
