import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    a, b, power, c, q, d, e = p
    u = max(float(row['u_now']), 0.0)
    s = max(float(row['s_grad']), 0.0)
    source = np.log1p(u / b) if abs(power) < 1e-5 else ((u + b)**power - b**power) / power
    return (a * source + d) / (1.0 + c * s)**q + e

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    lower = [0.0, 0.001, -2.0, 0.001, 0.01, -20.0, -20.0]
    upper = [50.0, 20.0, 2.0, 20.0, 4.0, 20.0, 20.0]
    best = None
    for power in (-0.5, 0.0, 0.5):
        for c in (0.2, 1.0, 3.0):
            result = least_squares(residual, [2.0, 0.5, power, c, 0.5, 1.0, 0.0], bounds=(lower, upper), max_nfev=4000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))
