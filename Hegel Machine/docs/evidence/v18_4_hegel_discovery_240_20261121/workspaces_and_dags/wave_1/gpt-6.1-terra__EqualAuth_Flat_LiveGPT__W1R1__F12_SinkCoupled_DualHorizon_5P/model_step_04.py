import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_sqrt, slow_amp, slow_tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(u)
    return base - slow_amp * u * (-math.expm1(-h / slow_tau))

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    best = None
    for tau in (2.0, 8.0, 20.0, 60.0):
        result = least_squares(residual, [4.0, 0.7, 0.3, 1.5, tau], bounds=([0.0, 0.0, -10.0, 0.0, 0.01], [100.0, 2.0, 10.0, 20.0, 1000.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))