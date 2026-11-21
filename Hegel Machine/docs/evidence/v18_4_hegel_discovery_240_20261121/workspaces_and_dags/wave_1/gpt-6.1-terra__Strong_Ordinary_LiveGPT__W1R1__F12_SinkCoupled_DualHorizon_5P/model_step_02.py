import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_sqrt, slow_amp, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(max(0.0, u))
    return base - slow_amp * u * (-math.expm1(-h / tau))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    best = None
    for tau in (1.0, 5.0, 15.0, 40.0, 100.0):
        for rho in (0.5, 0.9):
            result = least_squares(residual, [4.0, rho, 0.3, 1.0, tau], bounds=([0.0, 0.0, -20.0, 0.0, 0.01], [100.0, 2.0, 20.0, 100.0, 1000.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(row, _params))