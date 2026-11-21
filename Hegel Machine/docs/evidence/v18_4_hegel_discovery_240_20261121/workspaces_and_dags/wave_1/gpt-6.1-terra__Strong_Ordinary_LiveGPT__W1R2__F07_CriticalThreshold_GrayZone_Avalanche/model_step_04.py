import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, rho, c, loss, threshold, power = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u)
    return base - loss * u * max(0.0, s - threshold) ** power

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_model(p, r) for r in rows]) - y
    lower = [0.0, 0.0, -10.0, 0.0, 0.0, 0.5]
    upper = [100.0, 1.5, 10.0, 100.0, 3.55, 5.0]
    best = None
    for threshold in (1.8, 2.2, 2.5, 2.7):
        for power in (1.0, 2.0, 3.0):
            result = least_squares(residual, [8.0, 0.7, 0.5, 1.0, threshold, power], bounds=(lower, upper), max_nfev=10000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, row))
