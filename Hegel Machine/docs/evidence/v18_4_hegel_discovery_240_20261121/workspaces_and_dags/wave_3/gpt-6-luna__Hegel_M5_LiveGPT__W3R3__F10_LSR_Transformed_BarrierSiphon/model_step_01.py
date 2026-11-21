import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    tau = 0.02
    u = float(row['u_now']) / tau
    s = float(row['s_grad'])
    return u, s

def _value(params, row):
    j0, beta_bar, rho, c_sqrt = params
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(p, r) for r in rows]) - y
    starts = [
        [1.0, 1.0, 0.70, 0.1],
        [3.0, 2.0, 0.70, 0.5],
        [8.0, 0.5, 0.50, 0.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.05, -20.0], [1e4, 100.0, 0.95, 20.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return np.asarray(best[1], dtype=float)

def predict(row):
    params = fit(_training_rows)
    return float(_value(params, row))