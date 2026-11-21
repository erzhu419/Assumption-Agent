import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])  # tau_scale = 1
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    best = None
    # Use the authorized warm-start region and multiple starts to handle parameter correlation.
    for start in ([6.5, 0.8, 0.70, 0.1], [5.0, 1.5, 0.68, 0.7], [8.0, 0.3, 0.75, -0.5], [6.5, 1.0, 0.70, 0.0]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.01, 0.0, 0.05, -2.0], [30.0, 10.0, 0.95, 2.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))