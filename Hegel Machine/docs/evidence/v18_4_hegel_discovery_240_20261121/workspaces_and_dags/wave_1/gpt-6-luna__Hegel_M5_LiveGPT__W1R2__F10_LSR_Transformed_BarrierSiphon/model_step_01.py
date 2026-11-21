import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 0.02

def _predict_with(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _PARAMS
    def residuals(p):
        return np.asarray([_predict_with(r, p) - float(r['y']) for r in train_rows])
    starts = [np.array([1.0, 1.0, 0.70, 0.0]), np.array([2.0, 0.5, 0.70, -0.5]), np.array([1.0, 3.0, 0.5, 0.5])]
    best = None
    for x0 in starts:
        result = least_squares(residuals, x0, bounds=([-np.inf, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]), max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _PARAMS))