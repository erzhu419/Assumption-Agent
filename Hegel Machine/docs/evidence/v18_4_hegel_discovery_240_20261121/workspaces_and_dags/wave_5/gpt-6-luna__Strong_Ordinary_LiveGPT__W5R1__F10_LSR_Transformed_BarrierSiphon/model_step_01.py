import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.02
_PARAMS = None

def _evaluate(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_evaluate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.0]), bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, 1.0, np.inf]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_evaluate(row, _PARAMS))