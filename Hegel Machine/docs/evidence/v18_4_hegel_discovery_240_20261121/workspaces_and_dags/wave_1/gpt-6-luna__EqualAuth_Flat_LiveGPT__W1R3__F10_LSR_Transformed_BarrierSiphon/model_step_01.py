import math
import numpy as np
from scipy.optimize import least_squares

_tau = 3.0
_params = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now']) / _tau
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.2, 0.1]), bounds=(np.zeros(4), np.full(4, np.inf)))
    _params = result.x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))