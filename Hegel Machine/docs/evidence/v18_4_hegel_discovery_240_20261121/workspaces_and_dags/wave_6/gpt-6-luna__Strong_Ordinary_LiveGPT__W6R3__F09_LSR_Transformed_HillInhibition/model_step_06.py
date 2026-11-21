import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(row, p):
    kr, ku, ks = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return kr * ku * u / (1.0 + ku * u + ks * s) ** 2

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([10.0, 1.0, 1.0]), bounds=(np.array([0.0, 1e-8, 0.0]), np.array([1e4, 1e3, 1e3])), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(row, _PARAMS))