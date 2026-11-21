import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _pred(p, row):
    gain, tau_relax, k_eq = p
    un = float(row['u_now'])
    up = float(row['u_past'])
    H = float(row['H'])
    return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_pred(p, r) - float(r['y']) for r in rows], dtype=float)
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(_params, row))