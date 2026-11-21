import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.02
_params = None

def _model(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / _TAU_SCALE
    u_p = float(row['u_past']) / _TAU_SCALE
    return gain * (u_n - u_p) * math.exp(-float(row['H']) / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.1]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))