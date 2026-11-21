import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = row['u_now'] / 0.004
    u_p = row['u_past'] / 0.004
    return gain * (u_n - u_p) * np.exp(-row['H'] / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    ys = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.4]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))