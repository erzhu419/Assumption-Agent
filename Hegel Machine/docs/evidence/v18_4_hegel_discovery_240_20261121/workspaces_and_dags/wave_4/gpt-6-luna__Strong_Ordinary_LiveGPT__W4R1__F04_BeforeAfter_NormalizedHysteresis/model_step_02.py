import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    return (float(row['u_now']) - float(row['u_past']), float(row['u_now']), float(row['H']))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        gain, tau_relax, k_eq = p
        return gain * X[:, 0] * np.exp(-X[:, 2] / tau_relax) + k_eq * X[:, 1] - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.1]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    gain, tau_relax, k_eq = _PARAMS
    du, u_now, H = _features(row)
    return float(gain * du * np.exp(-H / tau_relax) + k_eq * u_now)
