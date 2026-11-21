import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004
_params = None

def _features(row):
    u_now = row['u_now'] / TAU_SCALE
    u_past = row['u_past'] / TAU_SCALE
    return u_now, u_now - u_past, row['H']

def _law(params, u, step, h):
    gain, tau_relax, k_eq = params
    return k_eq * u + gain * step * np.exp(-h / tau_relax)

def fit(train_rows):
    global _params
    u, step, h = np.asarray([_features(row) for row in train_rows], dtype=float).T
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    _params = least_squares(lambda p: _law(p, u, step, h) - y,
                            [-1.0, 1.0, 0.3],
                            bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    return float(_law(_params, *_features(row)))