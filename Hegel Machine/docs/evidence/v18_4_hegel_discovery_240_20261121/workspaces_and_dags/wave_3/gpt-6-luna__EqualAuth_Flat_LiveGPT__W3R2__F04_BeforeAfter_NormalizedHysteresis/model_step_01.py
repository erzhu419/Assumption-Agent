import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    return (float(row['u_now']), float(row['u_now']) - float(row['u_past']), float(row['H']))

def _predict_params(row, params):
    gain, tau_relax, k_eq = params
    u_now, delta, H = _features(row)
    return gain * delta * np.exp(-H / tau_relax) + k_eq * u_now

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 0.8, 0.64]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(row, _PARAMS))