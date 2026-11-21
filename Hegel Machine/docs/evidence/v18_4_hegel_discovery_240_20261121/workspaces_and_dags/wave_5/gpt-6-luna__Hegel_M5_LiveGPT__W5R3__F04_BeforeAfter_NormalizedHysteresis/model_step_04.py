import numpy as np
from scipy.optimize import least_squares

_tau_scale = 0.02
_params = None

def _predict_with(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / _tau_scale
    u_p = float(row['u_past']) / _tau_scale
    H = float(row['H'])
    return gain * (u_n - u_p) * np.exp(-H / max(1e-9, tau_relax)) + k_eq * u_n

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([-0.04, 0.5, 0.2]), bounds=([-10.0, 1e-5, -10.0], [10.0, 100.0, 10.0]), max_nfev=20000)
    _params = result.x
    return {"gain": float(_params[0]), "tau_relax": float(_params[1]), "k_eq": float(_params[2])}

def predict(row):
    if _params is None:
        raise ValueError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))