import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 0.004
_params = None

def _predict_with(row, p):
    gain, log_tau_relax, k_eq = p
    u_n = row['u_now'] / _tau_scale
    u_p = row['u_past'] / _tau_scale
    return gain * (u_n - u_p) * math.exp(-row['H'] / math.exp(log_tau_relax)) + k_eq * u_n

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, math.log(0.5), 0.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(row, _params))