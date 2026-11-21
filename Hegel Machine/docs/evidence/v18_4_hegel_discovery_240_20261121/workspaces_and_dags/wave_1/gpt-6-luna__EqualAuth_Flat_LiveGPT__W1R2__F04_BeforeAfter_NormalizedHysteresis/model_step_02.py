import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(params, row):
    gain, tau_relax, k_eq = params
    u_n = float(row['u_now'])
    u_p = float(row['u_past'])
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    def residual(z):
        gain, log_tau, k_eq = z
        params = (gain, math.exp(log_tau), k_eq)
        return np.asarray([_predict_with(params, row) - float(row['y']) for row in train_rows])
    result = least_squares(residual, np.array([1.0, 0.0, 0.0]))
    gain, log_tau, k_eq = result.x
    _PARAMS = (float(gain), math.exp(float(log_tau)), float(k_eq))
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(_PARAMS, row))