import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with_params(row, params):
    gain, tau_relax, k_eq = params
    u_n = row['u_now'] / 0.02
    u_p = row['u_past'] / 0.02
    return gain * (u_n - u_p) * math.exp(-row['H'] / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _params
    def residual(z):
        params = (z[0], math.exp(z[1]), z[2])
        return np.array([_predict_with_params(r, params) - r['y'] for r in train_rows])
    result = least_squares(residual, np.array([1.0, math.log(0.02), 0.0]))
    _params = (float(result.x[0]), float(math.exp(result.x[1])), float(result.x[2]))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with_params(row, _params))