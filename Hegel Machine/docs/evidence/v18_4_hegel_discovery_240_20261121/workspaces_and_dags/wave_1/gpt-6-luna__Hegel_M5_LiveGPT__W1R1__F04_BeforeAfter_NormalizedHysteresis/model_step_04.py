import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, params):
    gain, tau = params
    delta = float(row['u_now']) - float(row['u_past'])
    h = float(row['H'])
    return float(row['x_now']) + gain * delta * math.exp(-h / tau)

def fit(train_rows):
    global _PARAMS
    def residual(p):
        gain, log_tau = p
        tau = math.exp(log_tau)
        return np.asarray([
            _predict_with(r, (gain, tau)) - float(r['y'])
            for r in train_rows
        ])
    result = least_squares(residual, np.array([3.0, 0.0]))
    _PARAMS = (float(result.x[0]), float(math.exp(result.x[1])))
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _predict_with(row, _PARAMS)
