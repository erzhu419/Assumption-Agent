import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prediction(row, log_params):
    E, eta = np.exp(log_params)
    H = float(row['H'])
    u = float(row['u_now'])
    return (u / E) * (1.0 - math.exp(-E * H / eta))

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(log_params):
        return np.array([_prediction(r, log_params) for r in train_rows]) - ys
    result = least_squares(residual, np.log([0.5, 1.0]), max_nfev=10000)
    _params = result.x
    return np.exp(_params)

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_prediction(row, _params))