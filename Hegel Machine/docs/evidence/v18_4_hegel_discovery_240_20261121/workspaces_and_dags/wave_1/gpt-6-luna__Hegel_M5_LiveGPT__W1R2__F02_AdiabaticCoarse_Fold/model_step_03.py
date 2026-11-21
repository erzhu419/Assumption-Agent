import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    E, eta = np.exp(p)
    H = float(row['H'])
    u = float(row['u_now'])
    return (u / E) * (1.0 - math.exp(-E * H / eta))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, np.log([0.8, 0.3]), max_nfev=20000)
    _params = np.exp(result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    E, eta = _params
    return (float(row['u_now']) / E) * (1.0 - math.exp(-E * float(row['H']) / eta))