import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    H = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(z):
        E, eta = np.exp(z)
        return (u / E) * (-np.expm1(-E * H / eta)) - y
    opt = least_squares(residual, np.log([0.4, 0.05]), max_nfev=5000)
    _PARAMS = tuple(np.exp(opt.x))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    E, eta = _PARAMS
    h = float(row['H'])
    u = float(row['u_now'])
    return float((u / E) * (-math.expm1(-E * h / eta)))