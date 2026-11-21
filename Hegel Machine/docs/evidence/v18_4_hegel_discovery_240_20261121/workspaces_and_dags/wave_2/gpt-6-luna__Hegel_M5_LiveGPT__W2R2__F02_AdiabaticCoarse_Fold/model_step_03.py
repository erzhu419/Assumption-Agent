import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def predict_params(p):
        E, eta = p
        return (u / E) * (1.0 - np.exp(-E * h / eta))
    res = least_squares(lambda p: predict_params(p) - y, x0=[0.55, 1.0], bounds=([1e-6, 1e-6], [100.0, 1000.0]))
    _params = tuple(float(v) for v in res.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    E, eta = _params
    return (float(row['u_now']) / E) * (1.0 - math.exp(-E * float(row['H']) / eta))