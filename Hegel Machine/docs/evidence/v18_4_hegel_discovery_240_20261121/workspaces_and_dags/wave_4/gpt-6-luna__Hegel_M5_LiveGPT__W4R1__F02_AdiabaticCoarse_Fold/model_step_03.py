import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def fit(train_rows):
    global _MODEL
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        E, eta = np.exp(p)
        pred = (u / E) * (1.0 - np.exp(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.5, 1.0]), max_nfev=20000)
    E, eta = np.exp(result.x)
    _MODEL = {'E_kv': float(E), 'eta_kv': float(eta)}
    return _MODEL

def predict(row):
    if _MODEL is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    E = _MODEL['E_kv']
    eta = _MODEL['eta_kv']
    u = float(row['u_now'])
    H = float(row['H'])
    return float((u / E) * (1.0 - math.exp(-E * H / eta)))