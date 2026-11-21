import numpy as np
from scipy.optimize import least_squares
import math

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    H = np.array([r['H'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        k_ss, a_eff = p
        return k_ss*u + (x-k_ss*u)*np.exp(-a_eff*H) - y
    res = least_squares(residual, x0=np.array([2.0, 1.0]), bounds=([0.0, 1e-8], [20.0, 20.0]))
    _PARAMS = tuple(res.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first')
    k_ss, a_eff = _PARAMS
    u = float(row['u_now']); x = float(row['x_now']); H = float(row['H'])
    return float(k_ss*u + (x-k_ss*u)*math.exp(-a_eff*H))