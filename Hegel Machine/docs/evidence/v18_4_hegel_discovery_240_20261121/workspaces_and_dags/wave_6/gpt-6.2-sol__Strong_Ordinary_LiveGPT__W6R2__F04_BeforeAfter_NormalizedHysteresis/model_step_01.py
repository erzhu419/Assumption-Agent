import math
import numpy as np
from scipy.optimize import least_squares

SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    u, d, h, y = np.array([(r['u_now']/SCALE, (r['u_now']-r['u_past'])/SCALE, r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        gain, rate, k_eq = p
        return k_eq*u + gain*d*np.exp(-rate*h) - y
    _params = least_squares(residual, [2.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    gain, rate, k_eq = _params
    u = row['u_now']/SCALE
    d = (row['u_now']-row['u_past'])/SCALE
    return float(k_eq*u + gain*d*math.exp(-rate*row['H']))
