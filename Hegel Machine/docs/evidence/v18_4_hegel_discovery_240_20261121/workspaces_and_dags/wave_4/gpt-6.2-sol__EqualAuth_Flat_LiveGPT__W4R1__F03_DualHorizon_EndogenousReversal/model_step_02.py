import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _unpack(q):
    a, b, fast, gap = np.exp(q)
    return a, b, fast, fast + gap

def fit(train_rows):
    global _params
    u, h, y = np.asarray([[r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(q):
        a, b, fast, slow = _unpack(q)
        return u * (-a * np.expm1(-h / fast) + b * np.expm1(-h / slow)) - y
    q0 = np.log([1.5, 2.0, 0.5, max(1.0, 0.6 * h.max())])
    result = least_squares(residual, q0, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = _unpack(result.x)

def predict(row):
    a, b, fast, slow = _params
    h = float(row['H'])
    return float(row['u_now']) * (-a * math.expm1(-h / fast) + b * math.expm1(-h / slow))