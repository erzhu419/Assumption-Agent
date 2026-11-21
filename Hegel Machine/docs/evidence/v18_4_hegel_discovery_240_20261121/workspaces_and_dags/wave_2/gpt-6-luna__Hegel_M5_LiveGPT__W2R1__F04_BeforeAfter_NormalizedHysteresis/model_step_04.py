import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.asarray([r['x_now'] for r in train_rows], dtype=float)
    d = np.asarray([r['u_past'] - r['u_now'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        alpha, amp, rate = p
        return alpha*x + amp*d*np.exp(-rate*h) - y
    result = least_squares(residual, x0=np.array([1.0, 2.0, 1.5]), bounds=([-10.0, -20.0, 0.0], [10.0, 20.0, 20.0]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    alpha, amp, rate = _params
    return float(alpha*row['x_now'] + amp*(row['u_past']-row['u_now'])*math.exp(-rate*row['H']))