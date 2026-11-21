import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return u, g

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    g = np.asarray([float(r['s_grad']) for r in train_rows], dtype=float)
    # Saturating supply response plus a linear gradient-transfer contribution.
    def residual(p):
        offset, amp, rate, grad = p
        pred = offset + amp * (1.0 - np.exp(-rate * u)) + grad * (1.0 - g)
        return pred - y
    result = least_squares(residual, x0=np.array([0.0, 5.0, 0.5, 0.5]), bounds=([-100.0, -100.0, 1e-5, -100.0], [100.0, 100.0, 20.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    offset, amp, rate, grad = _PARAMS
    u, g = _features(row)
    return float(offset + amp * (1.0 - math.exp(-rate * u)) + grad * (1.0 - g))