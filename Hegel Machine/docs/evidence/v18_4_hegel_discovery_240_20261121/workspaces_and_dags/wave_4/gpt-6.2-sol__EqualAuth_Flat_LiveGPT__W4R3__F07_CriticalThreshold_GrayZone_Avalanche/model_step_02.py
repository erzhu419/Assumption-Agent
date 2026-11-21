import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _design(u, s, threshold):
    base = u / (1.2 + u)
    return np.column_stack((base, -base * s / (1.0 + s), np.sqrt(u), -u * np.maximum(0.0, s - threshold)))

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    lo, hi = 2.44, 2.78
    t0 = (lo + hi) / 2.0
    coef = np.linalg.lstsq(_design(u, s, t0), y, rcond=None)[0]
    result = least_squares(lambda p: _design(u, s, p[4]) @ p[:4] - y, np.r_[coef, t0], bounds=([-np.inf] * 4 + [lo], [np.inf] * 4 + [hi]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    a, b, c, k, threshold = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float((a - b * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u) - k * u * max(0.0, s - threshold))