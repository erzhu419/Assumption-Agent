import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(rows, threshold):
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    base = u / (1.2 + u)
    return np.column_stack((base, -base * s / (1.0 + s), np.sqrt(u), -np.maximum(0.0, s - threshold) * u))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [ ] if False else np.array([float(r['y']) for r in rows])
    def residual(p):
        return _features(rows, p[4]) @ p[:4] - y
    best = None
    for threshold in (2.50, 2.56, 2.62, 2.68):
        coeff = np.linalg.lstsq(_features(rows, threshold), y, rcond=None)[0]
        result = least_squares(residual, np.r_[coeff, threshold], bounds=([-np.inf]*4 + [2.50], [np.inf]*4 + [2.68]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float((_features([row], _params[4]) @ _params[:4])[0])
