import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return u, s, H

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    u = np.asarray([float(r['u_now']) for r in rows], dtype=float)
    s = np.asarray([float(r['s_grad']) for r in rows], dtype=float)
    # Saturating present-input response with a gradient-dependent offset and slope.
    def pred(p, uu, ss):
        a, b, k, c, d = p
        z = np.clip(k * uu, -50.0, 50.0)
        return a + b * (1.0 - np.exp(-z)) + (ss - 1.0) * (c + d * uu)
    p0 = np.array([0.5, 3.0, 0.8, 0.0, 0.0], dtype=float)
    result = least_squares(lambda p: pred(p, u, s) - y, p0, max_nfev=20000)
    return result.x.tolist()

def predict(row):
    p = fit(_TRAIN_ROWS)
    u, s, _H = _features(row)
    a, b, k, c, d = p
    return float(a + b * (1.0 - math.exp(-max(-50.0, min(50.0, k * u)))) + (s - 1.0) * (c + d * u))
