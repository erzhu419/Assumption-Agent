import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return u, s


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def pred(p, r):
        u, s = _features(r)
        # Saturating base-field response plus a gradient-dependent gain and
        # a linear sink contribution above the reference gradient.
        return (p[0] + p[1] * s) * u / (p[2] + u) + p[3] * math.sqrt(u) + p[4] * max(0.0, s - 1.0) * u
    def residual(p):
        return np.asarray([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.0, 1.0, 0.5, -0.1]), max_nfev=20000)
    return result.x.tolist()


def predict(row):
    p = fit(_train_rows)
    u, s = _features(row)
    return float((p[0] + p[1] * s) * u / (p[2] + u) + p[3] * math.sqrt(u) + p[4] * max(0.0, s - 1.0) * u)
