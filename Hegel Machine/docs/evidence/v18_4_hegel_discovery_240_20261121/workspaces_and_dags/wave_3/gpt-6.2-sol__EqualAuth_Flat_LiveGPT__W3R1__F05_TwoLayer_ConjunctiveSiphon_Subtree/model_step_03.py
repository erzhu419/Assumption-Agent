import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def design(k):
        f = u / (k + u)
        return np.column_stack((f, -f * sink, root))

    def residual(p):
        return design(p[0]) @ p[1:] - y

    k0 = max(float(np.median(u)), 1e-6)
    beta0 = np.linalg.lstsq(design(k0), y, rcond=None)[0]
    result = least_squares(residual, np.r_[k0, beta0], bounds=([1e-9, -np.inf, -np.inf, -np.inf], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    k, a, b, c = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float((a - b * s / (1.0 + s)) * u / (k + u) + c * math.sqrt(u))