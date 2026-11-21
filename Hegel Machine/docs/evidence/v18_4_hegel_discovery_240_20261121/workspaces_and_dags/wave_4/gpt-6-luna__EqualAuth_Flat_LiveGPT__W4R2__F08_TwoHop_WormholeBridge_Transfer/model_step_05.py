import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _features(p, row):
    a, b, c, d, e, f = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    # Saturating input response with a gradient-dependent exponential branch.
    return a + b*u/(c + u) + d*s*math.exp(-e*u) + f*s*u/(1.0 + u*u)

def fit(train_rows):
    global _P
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_features(p, r) for r in train_rows]) - y
    starts = [np.array([0.0, 4.0, 1.0, 0.5, 0.3, 0.1]), np.array([1.0, 3.0, 0.5, 1.0, 0.1, -0.1])]
    sols = [least_squares(residual, q, max_nfev=5000) for q in starts]
    _P = min(sols, key=lambda z: np.sum(z.fun*z.fun)).x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_features(_P, row))