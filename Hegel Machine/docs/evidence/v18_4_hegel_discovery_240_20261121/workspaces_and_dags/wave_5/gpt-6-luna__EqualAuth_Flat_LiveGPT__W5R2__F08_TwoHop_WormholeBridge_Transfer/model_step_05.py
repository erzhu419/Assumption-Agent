import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row, p):
    a, b, c, d, e, f = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    # Saturating source transfer with gradient-dependent bridge gain.
    z = math.exp(max(-30.0, min(30.0, -d * s)))
    return a + b*u/(c + u + 1e-9) + e*z + f*u*z

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_features(r, p) for r in train_rows]) - ys
    best = None
    for p0 in ([0.0, 3.0, 1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 2.0, 0.3, 2.0, -0.3], [0.0, 4.0, 3.0, 0.8, 3.0, -1.0]):
        result = least_squares(residual, p0, max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row, _PARAMS))