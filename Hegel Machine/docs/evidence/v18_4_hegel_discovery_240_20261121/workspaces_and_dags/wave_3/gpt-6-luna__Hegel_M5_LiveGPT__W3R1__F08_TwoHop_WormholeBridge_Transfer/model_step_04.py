import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    # Use normalized inputs and a smooth rational surface; the other inputs
    # are constant in the paid observations and therefore not identifiable.
    return u, s

def _value(p, row):
    u, s = _features(row)
    a, b, c, d, e, f = p
    den = 1.0 + abs(d) * u + abs(e) * s
    return (a*u + b*u*u + c*s + f*u*s) / den

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - y
    starts = [np.array([1.0, 0.0, 0.0, 0.2, 0.2, 0.0]),
              np.array([2.0, -0.2, 0.2, 0.3, 0.1, 0.1])]
    fits = [least_squares(residual, p, max_nfev=10000) for p in starts]
    _params = min(fits, key=lambda z: np.sum(z.fun*z.fun)).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_params, row))