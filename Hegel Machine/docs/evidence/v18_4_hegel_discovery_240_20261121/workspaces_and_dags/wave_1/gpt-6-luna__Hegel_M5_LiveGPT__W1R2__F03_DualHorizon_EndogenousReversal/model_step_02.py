import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, z):
    amp_fast, amp_slow, tau_fast, tau_slow = np.exp(z)
    h = float(row['H'])
    return float(row['u_now']) * (
        amp_fast * (-math.expm1(-h / tau_fast))
        - amp_slow * (-math.expm1(-h / tau_slow))
    )

def fit(train_rows):
    global _params
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    # Numerically fit the four positive dual-exponential parameters in log space.
    def residual(z):
        return np.array([_response(r, z) for r in rows]) - ys
    best = None
    for start in ([1.0, 0.5, 0.5, 10.0], [1.0, 0.5, 1.0, 20.0], [2.0, 1.0, 0.2, 15.0]):
        result = least_squares(
            residual, np.log(start),
            bounds=(np.log([1e-5, 1e-5, 1e-4, 1e-3]), np.log([1e3, 1e3, 1e3, 1e4])),
            max_nfev=20000
        )
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x.copy()
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _response(row, _params)
