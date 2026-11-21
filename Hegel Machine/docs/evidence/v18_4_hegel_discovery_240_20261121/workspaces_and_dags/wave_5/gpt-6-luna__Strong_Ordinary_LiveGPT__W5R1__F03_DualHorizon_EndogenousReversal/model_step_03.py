import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _response(p, row):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_response(p, r) for r in rows]) - y
    # Separate bounds encode the fast and slow adaptation timescales.
    result = least_squares(residual, x0=np.array([1.2, 1.0, 0.5, 12.0]),
                           bounds=(np.array([0.0, 0.0, 1e-4, 1.0]),
                                   np.array([10.0, 10.0, 5.0, 200.0])),
                           max_nfev=20000)
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_response(_PARAMS, row))