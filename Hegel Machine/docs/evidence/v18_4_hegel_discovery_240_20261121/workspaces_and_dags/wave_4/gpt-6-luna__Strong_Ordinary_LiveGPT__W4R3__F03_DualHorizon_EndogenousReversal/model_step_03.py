import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    result = least_squares(
        residual,
        x0=np.array([1.5, 0.4, 1.0, 8.0]),
        bounds=([0.0, 1e-4, 0.0, 1e-3], [20.0, 100.0, 20.0, 500.0]),
        max_nfev=20000
    )
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, tau_fast, amp_slow, tau_slow = _PARAMS
    u = float(row['u_now'])
    h = float(row['H'])
    return float(u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow))))