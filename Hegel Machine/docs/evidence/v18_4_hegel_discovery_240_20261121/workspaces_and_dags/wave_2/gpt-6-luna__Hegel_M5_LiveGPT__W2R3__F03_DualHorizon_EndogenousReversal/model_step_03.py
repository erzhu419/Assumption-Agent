import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def predict_params(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    result = least_squares(
        residual,
        x0=np.array([1.0, 1.0, 1.0, 10.0]),
        bounds=(np.array([0.0, 1e-6, 0.0, 1e-6]), np.array([100.0, 1e3, 100.0, 1e3])),
        max_nfev=50000
    )
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, tau_fast, amp_slow, tau_slow = _PARAMS
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow))))