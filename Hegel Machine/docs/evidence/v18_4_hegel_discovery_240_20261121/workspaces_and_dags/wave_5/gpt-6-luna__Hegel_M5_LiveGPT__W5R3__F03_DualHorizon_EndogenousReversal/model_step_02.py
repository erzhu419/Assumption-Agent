import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Log parameterization enforces positive amplitudes and fast < slow time constant.
    def unpack(z):
        af, tf, aslow, delta = np.exp(np.clip(z, -20.0, 20.0))
        return af, tf, aslow, tf + delta
    def residual(z):
        p = unpack(z)
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        np.log([1.5, 0.5, 1.0, 5.0]),
        np.log([2.0, 1.0, 1.5, 10.0]),
        np.log([1.0, 0.2, 0.8, 20.0]),
        np.log([3.0, 2.0, 2.5, 30.0]),
    ]
    best = None
    for z0 in starts:
        result = least_squares(residual, z0, max_nfev=50000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = unpack(best[1])
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))