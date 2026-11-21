import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = np.exp(p)
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in rows]) - y
    starts = [np.log([1.0, 0.5, 0.5, 5.0]), np.log([2.0, 1.0, 1.0, 10.0]), np.log([0.5, 0.2, 0.3, 20.0])]
    sols = [least_squares(residual, s, max_nfev=5000) for s in starts]
    p = min(sols, key=lambda z: np.sum(z.fun ** 2)).x
    params = np.exp(p)
    return {'params': params}

def predict(row):
    p = np.log(np.asarray(_MODEL['params'], dtype=float))
    return float(_predict_params(row, p))

_MODEL = {'params': np.array([1.0, 1.0, 1.0, 1.0])}
_FIT_ORIGINAL = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _FIT_ORIGINAL(train_rows)
    return _MODEL
