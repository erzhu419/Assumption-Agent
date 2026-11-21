import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    amp_fast, amp_slow, tau_fast, tau_gap = p
    tau_slow = tau_fast + tau_gap
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    # Positive amplitudes and ordered positive time constants; optimize in log space.
    def unpack(z):
        return np.exp(z)
    def residual(z):
        p = unpack(z)
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    starts = [np.log([1.0, 0.5, 0.5, 8.0]), np.log([1.5, 0.8, 0.2, 15.0]), np.log([2.0, 0.7, 1.0, 20.0])]
    best = None
    for z0 in starts:
        res = least_squares(residual, z0, max_nfev=30000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    p = unpack(best.x)
    return {'params': p.tolist()}

def predict(row):
    return float(_predict_params(row, np.asarray(_model['params'], dtype=float)))

_model = {'params': [1.0, 0.5, 0.5, 8.0]}
_original_fit = fit
def fit(train_rows):
    global _model
    _model = _original_fit(train_rows)
    return _model
