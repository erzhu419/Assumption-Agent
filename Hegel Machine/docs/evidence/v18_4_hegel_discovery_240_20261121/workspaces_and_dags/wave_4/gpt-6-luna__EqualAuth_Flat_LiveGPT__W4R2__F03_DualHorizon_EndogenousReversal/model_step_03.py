import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    ys = np.asarray([float(r['y']) for r in rows], dtype=float)
    def unpack(z):
        return (z[0], z[1], math.exp(z[2]), math.exp(z[3]))
    def residual(z):
        p = unpack(z)
        return np.asarray([_predict_params(r, p) for r in rows]) - ys
    best = None
    for start in ([1.0, 1.0, math.log(0.5), math.log(10.0)], [1.5, 1.0, math.log(0.3), math.log(20.0)], [2.0, 1.2, math.log(1.0), math.log(15.0)]):
        result = least_squares(residual, np.asarray(start, dtype=float), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, unpack(result.x))
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params))