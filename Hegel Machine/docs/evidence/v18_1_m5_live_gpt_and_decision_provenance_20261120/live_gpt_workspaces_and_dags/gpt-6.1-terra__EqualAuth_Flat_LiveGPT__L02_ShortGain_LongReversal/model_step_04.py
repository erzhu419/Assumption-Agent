import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, tf, a_s, ts = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.1, 0.3, 0.8, 1.5):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(residual, [1.5, tf, 2.0, ts], bounds=([0.0, 0.01, 0.0, 0.1], [100.0, 10.0, 100.0, 200.0]), max_nfev=4000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-math.expm1(-h / tf)) - a_s * (-math.expm1(-h / ts))))
