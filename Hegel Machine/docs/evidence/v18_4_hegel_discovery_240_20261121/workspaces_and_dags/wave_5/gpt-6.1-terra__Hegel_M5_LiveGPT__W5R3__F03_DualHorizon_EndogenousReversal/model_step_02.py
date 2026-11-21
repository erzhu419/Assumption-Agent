import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        af, a_s, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.2, 0.5, 1.0, 2.0):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(residual, [1.5, 2.0, tf, ts], bounds=([0.0, 0.0, 0.001, 0.01], [100.0, 100.0, 100.0, 1000.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    af, a_s, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now']) * (af * (-math.expm1(-h / tf)) - a_s * (-math.expm1(-h / ts)))
