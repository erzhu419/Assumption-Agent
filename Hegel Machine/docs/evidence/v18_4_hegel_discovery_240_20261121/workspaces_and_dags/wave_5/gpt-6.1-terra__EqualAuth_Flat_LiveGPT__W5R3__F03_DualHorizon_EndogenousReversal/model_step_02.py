import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        af, a_s, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y
    best = None
    for tf in (0.2, 0.7, 2.0):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(residual, [2.0, 3.0, tf, ts], bounds=([0.0, 0.0, 0.001, 0.01], [100.0, 100.0, 100.0, 1000.0]), max_nfev=5000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x
    return _params

def predict(row):
    af, a_s, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-math.expm1(-h / tf)) - a_s * (-math.expm1(-h / ts))))