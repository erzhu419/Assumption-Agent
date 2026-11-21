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
        af, ass, tf, gap = p
        ts = tf + gap
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.3, 1.0, 3.0):
        for ts in (8.0, 20.0, 50.0):
            result = least_squares(residual, [2.0, 3.0, tf, ts - tf], bounds=([0.0, 0.0, 0.001, 0.001], [100.0, 100.0, 100.0, 1000.0]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    af, ass, tf, gap = _params
    h = float(row['H'])
    return float(row['u_now']) * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / (tf + gap))))
