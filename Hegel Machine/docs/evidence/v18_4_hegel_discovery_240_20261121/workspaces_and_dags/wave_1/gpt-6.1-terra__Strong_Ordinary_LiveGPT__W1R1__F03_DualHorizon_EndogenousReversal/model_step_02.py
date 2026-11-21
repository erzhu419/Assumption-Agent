import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        af, slow, tf, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))) - y
    best = None
    for tf in (0.2, 0.7, 1.5):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(residual, np.log([1.5, 2.5, tf, ts]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = np.exp(best.x)

def predict(row):
    af, slow, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))))
