import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        af, tf, a_s, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y
    best = None
    for tf in (0.2, 0.7, 2.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(residual, np.log([2.0, tf, 3.0, ts]), bounds=(np.log([1e-4, 0.01, 1e-4, 0.1]), np.log([100.0, 100.0, 100.0, 1000.0])), max_nfev=3000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))
