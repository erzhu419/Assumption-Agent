import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, a_s, tf, delta = np.exp(p)
        ts = tf + delta
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.2, 0.8, 2.0):
        for ts in (5.0, 15.0, 40.0):
            p0 = np.log([2.0, 3.0, tf, ts - tf])
            result = least_squares(residual, p0, bounds=(np.log([1e-5, 1e-5, 1e-3, 1e-3]), np.log([1e3, 1e3, 1e3, 1e4])), max_nfev=3000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    af, a_s, tf, delta = np.exp(best.x)
    _params = (af, a_s, tf, tf + delta)
    return _params

def predict(row):
    af, a_s, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))