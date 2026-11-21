import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, h):
    af, tf, a_s, ts = p
    return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for tf in (0.3, 1.0, 3.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(lambda p: _values(p, u, h) - y,
                [2.0, tf, 3.0, ts],
                bounds=([0.0, 0.01, 0.0, 0.1], [100.0, 100.0, 100.0, 1000.0]),
                max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['H'])))