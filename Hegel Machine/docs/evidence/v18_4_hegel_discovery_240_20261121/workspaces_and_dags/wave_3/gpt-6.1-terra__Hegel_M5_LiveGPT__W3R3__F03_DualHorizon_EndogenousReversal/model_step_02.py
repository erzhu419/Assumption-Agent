import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, h):
    af, slow, tf, ts = p
    return u * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for tf in (0.3, 1.0, 3.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(lambda p: _evaluate(p, u, h) - y, [2.0, 3.0, tf, ts], bounds=([0.0, 0.0, 0.01, 0.1], [100.0, 100.0, 100.0, 1000.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['H'])))