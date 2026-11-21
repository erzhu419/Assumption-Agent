import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, h):
    a, b, tf, gap = p
    ts = tf + gap
    return u * (a * (-np.expm1(-h / tf)) - b * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for tf in (0.5, 1.0, 2.0):
        for ts in (5.0, 12.0, 25.0):
            result = least_squares(lambda p: _evaluate(p, u, h) - y,
                [2.0, 3.0, tf, ts - tf],
                bounds=([0.0, 0.0, 0.001, 0.001], [100.0, 100.0, 100.0, 1000.0]),
                max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['H'])))