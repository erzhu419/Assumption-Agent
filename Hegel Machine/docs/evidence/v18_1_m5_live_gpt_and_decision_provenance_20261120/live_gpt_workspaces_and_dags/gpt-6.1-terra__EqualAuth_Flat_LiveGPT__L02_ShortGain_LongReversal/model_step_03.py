import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, h):
    af, slow, tf, ts = np.exp(p)
    return u * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.array [r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _response(p, u, h) - y
    best = None
    for tf in (0.2, 0.7, 1.5):
        for ts in (3.0, 10.0, 25.0):
            start = np.log([2.0, 2.0, tf, ts])
            result = least_squares(residual, start, bounds=(np.log([0.001, 0.001, 0.01, 0.1]), np.log([100.0, 100.0, 100.0, 1000.0])), max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_response(_params, float(row['u_now']), float(row['H'])))