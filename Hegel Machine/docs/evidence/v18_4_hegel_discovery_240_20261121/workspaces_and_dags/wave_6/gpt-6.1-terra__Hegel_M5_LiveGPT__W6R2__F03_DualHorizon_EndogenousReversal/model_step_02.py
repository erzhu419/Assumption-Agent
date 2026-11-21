import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, h):
    af, tf, ass, ts = p
    return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _model(p, u, h) - y
    best = None
    for tf in (0.5, 1.5, 4.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(residual, [2.0, tf, 3.0, ts], bounds=([0.0, 0.001, 0.0, 0.001], [100.0, 1000.0, 100.0, 1000.0]), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['H'])))
