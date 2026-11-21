import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        af, tf, ass, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))) - y
    best = None
    for tf in (0.3, 1.0, 2.0):
        for ts in (5.0, 15.0, 30.0):
            result = least_squares(residual, np.log([2.0, tf, 3.0, ts]), max_nfev=3000)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, ass, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))))