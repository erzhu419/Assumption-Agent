import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        af, slow, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))) - y
    best = None
    for guess in ([2, 3, 1, 10], [1.5, 2, 0.5, 5], [3, 4, 2, 20], [1, 2, 0.3, 15]):
        result = least_squares(residual, guess, bounds=([0, 0, 0.001, 0.01], [100, 100, 100, 1000]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    af, slow, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))))
