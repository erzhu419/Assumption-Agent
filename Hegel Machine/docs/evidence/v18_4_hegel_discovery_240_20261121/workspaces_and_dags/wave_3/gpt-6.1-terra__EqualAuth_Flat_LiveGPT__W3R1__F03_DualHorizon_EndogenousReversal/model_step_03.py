import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, tf, a_s, ts = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    best = None
    for start in ([2, 0.7, 3, 10], [3, 1, 4, 20], [2, 2, 3, 5]):
        result = least_squares(residual, start, bounds=([0, 0.001, 0, 0.001], [100, 1000, 100, 1000]), max_nfev=10000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x
    return _params

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))
