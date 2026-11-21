import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        af, tf, a_s, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y
    best = None
    for start in ([2, 1, 2, 10], [1, 0.5, 2, 15], [3, 2, 3, 20]):
        result = least_squares(residual, np.log(start), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))