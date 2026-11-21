import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, h):
    af, tf, ass, ts = p
    return af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return u * _response(p, h) - y
    best = None
    for start in ([2, 1, 3, 10], [3, 0.5, 4, 15], [1, 2, 2, 20]):
        result = least_squares(residual, start, bounds=([0, 0.001, 0, 0.001], [100, 1000, 100, 1000]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(row['u_now'] * _response(_params, float(row['H'])))