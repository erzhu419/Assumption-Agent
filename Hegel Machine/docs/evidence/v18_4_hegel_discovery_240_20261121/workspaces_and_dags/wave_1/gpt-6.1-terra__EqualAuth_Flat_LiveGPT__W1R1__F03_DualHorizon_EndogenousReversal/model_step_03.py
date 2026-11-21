import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        af, tf, ass, ts = p
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))) - y
    best = None
    for start in ([2, 1, 3, 10], [3, 1, 4, 20], [2, 0.5, 2, 5]):
        result = least_squares(residual, start, bounds=([0, 0.001, 0, 0.001], [100, 1000, 100, 1000]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    af, tf, ass, ts = _params
    h = float(row['H'])
    return float(float(row['u_now']) * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))))
