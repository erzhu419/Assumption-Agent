import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, tf, ass, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))) - y

    best = None
    for start in ([2, 0.7, 3, 8], [3, 1, 4, 15], [1, 0.3, 2, 5]):
        result = least_squares(residual, np.log(start), bounds=(np.log([0.001, 0.01, 0.001, 0.1]), np.log([100, 100, 100, 1000])), max_nfev=3000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, ass, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))))