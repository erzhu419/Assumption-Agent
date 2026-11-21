import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array [float(r['H']) for r in train_rows]
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        af, ts, a_s, tf = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.2, 0.5, 1.0, 2.0):
        for ts in (5.0, 15.0, 40.0):
            result = least_squares(residual, [1.5, ts, 2.5, tf], bounds=([0.0, 0.1, 0.0, 0.01], [100.0, 1000.0, 100.0, 100.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    af, ts, a_s, tf = _params
    h = float(row['H'])
    return float(row['u_now']) * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts)))
