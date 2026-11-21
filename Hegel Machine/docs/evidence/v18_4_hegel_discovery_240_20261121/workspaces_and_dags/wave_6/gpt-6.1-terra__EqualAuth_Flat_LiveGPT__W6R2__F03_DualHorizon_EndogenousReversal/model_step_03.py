import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        af, slow, tf, ts = p
        return u * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.5, 1.0, 2.0):
        for ts in (5.0, 10.0, 20.0):
            result = least_squares(residual, [2.0, 3.0, tf, ts], bounds=([0.0, 0.0, 0.001, 0.001], [100.0, 100.0, 1000.0, 1000.0]), max_nfev=5000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    af, slow, tf, ts = _params
    h = float(row['H'])
    return float(float(row['u_now']) * (af * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts))))
