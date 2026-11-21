import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array [float(r['H']) for r in train_rows]
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        af, tf, a_s, ts = p
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    candidates = []
    for tf in (0.5, 1.0, 2.0):
        for ts in (5.0, 15.0, 30.0):
            result = least_squares(residual, [2.0, tf, 3.0, ts], bounds=([0.0, 0.01, 0.0, 0.1], [100.0, 100.0, 100.0, 1000.0]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(float(row['u_now']) * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))