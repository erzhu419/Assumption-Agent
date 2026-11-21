import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(q):
        af, ts, a_s, tf = np.exp(q)
        pred = u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts)))
        return pred - y

    best = None
    for tf in (0.3, 1.0, 3.0):
        for ts in (5.0, 15.0, 40.0):
            start = np.log([1.5, ts, 2.5, tf])
            result = least_squares(residual, start, bounds=(np.log([1e-5, 0.05, 1e-5, 0.01]), np.log([100.0, 1000.0, 100.0, 100.0])), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = np.exp(best.x)

def predict(row):
    af, ts, a_s, tf = _params
    h = float(row['H'])
    return float(float(row['u_now']) * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))
