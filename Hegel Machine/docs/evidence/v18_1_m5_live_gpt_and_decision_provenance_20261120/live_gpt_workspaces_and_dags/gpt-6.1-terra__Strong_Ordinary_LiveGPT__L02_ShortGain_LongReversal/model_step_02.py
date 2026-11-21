import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, h):
    af, a_s, tf, ts = np.exp(p)
    return af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))

def fit(train_rows):
    global _params
    h = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return u * _response(p, h) - y
    best = None
    for tf in (0.1, 0.3, 1.0):
        for ts in (2.0, 10.0, 30.0):
            p0 = np.log([1.5, 2.0, tf, ts])
            result = least_squares(residual, p0, bounds=(np.log([0.001, 0.001, 0.005, 0.1]), np.log([100.0, 100.0, 100.0, 1000.0])), max_nfev=4000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(float(row['u_now']) * _response(_params, float(row['H'])))