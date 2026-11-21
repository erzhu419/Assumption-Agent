import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    a, b, c, d, e, f = p
    return (a * u + b + c * s) / (1.0 + np.exp(d) * u + np.exp(e) * s + np.exp(f) * u * s)

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 0.0, 0.0, 0.0, -1.0, -1.0], [3.0, 0.0, 0.0, -1.0, -1.0, -1.0], [5.0, 0.5, 0.0, 0.0, 0.0, -2.0]):
        result = least_squares(residual, start, max_nfev=5000, bounds=([-100, -100, -100, -12, -12, -12], [100, 100, 100, 5, 5, 5]))
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))