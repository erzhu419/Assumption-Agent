import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a, b, c, d, e = p
        return e + a * u / (b + u) - c * s / (d + s) - y

    best = None
    for b in (0.3, 1.0, 3.0):
        for d in (0.3, 1.0, 3.0):
            result = least_squares(residual, [4.0, b, 2.0, d, 0.0],
                                   bounds=([0.0, 1e-5, 0.0, 1e-5, -20.0],
                                           [100.0, 100.0, 100.0, 100.0, 20.0]),
                                   max_nfev=3000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    a, b, c, d, e = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(e + a * u / (b + u) - c * s / (d + s))
