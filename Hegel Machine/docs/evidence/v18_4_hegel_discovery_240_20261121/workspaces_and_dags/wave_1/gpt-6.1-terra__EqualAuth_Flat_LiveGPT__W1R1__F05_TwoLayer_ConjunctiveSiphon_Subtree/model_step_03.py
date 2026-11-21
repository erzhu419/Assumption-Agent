import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    k = np.exp(p[0])
    z = u / (k + u)
    g = s / (1.0 + s)
    return z * (p[1] + p[2] * g + p[3] * g * g) + p[4] * z * z * (1.0 - g)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    best = None
    for k in (2.0, 4.8, 10.0):
        result = least_squares(residual, [np.log(k), 5.0, -3.5, 0.0, 0.0], bounds=([np.log(0.1), -50, -50, -50, -50], [np.log(100), 50, 50, 50, 50]), max_nfev=3000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(row, _params))