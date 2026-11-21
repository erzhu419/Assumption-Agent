import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return u, g

def _model(p, row):
    u, g = _features(row)
    a, b, c, d, e, f = p
    return a + b * u / (c + u) + d * (g - 1.0) * u + e * (g - 1.0) ** 2 * u + f * (g - 1.0) * u * u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    p0 = np.array([0.0, 6.0, 2.0, 0.0, 0.0, 0.0])
    result = least_squares(residual, p0, bounds=([-20, -100, 1e-4, -20, -20, -20], [20, 100, 100, 20, 20, 20]), max_nfev=20000)
    return result.x

def predict(row):
    raise RuntimeError('predict requires fitted parameters; use predict_with_fit(fit(train_rows), row)')

def predict_with_fit(params, row):
    return float(_model(params, row))