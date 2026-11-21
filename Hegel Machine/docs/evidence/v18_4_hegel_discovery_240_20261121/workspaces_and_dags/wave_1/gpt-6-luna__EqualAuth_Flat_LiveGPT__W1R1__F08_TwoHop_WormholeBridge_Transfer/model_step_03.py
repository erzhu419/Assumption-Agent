import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _model(p, row):
    u, s = _features(row)
    a, b, c, d, e, f = p
    sat = u / (c + u)
    return a + b * sat + d * (s - 1.0) / (1.0 + e * s) + f * (s - 1.0) * sat

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(p, r) for r in rows]) - y
    p0 = np.array([0.0, 6.0, 1.0, 0.0, 0.1, 0.0])
    result = least_squares(residual, p0, max_nfev=20000)
    return result.x

def predict(row):
    return float(_model(fit(_TRAIN_ROWS), row))