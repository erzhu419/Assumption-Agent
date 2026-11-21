import numpy as np
from scipy.optimize import least_squares

def _features(row):
    return np.array([float(row['u_now']), float(row['s_grad'])], dtype=float)

def _predict_params(p, row):
    u, s = _features(row)
    A, B, k, c0, c1 = p
    return A - B * np.exp(-k * u) + (1.0 - s) * (c0 + c1 * u)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(p, r) for r in train_rows]) - ys
    p0 = np.array([2.8, 2.0, 0.8, 0.5, 0.0])
    result = least_squares(residual, p0, max_nfev=20000)
    return result.x

def predict(row):
    return float(_predict_params(fit(train_rows), row))