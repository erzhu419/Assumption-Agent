import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(p, row):
    a, b, c, d = p
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return a + b * u / (c + u) + d * (1.0 - g) * u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_predict_params(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.0, 4.0, 1.0, -0.5]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    # Parameters are fitted from train_rows by fit(); callers should retain the returned vector.
    p = fit(_TRAIN_ROWS)
    return float(_predict_params(p, row))