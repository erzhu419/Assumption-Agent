import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return np.array([1.0, math.sqrt(u), u, s, s*s, u*s], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return X @ p - y
    p0 = np.linalg.lstsq(X, y, rcond=None)[0]
    return least_squares(residual, p0, max_nfev=20000).x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_features(row) @ p)