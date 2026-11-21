import math
import numpy as np
from scipy.optimize import least_squares

_coef = None
_center = np.array([1.5, 1.0], dtype=float)
_scale = np.array([1.5, 1.0], dtype=float)

def _features(row):
    x, z = (np.array([float(row['u_1']), float(row['u_2'])]) - _center) / _scale
    return np.array([1.0, x, z, x*x, x*z, z*z, x*x*x, x*x*z, x*z*z, z*z*z], dtype=float)

def fit(train_rows):
    global _coef
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Mild ridge stabilization for the sparsely sampled bivariate surface.
    lam = 1e-5
    A = np.vstack([X, math.sqrt(lam) * np.eye(X.shape[1])])
    b = np.concatenate([y, np.zeros(X.shape[1])])
    res = least_squares(lambda p: A @ p - b, x0=np.zeros(X.shape[1]))
    _coef = res.x
    return predict

def predict(row):
    if _coef is None:
        raise RuntimeError('fit must be called before predict')
    return float(_features(row) @ _coef)
