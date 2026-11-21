import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    indoor = float(row['u_indoor'])
    foot = float(row['a_foot'])
    impact = float(row['impact_idx'])
    x = np.array([outdoor, indoor, foot, impact], dtype=float)
    return np.array([1.0, *x, *(x*x), x[0]*x[1], x[0]*x[2], x[0]*x[3], x[1]*x[2], x[1]*x[3], x[2]*x[3]], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Ridge-stabilized least squares reduces the unstable extrapolation of an
    # underdetermined quadratic while retaining a flexible low-order response.
    scale = np.maximum(np.linalg.norm(X, axis=0), 1.0)
    Z = X / scale
    def residual(c):
        return np.concatenate([Z @ c - y, 0.05 * c[1:]])
    c = least_squares(residual, np.zeros(X.shape[1])).x
    _COEF = c / scale
    return predict

def predict(row):
    return float(_features(row) @ _COEF)
