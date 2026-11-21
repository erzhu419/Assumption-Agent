import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    dg = g - 1.0
    return np.array([u, u**4, dg*u, dg*u**2], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([1.0, -0.01, 0.0, 0.0]))
    global _model_params
    _model_params = result.x
    return result.x

def predict(row) -> float:
    return float(_features(row) @ _model_params)
