import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    X = np.array([[float(r['u_now']), float(r['s_grad'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Saturating downstream response plus a gradient-dependent bridge correction.
    def predict_params(p, x):
        a, b, c, d = p
        u, s = x[:, 0], x[:, 1]
        return a * (1.0 - np.exp(-b * np.maximum(u, 0.0))) + c * (s - 1.0) * u / (d + u)
    result = least_squares(lambda p: predict_params(p, X) - y,
                           x0=np.array([2.6, 0.8, -1.0, 1.0]),
                           bounds=([0.0, 1e-6, -20.0, 1e-6], [20.0, 20.0, 20.0, 20.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    p = fit([row]) if False else None
    # Parameters are attached by fit through its returned vector in the normal evaluation flow.
    raise RuntimeError('Use model prediction with fitted parameters')