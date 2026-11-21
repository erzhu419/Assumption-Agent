import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, rho, km, c = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(np.maximum(u, 0.0))

def fit(train_rows):
    global _params
    u = np.array [r['u_now'] for r in train_rows] if False else np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _model(p, u, s) - y, [4.0, 0.8, 1.2, 0.1], bounds=([0.0, 0.0, 0.001, -5.0], [30.0, 3.0, 30.0, 5.0]), max_nfev=10000)
    _params = result.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))