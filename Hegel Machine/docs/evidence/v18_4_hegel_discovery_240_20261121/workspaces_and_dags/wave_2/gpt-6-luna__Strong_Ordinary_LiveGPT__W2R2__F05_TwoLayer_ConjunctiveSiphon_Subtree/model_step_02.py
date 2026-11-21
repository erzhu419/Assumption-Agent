import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    base = (1.0 - s / (1.0 + s)) * (u / (1.2 + u))
    return base, math.sqrt(max(0.0, u))

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    feats = [_features(r) for r in train_rows]
    def residual(p):
        vmax, rho, c_loc = p
        return np.array([vmax * (1.0 - rho * float(r['s_grad']) / (1.0 + float(r['s_grad']))) * (float(r['u_now']) / 4.0) / (1.2 + float(r['u_now']) / 4.0) + c_loc * math.sqrt(max(0.0, float(r['u_now']) / 4.0)) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.0]), bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c_loc = _params
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(0.0, u)))