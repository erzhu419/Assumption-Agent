import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row['u_now']), 0.0)  # tau_scale = 1
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    vmax, rho, c_loc = p
    u, s = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    p0 = np.array([6.5, 0.70, 0.0])
    lo = np.array([3.836395, 0.658333, -0.822167])
    hi = np.array([9.163605, 0.748333, 1.108834])
    result = least_squares(residual, p0, bounds=(lo, hi))
    return result.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict expects fitted parameters; use fit(train_rows) first')
    return float(_model(row, p))