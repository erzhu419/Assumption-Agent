import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.05

def _features(row):
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    base_factor = (1.0 - s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return u_n, s, base_factor

def _model(row, p):
    vmax, rho, c_loc = p
    u_n, s, _ = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows], dtype=float) - y
    lower = np.array([3.836397, 0.658333, -0.822167])
    upper = np.array([9.163604, 0.748333, 1.108834])
    p0 = np.array([6.5, 0.70, 0.15])
    result = least_squares(residual, p0, bounds=(lower, upper))
    return result.x

def predict(row):
    return float(_model(row, fit([row])))