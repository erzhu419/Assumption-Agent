import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return u, s

def _predict_params(p, row):
    u, s = _features(row)
    vmax, km, c_sqrt, g_lin, g_cross = p
    return vmax * u / (km + u) + c_sqrt * math.sqrt(u) + g_lin * (s - 1.0) + g_cross * (s - 1.0) * u

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(p, r) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([4.0, 0.8, 0.8, 0.0, 0.0]), bounds=([-100.0, 1e-6, -100.0, -100.0, -100.0], [100.0, 100.0, 100.0, 100.0, 100.0]))
    return result.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else row['_fitted_params']
    return float(_predict_params(p, row))