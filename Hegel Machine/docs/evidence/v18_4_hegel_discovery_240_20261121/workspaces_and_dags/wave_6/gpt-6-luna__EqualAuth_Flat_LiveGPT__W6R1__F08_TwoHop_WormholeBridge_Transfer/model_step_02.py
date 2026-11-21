import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    return np.array([float(row['u_now']), float(row['s_grad'])], dtype=float)

def _model(p, row):
    u, g = _features(row)
    amp, halfsat, grad_slope, offset = p
    return offset + amp * u / (halfsat + u) * math.exp(grad_slope * (1.0 - g))

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    p0 = np.array([4.0, 1.0, 0.3, 0.0], dtype=float)
    result = least_squares(lambda p: np.array([_model(p, r) for r in train_rows]) - ys,
                           p0, bounds=([-100.0, 1e-6, -10.0, -100.0], [100.0, 100.0, 10.0, 100.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    global _fitted_params
    return float(_model(_fitted_params, row))

_fitted_params = None
_original_fit = fit
def fit(train_rows):
    global _fitted_params
    _fitted_params = _original_fit(train_rows)
    return _fitted_params
