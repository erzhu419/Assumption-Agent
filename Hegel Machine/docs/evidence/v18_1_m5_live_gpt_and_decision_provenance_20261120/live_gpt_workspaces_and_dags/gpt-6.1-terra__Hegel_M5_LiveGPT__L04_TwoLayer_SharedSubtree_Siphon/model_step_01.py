import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    features = [_features(row) for row in train_rows]
    targets = np.asarray [float(row['y']) for row in train_rows]
    def residual(p):
        vmax, rho, c = p
        return np.asarray([vmax * (1.0 - rho * sink) * base + c * branch for base, sink, branch in features]) - targets
    result = least_squares(residual, [6.5, 0.70, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x
    return _params

def predict(row):
    base, sink, branch = _features(row)
    vmax, rho, c = _params
    return float(vmax * (1.0 - rho * sink) * base + c * branch)
