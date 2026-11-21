import numpy as np
from scipy.optimize import least_squares
import math

def _f(row, p):
    vmax, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    un = u  # tau_scale = 1
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_sqrt * math.sqrt(un)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_f(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([7.0, 0.7, 1.0]), bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
    return result.x

def predict(row):
    return float(_f(row, _params))

_params = None
# The confirmation harness fits first; keep prediction compatible with its fit output.
def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_f(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([7.0, 0.7, 1.0]), bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
    _params = result.x
    return _params
