import math
import numpy as np
from scipy.optimize import least_squares

def _basis(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (1.0 - s / (1.0 + s)) * (u / (1.2 + u)), math.sqrt(max(u, 0.0))

def fit(train_rows):
    X = np.array([_basis(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, rho, c = p
        return vmax * (X[:, 0] + (1.0-rho)*0) - vmax*rho * np.array([float(r['s_grad'])/(1.0+float(r['s_grad'])) * (float(r['u_now'])/(1.2+float(r['u_now']))) for r in train_rows]) + c*X[:, 1] - y
    result = least_squares(residual, x0=np.array([6.5, 0.4, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x.tolist()

def predict(row):
    vmax, rho, c = _MODEL
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho*s/(1.0+s)) * (u/(1.2+u)) + c*math.sqrt(max(u, 0.0)))

_MODEL = [6.0, 0.4, 0.0]
_ORIGINAL_FIT = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _ORIGINAL_FIT(train_rows)
    return _MODEL
