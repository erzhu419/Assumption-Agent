import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 3.0 for r in train_rows])
    s = np.array([r['s_grad'] / (1.0 + r['s_grad']) for r in train_rows])
    y = np.array([r['y'] for r in train_rows])
    root = np.sqrt(u)

    def residual(p):
        j, beta, rho, c = p
        return j * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s) + c * root - y

    _params = least_squares(residual, [4.0, 1.0, 0.70, 0.0], bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    j, beta, rho, c = _params
    u = row['u_now'] / 3.0
    s = row['s_grad'] / (1.0 + row['s_grad'])
    return float(j * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s) + c * math.sqrt(u))