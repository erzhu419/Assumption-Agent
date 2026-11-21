import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    j, beta, rho, c = p
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residuals(p):
        return np.array([_value(p, row) - float(row['y']) for row in rows])
    _params = least_squares(residuals, [2.0, 1.0, 0.5, 0.0], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    return float(_value(_params, row))