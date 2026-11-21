import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    candidates = []
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(residual, [4.0, beta, 0.7, 0.2], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_value(row, _params))