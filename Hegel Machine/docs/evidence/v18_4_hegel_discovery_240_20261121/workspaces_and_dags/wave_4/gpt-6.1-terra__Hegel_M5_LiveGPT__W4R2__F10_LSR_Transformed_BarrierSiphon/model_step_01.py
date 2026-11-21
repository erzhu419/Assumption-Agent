import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(p, r) - float(r['y']) for r in rows])
    candidates = []
    for beta in (0.2, 1.0, 3.0, 8.0):
        result = least_squares(residual, [5.0, beta, 0.7, 0.5], bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 50.0, 1.0, 20.0]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_value(_params, row))