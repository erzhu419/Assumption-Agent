import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) for r in train_rows])
    s = np.asarray([float(r['s_grad']) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u) - y
    results = [least_squares(residual, [j, b, 0.7, c], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000) for j, b, c in [(2.0, 1.0, 0.5), (4.0, 2.0, 0.0), (1.0, 0.2, 1.0)]]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    j0, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u))