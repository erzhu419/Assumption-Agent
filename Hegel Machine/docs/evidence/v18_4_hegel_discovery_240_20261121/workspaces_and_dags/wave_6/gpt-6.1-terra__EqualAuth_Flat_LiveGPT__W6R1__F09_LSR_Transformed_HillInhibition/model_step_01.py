import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha, leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * u * u / (k_half + u * u) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    candidates = []
    for k in (0.3, 1.0, 3.0):
        for alpha in (0.2, 1.0, 3.0):
            result = least_squares(residual, [4.0, k, alpha, 0.1], bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=5000)
            candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_model(row, _params))