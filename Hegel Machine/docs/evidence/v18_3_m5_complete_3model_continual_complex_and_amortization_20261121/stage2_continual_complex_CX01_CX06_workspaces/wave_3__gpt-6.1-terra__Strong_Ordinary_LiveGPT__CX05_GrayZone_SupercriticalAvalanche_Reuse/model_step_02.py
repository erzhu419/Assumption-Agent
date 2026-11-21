import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    bounds = ([0.0, 0.0, -20.0, 0.0, 1.3], [100.0, 1.0, 20.0, 20.0, 1.5])
    results = [least_squares(residual, [v, 0.7, c, k, t], bounds=bounds, max_nfev=5000) for v in (5.0, 10.0) for c in (0.5, 2.0) for k in (1.0, 3.0) for t in (1.32, 1.45)]
    _params = min(results, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_model(row, _params))