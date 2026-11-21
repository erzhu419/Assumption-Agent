import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _value(row, p, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    term = np.sqrt(u) if branch == 'sqrt' else u
    return base + c * term

def fit(train_rows):
    global _params, _branch
    y = np.array [float(r['y']) for r in train_rows]
    best = None
    for branch in ('sqrt', 'linear'):
        def residual(p):
            return np.array([_value(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, [5.0, 0.7, 0.0], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]), max_nfev=5000)
        error = np.dot(result.fun, result.fun)
        if best is None or error < best[0]:
            best = (error, result.x, branch)
    _, _params, _branch = best

def predict(row):
    return float(_value(row, _params, _branch))