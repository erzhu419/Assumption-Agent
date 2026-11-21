import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p, branch):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic_tau_scale = 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return base + branch * c * np.sqrt(max(un, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for branch in (-1.0, 1.0):
        def residual(p):
            return np.array([_predict_params(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, x0=np.array([4.0, 0.7, 0.5]), max_nfev=20000)
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x, branch)
    _PARAMS = (best[1], best[2])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    p, branch = _PARAMS
    return float(_predict_params(row, p, branch))
