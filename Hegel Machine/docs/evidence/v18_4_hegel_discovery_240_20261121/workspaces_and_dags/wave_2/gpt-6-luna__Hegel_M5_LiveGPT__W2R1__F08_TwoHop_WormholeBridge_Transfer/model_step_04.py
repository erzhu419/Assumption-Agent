import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(float(row['u_now']), 1e-9)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * np.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 0.7, 0.2], [3.0, 0.2, 0.5], [5.0, 0.9, -0.2]):
        res = least_squares(residual, np.asarray(start, dtype=float), bounds=([-np.inf, -5.0, -np.inf], [np.inf, 5.0, np.inf]))
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _PARAMS))