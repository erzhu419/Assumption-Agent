import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    core = (1.0 - _PARAMS[1] * s / (1.0 + s)) * u / (1.2 + u)
    return core

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, rho, bx, bm = p
        preds = []
        for r in train_rows:
            u = max(float(r['u_now']), 0.0)
            s = float(r['s_grad'])
            core = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            preds.append(core + bx * float(r['x_now']) + bm * float(r['mem_conv']))
        return np.asarray(preds) - y
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.0, 0.0]), bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, 1.0, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    vmax, rho, bx, bm = _PARAMS
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + bx * float(row['x_now']) + bm * float(row['mem_conv']))