import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, tau, branch = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    un = u / max(1e-8, tau)
    base = vmax * (1.0 - rho * s / (1.0 + s)) * un / (1.2 + un)
    # A second, conjunctively gated branch contributes most at high substrate
    # and is attenuated by the gradient.
    gate_u = u / (2.0 + u)
    gate_s = 1.0 / (1.0 + s)
    return base + branch * gate_u * gate_s

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows], dtype=float) - ys
    result = least_squares(residual, x0=np.array([7.0, 0.7, 4.0, 2.0]),
                           bounds=([0.0, 0.0, 0.05, 0.0], [30.0, 0.999, 100.0, 30.0]),
                           max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))