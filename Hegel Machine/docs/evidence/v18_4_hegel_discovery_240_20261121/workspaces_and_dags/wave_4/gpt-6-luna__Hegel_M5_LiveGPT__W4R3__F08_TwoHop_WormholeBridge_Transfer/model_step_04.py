import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau_scale is 1.0
    return un / (1.2 + un), s / (1.0 + s), math.sqrt(max(un, 0.0))

def _model(p, row):
    sat, sink, root = _features(row)
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * sink) * sat + c_loc * root

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.5, 0.3]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(_params, row))