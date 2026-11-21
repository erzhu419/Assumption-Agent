import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    starts = [[4, 1, .2, .05], [6, 2, .5, .1], [3, .5, .1, .01], [10, 4, 1, .2]]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: np.array([_model(r,p) for r in rows])-y,
                            p0, bounds=([1e-8,1e-8,0,0],[100,100,100,100]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _params))