import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    def residual(p):
        return np.array([_pred(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5, 1.0]), bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_pred(_params, row))

_params = None
_original_fit = fit
def fit(train_rows):
    global _params
    _params = _original_fit(train_rows)
    return _params
