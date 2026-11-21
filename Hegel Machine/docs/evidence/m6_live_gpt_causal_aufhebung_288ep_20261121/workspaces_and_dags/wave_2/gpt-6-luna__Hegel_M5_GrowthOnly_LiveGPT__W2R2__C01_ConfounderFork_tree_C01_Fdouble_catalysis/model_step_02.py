import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float) / 1.5
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def residual(p):
        vmax, k_half, c_ctrl = p
        frac = x**2 / (k_half + x**2)
        return vmax * frac - c_ctrl * u - y

    result = least_squares(residual, x0=[8.0, 2.0, 0.2], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, k_half, c_ctrl = _params
    x = float(row['x_ind']) / 1.5
    u = float(row['u_ctrl'])
    return float(vmax * (x*x) / (k_half + x*x) - c_ctrl * u)
