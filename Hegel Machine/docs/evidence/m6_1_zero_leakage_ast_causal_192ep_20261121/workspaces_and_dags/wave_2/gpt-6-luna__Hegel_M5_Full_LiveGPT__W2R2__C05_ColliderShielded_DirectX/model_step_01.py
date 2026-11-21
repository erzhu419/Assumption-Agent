import numpy as np
from scipy.optimize import least_squares

_fit_parameters = None

def fit(train_rows):
    global _fit_parameters
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    basis_x = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * basis_x + p[1] * u - y,
                           x0=np.array([8.0, 0.5]))
    _fit_parameters = result.x
    return _fit_parameters

def predict(row):
    if _fit_parameters is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, c_ctrl = _fit_parameters
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)