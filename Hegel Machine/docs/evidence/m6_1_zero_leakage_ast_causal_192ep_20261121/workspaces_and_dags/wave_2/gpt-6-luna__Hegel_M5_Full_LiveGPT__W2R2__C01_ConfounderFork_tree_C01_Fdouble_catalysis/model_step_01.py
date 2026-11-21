import math
import numpy as np
from scipy.optimize import least_squares

_fit_parameters = None

def fit(train_rows):
    global _fit_parameters
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Signature and passive analog suggest Hill saturation, while the pilot
    # distinguishes an additive control term from a direct x-only response.
    def residual(p):
        vmax, k, c = p
        return vmax * x**2 / (k + x**2) + c * u - y
    result = least_squares(residual, x0=np.array([8.0, 4.0, 0.3]), bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _fit_parameters = result.x
    return _fit_parameters

def predict(row):
    if _fit_parameters is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, k, c = _fit_parameters
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x*x / (k + x*x) + c*u)