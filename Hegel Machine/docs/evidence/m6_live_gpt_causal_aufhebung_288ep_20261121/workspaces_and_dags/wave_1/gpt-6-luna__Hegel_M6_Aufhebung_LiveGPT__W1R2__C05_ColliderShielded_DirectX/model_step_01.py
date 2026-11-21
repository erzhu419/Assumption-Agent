import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    basis = x / (1.1 + x)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * basis + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([6.0, 0.5]))
    fit.params_ = tuple(float(v) for v in result.x)
    return fit

def predict(row):
    vmax, c_ctrl = fit.params_
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)