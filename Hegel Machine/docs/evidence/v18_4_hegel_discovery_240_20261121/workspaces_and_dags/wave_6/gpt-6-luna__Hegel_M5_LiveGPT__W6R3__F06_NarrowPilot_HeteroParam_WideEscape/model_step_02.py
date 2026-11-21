import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    u = np.array([_features(r)[0] for r in rows])
    s = np.array([_features(r)[1] for r in rows])
    sat = u / (1.2 + u)
    grad = s / (1.0 + s)
    root = np.sqrt(np.maximum(u, 0.0))
    def residual(p):
        vmax, rho, c_loc = p
        pred = vmax * (1.0 - rho * grad) * sat + c_loc * root
        return pred - y
    result = least_squares(residual, x0=np.array([6.0, 0.7, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    fit.params_ = result.x
    return fit

def predict(row):
    vmax, rho, c_loc = fit.params_
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(max(u, 0.0)))