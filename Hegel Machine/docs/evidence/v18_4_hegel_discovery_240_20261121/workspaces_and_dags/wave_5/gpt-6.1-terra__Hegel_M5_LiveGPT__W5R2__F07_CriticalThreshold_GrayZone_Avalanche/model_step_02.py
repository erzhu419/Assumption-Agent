import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [6.5, 0.7, 0.3, 1.5, 2.6], bounds=([1.0, 0.05, -2.0, 0.0, 2.5], [15.0, 0.95, 2.0, 20.0, 2.68]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_model(row, _params))