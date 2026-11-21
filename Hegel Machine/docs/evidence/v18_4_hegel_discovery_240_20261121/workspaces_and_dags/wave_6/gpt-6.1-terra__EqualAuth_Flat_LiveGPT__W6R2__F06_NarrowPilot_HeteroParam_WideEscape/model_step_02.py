import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [7.0, 0.2, 1.0, 0.2, 2.58], bounds=([0.0, 0.0, -10.0, 0.0, 2.50], [100.0, 2.0, 20.0, 10.0, 2.68]), max_nfev=10000)
    _params = result.x

def predict(row):
    return float(_model(row, _params))