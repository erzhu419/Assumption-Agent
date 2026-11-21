import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(max(u, 0.0))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.2, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
    return result.x

def predict(row):
    return float(_value(row, _fitted_params))

_fitted_params = None
_train_rows = None
# The evaluation harness fits the supplied training rows before prediction; this
# wrapper stores parameters when fit is called while preserving the required API.
def fit(train_rows):
    global _fitted_params, _train_rows
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.2, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
    _fitted_params = result.x
    _train_rows = rows
    return result.x