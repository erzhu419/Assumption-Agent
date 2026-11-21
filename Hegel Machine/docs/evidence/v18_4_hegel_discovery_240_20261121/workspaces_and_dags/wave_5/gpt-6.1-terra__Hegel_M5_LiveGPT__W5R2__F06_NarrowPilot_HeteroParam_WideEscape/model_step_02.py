import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    result = least_squares(
        lambda p: np.array([_model(r, p) - float(r['y']) for r in rows]),
        x0=[6.5, 0.5, 0.0],
        bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]),
        ftol=1e-12, xtol=1e-12, gtol=1e-12
    )
    _params = result.x

def predict(row):
    return float(_model(row, _params))