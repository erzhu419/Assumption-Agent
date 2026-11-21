import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array [float(r['u_now']) for r in train_rows]
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, km, rho = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) - y
    result = least_squares(residual, [9.0, 1.2, 0.2], bounds=([0.0, 1e-8, -2.0], [np.inf, np.inf, 1.0]))
    _params = result.x

def predict(row):
    vmax, km, rho = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u))