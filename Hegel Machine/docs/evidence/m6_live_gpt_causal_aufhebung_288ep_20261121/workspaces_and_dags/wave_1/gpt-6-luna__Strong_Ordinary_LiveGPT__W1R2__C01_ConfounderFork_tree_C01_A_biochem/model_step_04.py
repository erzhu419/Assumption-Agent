import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    vmax, k_half, c_ctrl = p
    z = float(row['z_2'])
    u = float(row['u_ctrl'])
    q = z * z
    return vmax * q / (k_half + q) - c_ctrl * u

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[8.0, 1.0, 0.2], bounds=([0.0, 1e-9, -20.0], [100.0, 100.0, 20.0]))
    _params = result.x
    return {'vmax': float(_params[0]), 'k_half': float(_params[1]), 'c_ctrl': float(_params[2])}

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))