import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z = np.asarray([float(r['z_2']) for r in train_rows])
    u = np.asarray([float(r['u_ctrl']) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z**2 / (k_half + z**2) - c_ctrl * u - y

    result = least_squares(residual, x0=[8.0, 1.5, 0.2], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, k_half, c_ctrl = _params
    z = float(row['z_2'])
    u = float(row['u_ctrl'])
    return vmax * z**2 / (k_half + z**2) - c_ctrl * u