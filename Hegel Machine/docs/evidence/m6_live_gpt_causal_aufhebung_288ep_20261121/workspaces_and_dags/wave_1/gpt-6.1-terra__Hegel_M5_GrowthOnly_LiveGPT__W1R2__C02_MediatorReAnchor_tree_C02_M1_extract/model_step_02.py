import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.asarray([r['m_2'] for r in train_rows], dtype=float)
    u = np.asarray([r['u_raw'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, km, c_raw = p
        return vmax * m / (km + m) + c_raw * np.sqrt(u) - y
    result = least_squares(residual, [8.3, 1.35, 0.32], bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row) -> float:
    vmax, km, c_raw = _params
    m = float(row['m_2'])
    return float(vmax * m / (km + m) + c_raw * np.sqrt(float(row['u_raw'])))