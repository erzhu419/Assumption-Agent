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
    result = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    vmax, km, c_raw = _params
    return float(vmax * row['m_2'] / (km + row['m_2']) + c_raw * np.sqrt(row['u_raw']))