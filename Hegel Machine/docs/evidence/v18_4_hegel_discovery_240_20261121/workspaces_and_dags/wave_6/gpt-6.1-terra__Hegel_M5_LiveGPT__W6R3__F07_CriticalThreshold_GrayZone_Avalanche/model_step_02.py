import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    vmax, rho, c, k, scrit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u) - k * np.maximum(0.0, s - scrit) * u

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _values(p, u, s) - y, [6.5, 0.7, 0.3, 2.0, 2.6], bounds=([1.0, 0.05, -2.0, 0.0, 2.44], [15.0, 0.95, 2.0, 20.0, 2.78]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad'])))