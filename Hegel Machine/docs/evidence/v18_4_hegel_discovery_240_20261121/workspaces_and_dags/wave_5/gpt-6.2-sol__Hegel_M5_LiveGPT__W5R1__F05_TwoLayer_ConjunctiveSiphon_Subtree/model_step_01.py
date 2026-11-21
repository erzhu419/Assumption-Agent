import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), np.sqrt(u)

def _law(p, a, g, q):
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * g) * a + c_loc * q

def fit(train_rows):
    global _params
    rows = list(train_rows)
    a, g, q = np.array([_features(row) for row in rows], dtype=float).T
    y = np.array([float(row['y']) for row in rows])
    _params = least_squares(lambda p: _law(p, a, g, q) - y,
                            x0=[6.5, 0.70, 0.0],
                            bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row):
    return float(_law(_params, *_features(row)))
