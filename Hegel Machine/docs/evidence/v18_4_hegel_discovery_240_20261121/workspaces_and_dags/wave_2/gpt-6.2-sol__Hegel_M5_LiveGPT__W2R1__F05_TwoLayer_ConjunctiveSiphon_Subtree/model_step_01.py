import numpy as np
from scipy.optimize import least_squares


def _law(p, u, s):
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u)


def fit(train_rows):
    global _params
    data = np.array([(r['u_now'] / 0.05, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    _params = least_squares(lambda p: _law(p, u, s) - y, [6.5, 0.70, 0.14], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x


def predict(row):
    return float(_law(_params, row['u_now'] / 0.05, row['s_grad']))
