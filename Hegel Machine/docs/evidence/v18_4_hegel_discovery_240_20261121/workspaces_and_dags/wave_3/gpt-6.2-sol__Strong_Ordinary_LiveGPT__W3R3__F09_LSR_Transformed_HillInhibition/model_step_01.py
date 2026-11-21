import numpy as np
from scipy.optimize import least_squares


def _law(p, q, s):
    vmax, k_half, alpha, leak = p
    return vmax * q**2 / ((k_half + q**2) * (1.0 + alpha * s)) - leak * q


def fit(train_rows):
    global _params
    data = np.array([(r['u_now'] / 0.1, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    q, s, y = data.T
    _params = least_squares(lambda p: _law(p, q, s) - y, [8.0, 2.0, 1.0, 0.25], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf)).x


def predict(row) -> float:
    return float(_law(_params, row['u_now'] / 0.1, row['s_grad']))
