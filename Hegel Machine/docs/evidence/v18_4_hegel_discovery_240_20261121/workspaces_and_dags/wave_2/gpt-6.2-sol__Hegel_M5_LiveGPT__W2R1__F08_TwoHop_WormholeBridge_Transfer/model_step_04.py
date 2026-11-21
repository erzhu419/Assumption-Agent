import math
import numpy as np
from scipy.optimize import least_squares


def _law(u, s, root_u, p):
    amplitude, half_sat, attenuation, loss = p
    return amplitude * u / (half_sat + u) * (1.0 - attenuation * s / (1.0 + s)) - loss * root_u


def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    _params = least_squares(lambda p: _law(u, s, root_u, p) - y, [6.0, 1.0, 0.5, 0.3], bounds=([0.0, 1e-8, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf])).x


def predict(row) -> float:
    u = float(row['u_now'])
    return float(_law(u, float(row['s_grad']), math.sqrt(u), _params))