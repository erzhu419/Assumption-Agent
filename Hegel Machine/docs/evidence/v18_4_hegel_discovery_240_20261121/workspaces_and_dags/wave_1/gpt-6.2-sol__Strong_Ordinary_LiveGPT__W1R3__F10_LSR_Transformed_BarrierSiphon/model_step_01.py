import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global params
    rows = list(train_rows)
    u, s, y = np.array([(r['u_now'] / 3.0, r['s_grad'], r['y']) for r in rows], dtype=float).T
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - y

    params = least_squares(residual, [3.0, 1.0, 0.7, 0.2], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf])).x


def predict(row) -> float:
    j0, beta, rho, c = params
    u = row['u_now'] / 3.0
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))