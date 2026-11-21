import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global params
    u, p, h, y = np.array([[r['u_now']/5.0, r['u_past']/5.0, r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(q):
        gain, rate, k_eq = q
        return gain * (u - p) * np.exp(-rate * h) + k_eq * u - y
    params = least_squares(residual, [1.0, 1.0, 0.3], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x


def predict(row) -> float:
    gain, rate, k_eq = params
    u = row['u_now'] / 5.0
    p = row['u_past'] / 5.0
    return float(gain * (u - p) * math.exp(-rate * row['H']) + k_eq * u)
