import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(params):
        c0, beta = params
        return c0 * u * (1.0 + beta * np.cbrt(H)) - y

    result = least_squares(residual, x0=np.array([1.0, 0.1]))
    return result.x


def predict(row):
    c0, beta = fit(train_rows)
    return float(c0 * row['u_now'] * (1.0 + beta * (row['H'] ** (1.0 / 3.0))))