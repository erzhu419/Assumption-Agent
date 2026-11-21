import math
import numpy as np
from scipy.optimize import least_squares


def features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    flux = u / (1.2 + u)
    return flux, -flux * s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global params
    rows = list(train_rows)
    X = np.array([features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    # Coefficients are vmax, vmax*rho, and the signed local branch amplitude.
    params = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x


def predict(row):
    return float(np.dot(features(row), params))
