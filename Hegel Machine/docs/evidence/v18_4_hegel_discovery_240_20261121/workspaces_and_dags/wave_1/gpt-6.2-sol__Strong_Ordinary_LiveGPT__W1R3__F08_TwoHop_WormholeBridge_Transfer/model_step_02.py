import math
import numpy as np
from scipy.optimize import least_squares


def features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return np.array([base, -base * s / (1.0 + s), math.sqrt(u)])


def fit(train_rows):
    global params
    rows = list(train_rows)
    X = np.array([features(row) for row in rows])
    y = np.array([float(row['y']) for row in rows])
    # Fit vmax, vmax*rho, and the square-root branch coefficient.
    params = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x


def predict(row) -> float:
    return float(features(row) @ params)
