import math
import numpy as np
from scipy.optimize import least_squares


def features(row):
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global params
    X = np.asarray([features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    params = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x


def predict(row):
    return float(np.dot(features(row), params))
