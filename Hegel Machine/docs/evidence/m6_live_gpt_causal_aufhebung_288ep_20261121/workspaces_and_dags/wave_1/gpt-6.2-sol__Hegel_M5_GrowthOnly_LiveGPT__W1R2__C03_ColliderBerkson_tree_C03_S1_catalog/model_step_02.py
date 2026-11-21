import math
import numpy as np
from scipy.optimize import least_squares


def features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return math.log1p(u1), u2 / (1.0 + u2)


def fit(train_rows):
    global coefficients
    rows = list(train_rows)
    x = np.array([features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    coefficients = least_squares(lambda p: x @ p - y, np.zeros(2), jac=lambda p: x).x
    return coefficients


def predict(row) -> float:
    return float(np.dot(coefficients, features(row)))
