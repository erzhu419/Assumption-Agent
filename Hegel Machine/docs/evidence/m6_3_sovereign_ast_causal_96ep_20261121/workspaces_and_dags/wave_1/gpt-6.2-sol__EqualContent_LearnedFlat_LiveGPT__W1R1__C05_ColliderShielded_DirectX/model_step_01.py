import numpy as np
from scipy.optimize import least_squares


def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])


def fit(train_rows):
    global _coefficients
    rows = list(train_rows)
    design = np.array([_features(row) for row in rows], dtype=float)
    target = np.array([float(row['y']) for row in rows], dtype=float)
    _coefficients = least_squares(lambda p: design @ p - target, x0=[7.9, 0.385], jac=lambda p: design).x
    return predict


def predict(row) -> float:
    return float(np.dot(_coefficients, _features(row)))
