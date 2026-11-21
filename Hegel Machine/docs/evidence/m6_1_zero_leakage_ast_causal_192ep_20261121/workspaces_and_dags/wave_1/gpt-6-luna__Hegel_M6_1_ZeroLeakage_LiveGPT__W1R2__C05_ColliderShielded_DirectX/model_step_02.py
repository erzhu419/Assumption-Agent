import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    x = float(row.get('x_ind', 0.0))
    return np.array([x / (1.1 + x), float(row.get('u_ctrl', 0.0)), float(row.get('z_1', 0.0)), float(row.get('z_2', 0.0)), 1.0], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Least-squares fit of direct saturating x response with additive control and measured covariates.
    result = least_squares(lambda b: X @ b - y, np.zeros(X.shape[1]))
    return result.x

def predict(row):
    return float(_features(row) @ _MODEL_COEFFICIENTS)
