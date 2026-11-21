import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['m_2']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        base, vmax, k = p
        return base + vmax * x / (k + x) - y
    result = least_squares(residual, x0=[0.0, 8.0, 1.5], bounds=([-20.0, 0.0, 1e-6], [20.0, 30.0, 100.0]))
    return result.x.tolist()

def predict(row):
    # Parameters are attached by the evaluation harness after fit.
    base, vmax, k = _MODEL_PARAMS
    x = float(row['m_2'])
    return float(base + vmax * x / (k + x))