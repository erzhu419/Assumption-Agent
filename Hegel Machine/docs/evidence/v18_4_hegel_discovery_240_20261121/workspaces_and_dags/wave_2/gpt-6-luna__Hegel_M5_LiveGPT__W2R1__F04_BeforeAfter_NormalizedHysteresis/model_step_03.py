import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X = np.array([r['x_now'] for r in train_rows], dtype=float)
    D = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        alpha, amp, rate = p
        return alpha * X - amp * D * np.exp(-rate * H) - y
    p0 = [1.0, 1.0, 1.0]
    result = least_squares(residual, p0, bounds=([-10.0, -10.0, 0.0], [10.0, 10.0, 20.0]))
    return result.x

def predict(row):
    alpha, amp, rate = fit(_train_rows)
    return float(alpha * row['x_now'] - amp * (row['u_now'] - row['u_past']) * math.exp(-rate * row['H']))