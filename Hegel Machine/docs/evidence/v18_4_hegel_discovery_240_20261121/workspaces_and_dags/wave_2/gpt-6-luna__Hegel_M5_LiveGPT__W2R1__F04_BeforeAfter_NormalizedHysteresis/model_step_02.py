import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    d = np.array([r['u_past'] - r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        alpha, amp, rate = p
        return alpha*x + amp*d*np.exp(-rate*h) - y
    result = least_squares(residual, [1.0, 1.9, 1.5], bounds=([-10.0, -20.0, 0.0], [10.0, 20.0, 20.0]))
    return result.x

def predict(row):
    alpha, amp, rate = fit(_train_rows)
    return float(alpha*row['x_now'] + amp*(row['u_past']-row['u_now'])*np.exp(-rate*row['H']))