import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_now']) - float(r['u_past']), float(r['H'])] for r in rows], dtype=float)
    y = np.array([float(r['y']) - float(r['x_now']) for r in rows], dtype=float)
    def residual(p):
        a, tau, b = p
        tau = max(abs(tau), 1e-8)
        return X[:, 0] * (a * np.exp(-X[:, 1] / tau) + b) - y
    starts = [[3.0, 1.0, 0.0], [2.0, 0.6, 0.0], [4.0, 2.0, 0.2]]
    sols = [least_squares(residual, p, max_nfev=5000) for p in starts]
    p = min(sols, key=lambda s: float(np.sum(s.fun ** 2))).x
    return tuple(float(v) for v in p)

def predict(row):
    a, tau, b = fit(_MODEL_TRAIN_ROWS)
    tau = max(abs(tau), 1e-8)
    return float(row['x_now']) + (float(row['u_now']) - float(row['u_past'])) * (a * math.exp(-float(row['H']) / tau) + b)
