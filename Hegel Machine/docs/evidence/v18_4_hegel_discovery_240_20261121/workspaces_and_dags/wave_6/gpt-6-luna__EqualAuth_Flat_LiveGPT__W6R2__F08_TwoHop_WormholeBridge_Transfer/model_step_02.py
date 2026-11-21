import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def model(p, u, s):
        a, b, c, d = p
        return ((a + b * u) / (1.0 + c * u)) * np.exp(-d * (s - 1.0))

    result = least_squares(
        lambda p: model(p, u, s) - y,
        x0=np.array([0.5, 2.0, 0.5, 0.5]),
        bounds=([-100.0, -100.0, 1e-8, -10.0], [100.0, 100.0, 100.0, 10.0]),
        max_nfev=20000,
    )
    return result.x


def predict(row):
    p = fit(_MODEL_TRAIN_ROWS)
    a, b, c, d = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(((a + b * u) / (1.0 + c * u)) * math.exp(-d * (s - 1.0)))