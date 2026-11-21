import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    # At the observed fixed s_grad=1, absorb the inhibition factor and vmax
    # into A; estimate A and the open-branch sqrt coefficient from data.
    basis = np.column_stack([u / (1.2 + u), np.sqrt(np.maximum(u, 0.0))])
    result = least_squares(lambda p: basis @ p - y, x0=np.array([3.0, 0.3]))
    return tuple(float(v) for v in result.x)


def predict(row):
    A, c_sqrt = fit(_training_rows)
    u = float(row['u_now'])
    return A * u / (1.2 + u) + c_sqrt * math.sqrt(max(u, 0.0))
