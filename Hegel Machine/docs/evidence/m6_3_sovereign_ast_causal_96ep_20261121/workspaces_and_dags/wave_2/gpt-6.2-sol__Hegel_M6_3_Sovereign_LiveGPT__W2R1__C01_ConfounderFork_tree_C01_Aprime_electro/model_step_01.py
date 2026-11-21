import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    z2_sq = np.array([float(row['z_2']) ** 2 for row in rows])
    u = np.array([float(row['u_ctrl']) for row in rows])
    y = np.array([float(row['y']) for row in rows])

    def residual(p):
        vmax, k, c = p
        return vmax * z2_sq / (k + z2_sq) - c * u - y

    vmax, k, c = least_squares(
        residual, [8.02, 1.44, 0.22], bounds=([0.0, 1e-12, 0.0], math.inf)
    ).x

    def predict(row) -> float:
        s = float(row['z_2']) ** 2
        return float(vmax * s / (k + s) - c * float(row['u_ctrl']))

    return predict