import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    z_sq = np.array([float(row['z_2']) ** 2 for row in rows])
    u = np.array([float(row['u_ctrl']) for row in rows])
    y = np.array([float(row['y']) for row in rows])

    def residuals(params):
        vmax, k_half, c_ctrl = params
        return vmax * z_sq / (k_half + z_sq) - c_ctrl * u - y

    vmax, k_half, c_ctrl = least_squares(
        residuals, [8.7, 1.3, 0.3],
        bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]),
    ).x

    def predict(row) -> float:
        z_sq = math.pow(float(row['z_2']), 2)
        return float(vmax * z_sq / (k_half + z_sq) - c_ctrl * float(row['u_ctrl']))

    return predict
