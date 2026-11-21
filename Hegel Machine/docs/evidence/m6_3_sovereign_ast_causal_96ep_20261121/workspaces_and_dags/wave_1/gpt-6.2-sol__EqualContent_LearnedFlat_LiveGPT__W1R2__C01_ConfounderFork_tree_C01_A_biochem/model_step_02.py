import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    z, u, y = np.array([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float).T
    q = np.square(z)
    result = least_squares(
        lambda p: p[0] * q / (p[1] + q) - p[2] * u - y,
        x0=[8.02, 1.44, 0.22],
        bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]),
    )
    vmax, k_half, c_ctrl = map(float, result.x)

    def predict(row) -> float:
        q = math.pow(float(row['z_2']), 2)
        return vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl'])

    return predict
