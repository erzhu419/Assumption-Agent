import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    z2 = np.square(np.array([float(r['z_2']) for r in rows]))
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def residual(p):
        vmax, k, c = p
        return vmax * z2 / (k + z2) - c * u - y

    vmax, k, c = least_squares(residual, [8.65, 1.24, 0.30], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf])).x

    def predict(row) -> float:
        s = math.pow(float(row['z_2']), 2)
        return float(vmax * s / (k + s) - c * float(row['u_ctrl']))

    return predict
