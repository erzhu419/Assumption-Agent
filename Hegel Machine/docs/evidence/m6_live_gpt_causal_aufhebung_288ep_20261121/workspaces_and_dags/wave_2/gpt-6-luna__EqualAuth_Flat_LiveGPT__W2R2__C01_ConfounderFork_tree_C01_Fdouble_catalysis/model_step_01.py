import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r.get('u_ctrl', 0.0)) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)

    def predict_params(p):
        vmax, k_half, c_ctrl = p
        q = (x / 1.5) ** 2
        return vmax * q / (k_half + q) - c_ctrl * u

    result = least_squares(
        lambda p: predict_params(p) - y,
        x0=np.array([8.0, 2.0, 0.1]),
        bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]),
    )
    vmax, k_half, c_ctrl = result.x

    def predict(row):
        q = (float(row['x_ind']) / 1.5) ** 2
        return float(vmax * q / (k_half + q) - c_ctrl * float(row.get('u_ctrl', 0.0)))

    return predict


def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')